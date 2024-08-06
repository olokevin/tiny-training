import torch
import torch.nn.functional as F
from .quantized_ops import to_pt, QuantizedAvgPool, QuantizedConv2d, QuantizedElementwise, QuantizedMbBlock
from core.utils.config import configs
from core.builder.frn import FilterResponseNorm2d

QUANTIZED_GRADIENT = False
ROUNDING = 'round'
CONV_W_GRAD = True

# sanity check
if QUANTIZED_GRADIENT:
    raise NotImplementedError


def round_tensor(x):
    if ROUNDING == 'round':
        return x.round()
    elif ROUNDING == 'floor':
        return x.int().float()
    elif ROUNDING == 'debug':
        return x
    else:
        raise NotImplementedError

def min_divisible_value(n1, v1):
    """make sure v1 is divisible by n1, otherwise decrease v1"""
    if v1 >= n1:
        return n1
    while n1 % v1 != 0:
        v1 -= 1
    return v1


class _QuantizedAvgPoolFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.input_shape = x.shape
        assert x.dtype == torch.float32
        x = x.mean([-1, -2], keepdim=True)
        return round_tensor(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_shape = ctx.input_shape
        grad_input = grad_output.repeat(1, 1, *input_shape[-2:]) / (input_shape[-1] * input_shape[-2])
        return grad_input


class QuantizedAvgPoolDiff(QuantizedAvgPool):
    def forward(self, x):
        x = _QuantizedAvgPoolFunc.apply(x)
        return x


class _QuantizedElementwiseAddFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, zero_x1, zero_x2, zero_y, scale_x1, scale_x2, scale_y):
        # ensure x1 and x2 are int
        x1 = x1.round()  
        x2 = x2.round()
        assert x1.shape == x2.shape
        ctx.save_for_backward(scale_x1, scale_x2, scale_y)

        x1 = (x1 - zero_x1) * scale_x1
        x2 = (x2 - zero_x2) * scale_x2

        out = x1 + x2

        ### OLD
        # out = round_tensor(out / scale_y)
        # out = out + zero_y

        return out

    @staticmethod
    def backward(ctx, grad_output):
        # only return gradient of zero_y, zero_x1, zero_x2, x1, x2
        scale_x1, scale_x2, scale_y = ctx.saved_tensors

        ### OLD
        # grad_sum = grad_output / scale_y.item()

        grad_sum = grad_output
        grad_x1 = grad_sum * scale_x1.item()
        grad_x2 = grad_sum * scale_x2.item()
        grad_zero_x1 = - grad_x1.sum([0, 2, 3]).view(1,-1,1,1)
        grad_zero_x2 = - grad_x2.sum([0, 2, 3]).view(1,-1,1,1)
        return grad_x1, grad_x2, grad_zero_x1, grad_zero_x2, None, None, None, None


class QuantizedElementwiseDiff(QuantizedElementwise):
    def __init__(self, operator, zero_x1, zero_x2, zero_y, scale_x1, scale_x2, scale_y):
        super().__init__(operator, zero_x1, zero_x2, zero_y, scale_x1, scale_x2, scale_y)
        assert self.operator == 'add'  # for mult, we do not support bias-only update

        # if configs.train_config.train_scale:
        #     # print('Note: the scale is also trained...')
        #     self.register_parameter('scale_y', torch.nn.Parameter(scale_y))
        # else:
        #     self.register_buffer('scale_y', scale_y)
        
        # if configs.train_config.train_zero:
        #     # print('Note: the scale is also trained...')
        #     self.register_parameter('zero_y', torch.nn.Parameter(zero_y))
        # else:
        #     self.register_buffer('zero_y', zero_y)

    def forward(self, x1, x2):
        return _QuantizedElementwiseAddFunc.apply(x1, x2,
                                                  self.zero_x1, self.zero_x2, self.zero_y,
                                                  self.scale_x1, self.scale_x2, self.scale_y)
    
    def update_quantize_params(self, signSGD=False, param_lr=None): 
        if self.scale_y.grad is not None:
            scale_y_lr = param_lr['scale_y']
            if signSGD:
                self.scale_y.data -= scale_y_lr * torch.sign(self.scale_y.grad)
            else:
                self.scale_y.data -= scale_y_lr * self.scale_y.grad
            
            self.scale_y.grad = None
        
        if self.zero_y.grad is not None:
            zero_y_lr = param_lr['zero_y']
            if signSGD:
                self.zero_y.data -= zero_y_lr * torch.sign(self.zero_y.grad)
            else:
                self.zero_y.data -= zero_y_lr * self.zero_y.grad
            
            self.zero_y.grad = None

class _QuantizedConv2dFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, zero_x, effective_scale, stride, padding, dilation, groups, grad_output_prune_ratio):
        x = x.round()  # ensure x is int
        weight = weight.round()  # ensure weight is int

        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.input_size = x.shape
        ctx.weight_size = weight.shape
        ctx.grad_output_prune_ratio = grad_output_prune_ratio

        # weight = weight.int()  # - self.zero_w
        x = x - zero_x
        
        out = F.conv2d(x, weight, None, stride, padding, dilation, groups)
        out = round_tensor(out)  # ensure output is still int
        # here we allow bias saved as fp32, and round to int during inference (keep fp32 copy in memory)
        out = out + bias.view(1, -1, 1, 1)  # Confirmed: we don't need to cast bias
        out = out * effective_scale.view(1, -1, 1, 1)
        
        # out = round_tensor(out)
        # out = out + zero_y
        
        if CONV_W_GRAD:
            if grad_output_prune_ratio is not None:
                topk_mask = torch.zeros_like(out, dtype=torch.bool)

                ### Output actv magnitude top-k sparsity
                # topk_dim = int((1.0-grad_output_prune_ratio) * out.numel())
                # _, indices = torch.topk(out.flatten(), topk_dim)
                # topk_mask.view(-1)[indices] = True

                ### Output actv magnitude top-k sparsity, batch-wise
                # batch_sz = out.shape[0]
                # topk_dim = int((1.0-grad_output_prune_ratio) * (out.numel() / batch_sz))
                # for b in range(batch_sz):
                #     _, indices = torch.topk(out[b].flatten(), topk_dim)
                #     topk_mask[b].view(-1)[indices] = True
                
                ### Output actv magnitude top-k sparsity, channel-wise
                batch_sz = out.shape[0]
                topk_dim = int((1.0-grad_output_prune_ratio) * (out.size(1)))
                for b in range(batch_sz):
                    _, indices = torch.topk(torch.linalg.norm((out[b]), dim=(1,2)), topk_dim)
                    topk_mask[b,indices,:,:] = True

                ctx.save_for_backward(weight, effective_scale, x, topk_mask)
            else:
                ctx.save_for_backward(weight, effective_scale, x)
        else:
            ctx.save_for_backward(weight, effective_scale)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        # effective_scale = scale_x * scale_w / scale_y
        # b_quantized = b / (w_scales * x_scale), so we may wanna compute grad_b / (w_scale * x_scale)
        # which is grad_b / (effective_scale * scale_y)
        
        if CONV_W_GRAD:
            if ctx.grad_output_prune_ratio is not None:
                weight, effective_scale, _x, topk_mask = ctx.saved_tensors
            else:
                weight, effective_scale, _x = ctx.saved_tensors
        else:
            weight, effective_scale = ctx.saved_tensors       

        _grad_conv_out = grad_output * effective_scale.view(1, -1, 1, 1)
        # _grad_conv_out = grad_output
        grad_bias = _grad_conv_out.sum([0, 2, 3])
        _grad_conv_in = torch.nn.grad.conv2d_input(ctx.input_size, weight, _grad_conv_out,
                                                   stride=ctx.stride, padding=ctx.padding,
                                                   dilation=ctx.dilation, groups=ctx.groups)
        grad_zero_x = - _grad_conv_in.sum([0, 2, 3])
        grad_x = _grad_conv_in

        if CONV_W_GRAD:
            if ctx.grad_output_prune_ratio is not None:
                _grad_conv_out = _grad_conv_out * topk_mask
                
            grad_w = torch.nn.grad.conv2d_weight(_x, ctx.weight_size, _grad_conv_out,
                                                 stride=ctx.stride, padding=ctx.padding,
                                                 dilation=ctx.dilation, groups=ctx.groups)
        else:
            grad_w = None
        
        if configs.backward_config.quantize_gradient:  # perform per-channel quantization
            # quantize grad_x and grad_w
            from .quantize_helper import get_weight_scales
            w_scales = get_weight_scales(grad_w, n_bit=8)
            grad_w = (grad_w / w_scales.view(-1, 1, 1, 1)).round() * w_scales.view(-1, 1, 1, 1)
            x_scales = get_weight_scales(grad_x.transpose(0, 1))
            grad_x = (grad_x / x_scales.view(1, -1, 1, 1)).round() * x_scales.view(1, -1, 1, 1)

        return grad_x, grad_w, grad_bias, None, None, None, None, None, None, None
        # return grad_x, grad_w, grad_bias, None, None, None, None, None, None, None

class _QuantizedNormalization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, gamma, beta, a_bit, normalization_func=None):
        ctx.normalization_func = normalization_func
        if normalization_func is None:
            pass
        elif normalization_func == 'SSF':
            ctx.save_for_backward(y,)
            y = y * gamma.view(1,-1,1,1) + beta.view(1,-1,1,1)
        # elif normalization_func == 'L1FRN':
        #     v = torch.mean(y.abs(), dim=[-1, -2], keepdim=True)
        #     # epsilon = torch.ones(1, y.size(1), 1, 1).cuda()
        #     epsilon = torch.tensor(1e-5).cuda()
        #     eta = 1 / (v+epsilon)
        #     y_hat = y * eta
        #     y = y_hat * gamma.view(1,-1,1,1) + beta.view(1,-1,1,1)

        #     ctx.save_for_backward(gamma, eta, y_hat)
        else:
            raise NotImplementedError('Normalization function not implemented')
        
        return y

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.normalization_func is None:
            # STE estimator
            return grad_output, None, None, None, None
        elif ctx.normalization_func == 'SSF':
            y, = ctx.saved_tensors
            grad_gamma = (grad_output * y).sum([0, 2, 3])
            grad_beta = grad_output.sum([0, 2, 3])
            return grad_output, grad_gamma, grad_beta, None, None
        # elif ctx.normalization_func == 'L1FRN':
        #     # grad_output: dL/dy_{fp}
        #     gamma, eta, y_hat = ctx.saved_tensors

        #     grad_beta = grad_output.sum([0, 2, 3])
        #     grad_gamma = (grad_output * y_hat).sum([0, 2, 3])

        #     mean = (grad_gamma / grad_output.size(-2) / grad_output.size(-1)).view(1,-1,1,1)
        #     gamma_eta = (gamma.view(1,-1) * eta.squeeze()).unsqueeze(-1).unsqueeze(-1)
        #     grad_y = gamma_eta * (grad_output - torch.sign(y_hat) * mean)
        #     return grad_y, grad_gamma, grad_beta, None, None
        else:
            return grad_output, None, None, None, None

class _TruncateActivationRange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, scale_y, zero_y, a_bit, activation_func):
        ctx.a_bit = a_bit
        ctx.activation_func = activation_func

        if activation_func is None:
            y = round_tensor(y / scale_y)
            out = y + zero_y
            # STE mask
            binary_mask = (- 2 ** (a_bit - 1) <= out) & (out <= 2 ** (a_bit - 1) - 1)
            # PACT mask for grad_scale_y
            PACT_mask = (y > 2 ** (a_bit - 1) - 1)
            ctx.save_for_backward(scale_y, binary_mask, PACT_mask)
            out = out.clamp(- 2 ** (a_bit - 1), 2 ** (a_bit - 1) - 1)

        elif activation_func == 'TLU':
            y = round_tensor(y / scale_y)
            upper_bound = 2 ** a_bit - 1
            # upper_bound = 2 ** (a_bit - 1) - 1
            binary_mask = (y <= upper_bound)
            PACT_mask = (y > upper_bound)
            y = y.clamp(max=upper_bound)
            out = y + zero_y
            ctx.save_for_backward(scale_y, binary_mask, PACT_mask)

            ### self desgined TLU
            # # tau: initializaed as ReLU
            # tau = - zero_y - 2 ** (a_bit - 1)
            # lower_bound = tau.view(1,-1,1,1)
            # upper_bound = 2 ** a_bit - 1
            # upper_bound = 2 ** (a_bit - 1) - 1
            # upper_bound = tau.view(1,-1,1,1) + 2 ** a_bit - 1
            # binary_mask = (lower_bound <= y) & (y <= upper_bound)
            # PACT_mask = (y > upper_bound)
            # y = y.clamp(lower_bound, upper_bound)
            # out = y + zero_y
            # ctx.save_for_backward(scale_y, binary_mask, PACT_mask)
        
        elif activation_func == 'LSQ':
            y = round_tensor(y / scale_y)
            out = y + zero_y
            binary_mask = (- 2 ** (a_bit - 1) <= out) & (out <= 2 ** (a_bit - 1) - 1)
            pos_mask = (out > 2 ** (a_bit - 1) - 1)
            neg_mask = (out < - 2 ** (a_bit - 1))
            
            out = y.clamp(- 2 ** (a_bit - 1), 2 ** (a_bit - 1) - 1)
            ctx.save_for_backward(scale_y, y, out, binary_mask, pos_mask, neg_mask)
        
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a_bit = ctx.a_bit
        activation_func = ctx.activation_func
        if activation_func is None:
            scale_y, binary_mask, PACT_mask = ctx.saved_tensors
            grad_y = (grad_output / scale_y) * binary_mask
            grad_scale_y = (grad_output * PACT_mask).sum()
            grad_zero_y = (grad_y.sum([0, 2, 3])).view(1,-1,1,1)
            return grad_y, grad_scale_y, grad_zero_y, None, None 
        
        elif activation_func == 'TLU':
            scale_y, binary_mask, PACT_mask = ctx.saved_tensors
            # grad_output: dL/dy_{int}; grad_y: dL.dy_{fp}
            grad_y =  (grad_output / scale_y) * binary_mask
            grad_scale_y = (grad_output * PACT_mask).sum()
            grad_zero_y = -((grad_output * (~binary_mask)).sum([0, 2, 3]).view(1,-1,1,1))
            return grad_y, grad_scale_y, grad_zero_y, None, None      

        elif activation_func == 'LSQ':
            scale_y, y, out, binary_mask, pos_mask, neg_mask = ctx.saved_tensors
            grad_y = (grad_output / scale_y) * binary_mask
            dy_ds = (out - y) * binary_mask + (2 ** a_bit - 1) * pos_mask - 2 ** (a_bit - 1) * neg_mask
            grad_scale_y = (grad_output * dy_ds).sum() / (y.numel() * 2 ** (a_bit - 1))
            grad_zero_y = (grad_y.sum([0, 2, 3])).view(1,-1,1,1)
            return grad_y, grad_scale_y, grad_zero_y, None, None
      
class QuantizedConv2dDiff(QuantizedConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 w_bit=8, a_bit=None,
                 normalization_func=None, activation_func=None, grad_output_prune_ratio=None,
                 zero_x=0, zero_w=0, zero_y=0,  # keep same args
                 scale_x=0, scale_w=0, scale_y=0,
                 ):
        super(QuantizedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                              padding, dilation, groups, bias, padding_mode)
        self.register_buffer('scale_x', to_pt(scale_x))
        self.register_buffer('zero_x', to_pt(zero_x))
        self.register_buffer('zero_w', to_pt(zero_w))

        # self.register_buffer('zero_y', to_pt(zero_y))

        if configs.train_config.train_scale:
            # print('Note: the scale is also trained...')
            self.register_parameter('scale_y', torch.nn.Parameter(scale_y))
            self.register_parameter('scale_w', torch.nn.Parameter(scale_w))
        else:
            self.register_buffer('scale_y', scale_y)
            self.register_buffer('scale_w', scale_w)
        
        if configs.train_config.train_zero:
            # print('Note: the scale is also trained...')
            self.register_parameter('zero_y', torch.nn.Parameter(zero_y))
        else:
            self.register_buffer('zero_y', zero_y)

        self.w_bit = w_bit
        self.a_bit = a_bit if a_bit is not None else w_bit

        self.normalization_func = normalization_func
        self.activation_func = activation_func
        self.grad_output_prune_ratio = grad_output_prune_ratio

        self.normalization_layer = None
        
        if self.normalization_func is not None:
            if self.normalization_func == 'BN':
                self.normalization_layer = torch.nn.BatchNorm2d(out_channels)
                self.gamma = self.normalization_layer.weight
                self.beta = self.normalization_layer.bias
            elif self.normalization_func == 'GN':
                num_groups = out_channels // min_divisible_value(out_channels, 8)
                self.normalization_layer = torch.nn.GroupNorm(num_channels=out_channels,num_groups=num_groups,affine=True)
                self.gamma = self.normalization_layer.weight
                self.beta = self.normalization_layer.bias
            elif self.normalization_func == 'L1FRN':
                self.normalization_layer = FilterResponseNorm2d(out_channels, mode='L1')
                self.gamma = self.normalization_layer.gamma
                self.beta = self.normalization_layer.beta
                self.tau = self.normalization_layer.tau
            elif self.normalization_func == 'L2FRN':
                self.normalization_layer = FilterResponseNorm2d(out_channels, mode='L2')
                self.gamma = self.normalization_layer.gamma
                self.beta = self.normalization_layer.beta
                self.tau = self.normalization_layer.tau
            else:  # self-defined normalization
                if configs.train_config.train_normalization:
                    self.register_parameter('gamma', torch.nn.Parameter(torch.ones(out_channels).cuda()))
                    self.register_parameter('beta', torch.nn.Parameter(torch.zeros(out_channels).cuda()))
                else:
                    self.gamma = torch.ones(out_channels).cuda()
                    self.beta = torch.zeros(out_channels).cuda()
        else:
            self.gamma = None
            self.beta = None
    
    def forward(self, x):
        # self.effective_scale = (self.scale_x.to(torch.float64) * self.scale_w.to(torch.float64) / self.scale_y.to(torch.float64)).to(torch.float32)
        self.effective_scale = (self.scale_x.to(torch.float64) * self.scale_w.to(torch.float64)).to(torch.float32)
        out = _QuantizedConv2dFunc.apply(x, self.weight, self.bias, self.zero_x,
                                         self.effective_scale, 
                                         self.stride, self.padding, self.dilation, self.groups, self.grad_output_prune_ratio)
        if self.normalization_layer is not None:
            out = self.normalization_layer(out)
        else:
            out = _QuantizedNormalization.apply(out, self.gamma, self.beta, self.a_bit, self.normalization_func)
        
        # STE mask
        temp_out = round_tensor(out / self.scale_y) + self.zero_y
        self.binary_mask = (- 2 ** (self.a_bit - 1) <= temp_out) & (temp_out <= 2 ** (self.a_bit - 1) - 1)
        
        out = _TruncateActivationRange.apply(out, self.scale_y, self.zero_y, self.a_bit, self.activation_func)

        return out
    
    def forward_ZO_before_round(self, x):
        self.effective_scale = (self.scale_x.to(torch.float64) * self.scale_w.to(torch.float64)).to(torch.float32)
        out = _QuantizedConv2dFunc.apply(x, self.weight, self.bias, self.zero_x,
                                         self.effective_scale, 
                                         self.stride, self.padding, self.dilation, self.groups)
        return out
    
    def forward_ZO_after_round(self, x):
        if self.normalization_layer is not None:
            out = self.normalization_layer(x)
        else:
            out = _QuantizedNormalization.apply(x, self.gamma, self.beta, self.a_bit, self.normalization_func)
        
        # STE mask
        temp_out = round_tensor(out / self.scale_y) + self.zero_y
        self.binary_mask = (- 2 ** (self.a_bit - 1) <= temp_out) & (temp_out <= 2 ** (self.a_bit - 1) - 1)
        
        out = _TruncateActivationRange.apply(out, self.scale_y, self.zero_y, self.a_bit, self.activation_func)

        return out
    
    def local_backward(self, input, grad_output, binary_mask=None):
        if binary_mask is not None:
            grad_output = grad_output * binary_mask

        # effective_scale = scale_x * scale_w / scale_y
        # b_quantized = b / (w_scales * x_scale), so we may wanna compute grad_b / (w_scale * x_scale)
        # which is grad_b / (effective_scale * scale_y)
        input = input.round()
        input = input - self.zero_x
        weight = self.weight.round()

        ### grad_output is for y(int)
        effective_scale = (self.scale_x.to(torch.float64) * self.scale_w.to(torch.float64) / self.scale_y.to(torch.float64)).to(torch.float32)
        # effective_scale = (self.scale_x.to(torch.float64) * self.scale_w.to(torch.float64)).to(torch.float32)
        _grad_conv_out = grad_output * effective_scale.view(1, -1, 1, 1)

        grad_bias = _grad_conv_out.sum([0, 2, 3])
        _grad_conv_in = torch.nn.grad.conv2d_input(input.shape, weight, _grad_conv_out,
                                                   stride=self.stride, padding=self.padding,
                                                   dilation=self.dilation, groups=self.groups)
        grad_x = _grad_conv_in

        if CONV_W_GRAD:
            grad_w = torch.nn.grad.conv2d_weight(input, self.weight.shape, _grad_conv_out,
                                                 stride=self.stride, padding=self.padding,
                                                 dilation=self.dilation, groups=self.groups)
        else:
            grad_w = None

        if configs.backward_config.quantize_gradient:  # perform per-channel quantization
            # quantize grad_x and grad_w
            from .quantize_helper import get_weight_scales
            w_scales = get_weight_scales(grad_w, n_bit=8)
            grad_w = (grad_w / w_scales.view(-1, 1, 1, 1)).round() * w_scales.view(-1, 1, 1, 1)
            x_scales = get_weight_scales(grad_x.transpose(0, 1))
            grad_x = (grad_x / x_scales.view(1, -1, 1, 1)).round() * x_scales.view(1, -1, 1, 1)

        return grad_x, grad_w, grad_bias
    
    def update_quantize_params(self, signSGD=False, param_lr=None):
        if self.scale_y.grad is not None:
            scale_y_lr = param_lr['scale_y']
            if signSGD:
                self.scale_y.data -= scale_y_lr * torch.sign(self.scale_y.grad)
            else:
                self.scale_y.data -= scale_y_lr * self.scale_y.grad
          
            self.scale_y.grad = None
        
        if self.zero_y.grad is not None:
            zero_y_lr = param_lr['zero_y']
            if signSGD:
                self.zero_y.data -= zero_y_lr * torch.sign(self.zero_y.grad)
            else:
                self.zero_y.data -= zero_y_lr * self.zero_y.grad
            
            self.zero_y.grad = None
        
        if self.scale_w.grad is not None:
            scale_w_lr = param_lr['scale_w']
            if signSGD:
                self.scale_w.data -= scale_w_lr * torch.sign(self.scale_w.grad)
            else:
                self.scale_w.data -= scale_w_lr * self.scale_w.grad
            
            self.scale_w.grad = None
        
        if self.gamma is not None and self.gamma.grad is not None:
            gamma_lr = param_lr['gamma']
            if signSGD:
                self.gamma.data -= gamma_lr * torch.sign(self.gamma.grad)
            else:
                self.gamma.data -= gamma_lr * self.gamma.grad
            
            self.gamma.grad = None
        
        if self.beta is not None and self.beta.grad is not None:
            beta_lr = param_lr['beta']
            if signSGD:
                self.beta.data -= beta_lr * torch.sign(self.beta.grad)
            else:
                self.beta.data -= beta_lr * self.beta.grad
            
            self.beta.grad = None
        
        self.effective_scale = (self.scale_x.to(torch.float64) * self.scale_w.to(torch.float64) / self.scale_y.to(torch.float64)).to(torch.float32)


class QuantizedMbBlockDiff(QuantizedMbBlock):
    def forward(self, x):
        out = self.conv(x)
        if self.q_add is not None:
            if self.residual_conv is not None:
                x = self.residual_conv(x)
            out = self.q_add(x, out)

            # No normalization for residual block  
            out = _QuantizedNormalization.apply(out, None, None, self.a_bit, None)

            temp_out = round_tensor(out / self.q_add.scale_y) + self.q_add.zero_y
            self.binary_mask = (- 2 ** (self.a_bit - 1) <= temp_out) & (temp_out <= 2 ** (self.a_bit - 1) - 1)

            # No TLU activation for residual block
            out = _TruncateActivationRange.apply(out, self.q_add.scale_y, self.q_add.zero_y, self.a_bit, None)
            return out
        else:
            self.binary_mask = torch.ones_like(out, dtype=torch.bool)
            return out
    
    @torch.no_grad()
    def forward_q_add(self, x, out):
        if self.q_add is not None:
            if self.residual_conv is not None:
                x = self.residual_conv(x)
            out = self.q_add(x, out)

            # No normalization for residual block  
            out = _QuantizedNormalization.apply(out, None, None, self.a_bit, None)

            temp_out = round_tensor(out / self.q_add.scale_y) + self.q_add.zero_y
            self.binary_mask = (- 2 ** (self.a_bit - 1) <= temp_out) & (temp_out <= 2 ** (self.a_bit - 1) - 1)

            # No TLU activation for residual block
            out = _TruncateActivationRange.apply(out, self.q_add.scale_y, self.q_add.zero_y, self.a_bit, None)
            return out
        else:
            self.binary_mask = torch.ones_like(out, dtype=torch.bool)
            return out
    
    def block_input_update_quantize_params(self, block_input_scale=None, block_input_zero=None):
        self.conv[0].scale_x = block_input_scale
        self.conv[0].zero_x = block_input_zero
        self.conv[0].effective_scale = (self.conv[0].scale_x.to(torch.float64) * self.conv[0].scale_w.to(torch.float64) / self.conv[0].scale_y.to(torch.float64)).to(torch.float32)

        if self.q_add is not None:
            self.q_add.scale_x1 = block_input_scale
            self.q_add.zero_x1 = block_input_zero
    
    def block_conv_update_quantize_params(self, signSGD=False, param_lr=None):
        for conv_idx in range(len(self.conv)): 
            self.conv[conv_idx].update_quantize_params(signSGD, param_lr)
            try:
                self.conv[conv_idx+1].scale_x = self.conv[conv_idx].scale_y * 1.0
                self.conv[conv_idx+1].zero_x = self.conv[conv_idx].zero_y * 1
            except IndexError:
                if self.q_add is not None:
                    self.q_add.scale_x2 = self.conv[conv_idx].scale_y * 1.0
                    self.q_add.zero_x2 = self.conv[conv_idx].zero_y * 1
        
        if self.q_add is not None:
            self.q_add.update_quantize_params(signSGD, param_lr)

class ScaledLinear(torch.nn.Linear):
    # a fp version of fc used for training
    def __init__(self, in_features: int, out_features: int, scale_x, zero_x, bias: bool = True,
                 device=None, dtype=None, norm_feat=False):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.register_buffer('scale_x', to_pt(scale_x))
        self.register_buffer('zero_x', to_pt(zero_x))

        self.norm_feat = norm_feat
        if norm_feat:
            self.bias.data.fill_(2.)
            self.eps = 1e-5

    def forward(self, x):
        x = (x.squeeze(-1).squeeze(-1) - self.zero_x.detach().view(1, -1)) * self.scale_x.detach().view(1, -1)
        if self.norm_feat:
            x_norm = x.div(torch.norm(x, p=2, dim=1).view(-1, 1) + self.eps)
            weight_norm = self.weight.div(torch.norm(self.weight, p=2, dim=1).view(-1, 1) + self.eps)
            cos_dist = (x_norm @ weight_norm.T) * self.bias.view(1, -1)
            return cos_dist
        else:
            return super().forward(x)

