from typing import Callable

import math
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from quantize.quantized_ops_diff import QuantizedConv2dDiff as QuantizedConv2d
from quantize.quantized_ops_diff import QuantizedMbBlockDiff as QuantizedMbBlock
from quantize.quantized_ops_diff import ScaledLinear

from scipy.stats import qmc
from .ZO_Estim_entry import split_model, split_named_model, SplitedBlock
from .QMC_sampler import sphere_n, coord_basis, block_mask_generator, layer_mask_generator

from ..utils.config import configs
class XORRand:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self):
        seed = self.seed
        
        seed ^= seed << 13
        seed ^= seed >> 17
        seed ^= seed << 5
        
        self.seed = seed
        return seed

    def get_rng_state(self):
        return self.seed

    def manual_seed(self, seed):
        self.seed = seed

class ZO_Estim_MC(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        obj_fn: Callable,

        sigma: float = 0.1,
        n_sample: int = 20,
        signSGD: bool = False,
        trainable_param_list: list = 'all',
        trainable_layer_list: list = 'all',

        estimate_method: str = 'forward',
        perturb_method: str = 'batch',
        sample_method: str = 'gaussian',
        ):
        super().__init__()

        self.model = model
        self.obj_fn = obj_fn
        self.sigma = sigma
        self.n_sample = n_sample
        self.signSGD = signSGD

        if trainable_param_list == 'all':
            self.trainable_param_list = []
            for name, m in model.named_parameters():
                if m.requires_grad == True:
                    self.trainable_param_list.append(name) 
        else:
            self.trainable_param_list = trainable_param_list

        splited_named_modules = split_named_model(model)
        self.splited_block_list = []
        idx = 0
        for name, block in splited_named_modules.items():
            self.splited_block_list.append(SplitedBlock(idx, name, block))
            # print(name, layer)
            idx += 1
        
        self.trainable_layer_list = []

        if type(trainable_layer_list) is list:
            self.trainable_layer_list = trainable_layer_list
        elif trainable_layer_list == 'block-all':
            for splited_block in self.splited_block_list:
                if type(splited_block.block) in (QuantizedMbBlock,):
                    self.trainable_layer_list.append(splited_block.name)
        elif trainable_layer_list == 'layer-all':
            for splited_block in self.splited_block_list:
                if type(splited_block.block) is QuantizedMbBlock:
                    for conv_idx in range(len(splited_block.block.conv)):
                        self.trainable_layer_list.append(f'{splited_block.name}.conv.{conv_idx}')
        else:
            raise ValueError(f'Not supported trainable_layer_list {trainable_layer_list}')
        
        
        self.trainable_splited_block_list = []
        for splited_block in self.splited_block_list:
            if splited_block.name in self.trainable_layer_list:
                self.trainable_splited_block_list.append(splited_block)
        
        self.estimate_method = estimate_method
        self.perturb_method = perturb_method
        self.sample_method = sample_method

        self.device = next(self.model.parameters()).device
        self.dtype = next(self.model.parameters()).dtype
        self.param_lr = None

        # self.ZO_dimension = sum(p.numel() for p in filter(lambda x: x.requires_grad, self.model.parameters()))
        self.ZO_dimension = sum(p.numel() for name, p in self.model.named_parameters() if name in self.trainable_param_list)
        
        
        self.ZO_dimension = 0
        if 'param' in self.perturb_method:
            for name, param in self.model.named_parameters():
                if any(keyword in name for keyword in self.trainable_layer_list):
                    self.ZO_dimension += param.numel()
                    param.grad = torch.zeros_like(param)

        elif 'activation' in self.perturb_method:
            ### get activation dimension
            fwd_hook_handle_list = []
            for name, module in self.model.named_modules():
                if any(keyword in name for keyword in self.trainable_layer_list):
                    fwd_hook_get_out_dimension = self.create_fwd_hook_get_out_dimension()
                    fwd_hook_handle_list.append(module.register_forward_hook(fwd_hook_get_out_dimension))
                    
            _, old_loss = self.obj_fn(return_loss_reduction='none')
            
            for name, module in self.model.named_modules():
                if any(keyword in name for keyword in self.trainable_layer_list):
                    self.ZO_dimension += module.out_dimension
                    
            for fwd_hook_handle in fwd_hook_handle_list:
                fwd_hook_handle.remove()  
        
        self.ZO_dimension = int(self.ZO_dimension)
        print('ZO_dimension=', self.ZO_dimension)
        
        if self.sample_method == 'coord_basis':
            self.n_sample = self.ZO_dimension
        
        self.set_seed = configs.ZO_Estim.set_seed
        if self.set_seed == 'xorshift':
            self.xorshift_rand = XORRand(seed=torch.randint(low=1,high=2**31,size=(1,), dtype=torch.uint32).item())
        
        self.forward_counter = 0

    def _init_sampler(self, dimension):
        if self.sample_method == 'sobol':
            sampler = qmc.Sobol(d=dimension, scramble=False)
        elif self.sample_method == 'halton':
            sampler = qmc.Halton(d=dimension, scramble=True)
        elif self.sample_method == 'sphere_n':
            sampler = sphere_n(n=dimension)
        elif self.sample_method == 'coord_basis':
            sampler = coord_basis(dimension=dimension)
        else:
            sampler = None
        return sampler
            
    ### Generate random vectors from a normal distribution
    def _sample_unit_sphere(self, dimension, device):
        
        if self.sample_method == 'uniform':
            sample = torch.randn(dimension, device=device)
            sample = torch.nn.functional.normalize(sample, p=2, dim=0)
        elif self.sample_method == 'gaussian':
            sample = torch.randn(dimension, device=device) / dimension
        elif self.sample_method == 'bernoulli':
            ### Rademacher
            if 'u_int' in self.quantize_method:
                sample = torch.ones(dimension, device=device) - 2*torch.bernoulli(0.5*torch.ones(dimension, device=device))
            else:
                sample = torch.ones(dimension, device=device) - 2*torch.bernoulli(0.5*torch.ones(dimension, device=device))
                sample = sample / torch.sqrt(torch.tensor(dimension, device=device))
        elif self.sample_method == 'coord_basis':
            sample = next(self.sampler)
            sample = sample.to(device)
        elif self.sample_method in ('sobol', 'halton'):
            if self.sampler == None:
                raise ValueError('Need sampler input')
            else:
                sample = torch.Tensor(self.sampler.random(1)).squeeze()
                sample = 2*sample-torch.ones_like(sample)
                sample = torch.nn.functional.normalize(sample, p=2, dim=0)
                sample = sample.to(device)
        elif self.sample_method == 'sphere_n':
            sample = next(self.sampler)
            sample = sample.to(device)
        else:
            return NotImplementedError('Unlnown sample method', self.sample_method)
        
        return sample

    def _sample_unit_sphere_quantized(self, shape, sample_method, device):
        # if seed is not None:
        #     # Save the current random state
        #     current_random_state = torch.get_rng_state()
        #     # Set the new seed
        #     torch.manual_seed(seed)
        
        if sample_method == 'bernoulli':
            if self.set_seed == 'xorshift':
                # random_numbers = torch.tensor([self.xorshift_rand() for _ in range(torch.prod(torch.tensor(shape)))], device=device)
                # lsb = random_numbers & 1
                # sample = 1 - 2 * lsb.float()
                # sample = sample.view(shape)
    
                sample = torch.ones(shape, device=device).view(-1)
                for i in range(sample.numel()):
                    sample[i] = 1 - 2*(self.xorshift_rand() & 1)
                sample = sample.view(shape)
            else:
                sample = torch.ones(shape, device=device) - 2*torch.bernoulli(0.5*torch.ones(shape, device=device))
        else:
            raise NotImplementedError('Unlnown sample method', self.sample_method)
        
        # if seed is not None:
        #     # Restore the original random state
        #     torch.set_rng_state(current_random_state)
        
        return sample

    def _add_params_perturbation(self, sigma, u):
        u_idx = 0
        for name, param in self.model.named_parameters():
            if name in self.trainable_param_list:
                # Generate random perturbation with the same shape as the parameter
                param_len = param.numel()
                param_shape = param.shape

                if 'u_fp' in self.quantize_method:
                    name_list = name.split('.')
                    w_scale = torch.tensor(self.model[int(name_list[0])][int(name_list[1])].conv[int(name_list[3])].w_scale, device=self.device).view(-1, 1, 1, 1)
                    perturbation = u[u_idx : u_idx+param_len].reshape(param_shape) * sigma / w_scale
                    perturbation = perturbation.round()
                elif 'u_int' in self.quantize_method:
                    assert type(sigma) is int
                    perturbation = u[u_idx : u_idx+param_len].round().reshape(param_shape) * sigma
                else:
                    perturbation = u[u_idx : u_idx+param_len].reshape(param_shape) * sigma
                # Add perturbation to the parameter. Should use in-place addition
                param.data.add_(perturbation)
                u_idx += param_len 
    
    def get_actv_ZO_gradient(self, verbose=False):
        
        # if self.estimate_method == 'forward':
        _, old_loss = self.obj_fn(return_loss_reduction='none')

        if configs.train_config.layerwise_update == 'one':
            trainable_layer_list = [random.choice(self.trainable_layer_list)]
        else:
            trainable_layer_list = self.trainable_layer_list
        
        for trainable_layer_name in trainable_layer_list:
            trainable_layer_name = trainable_layer_name.split('.')
            block_name = f'{trainable_layer_name[0]}.{trainable_layer_name[1]}'
            if 'conv' in trainable_layer_name:
                conv_idx = int(trainable_layer_name[3])
            else:
                conv_idx = None
            
            for splited_block in self.splited_block_list:
                if splited_block.name == block_name:
                    # splited_layer = splited_block
                    break
            
            ##### Estimate gradient
            # Update all conv layers in this block
            if conv_idx == None:
                ZO_grad, pre_activ, mask = self.get_block_actv_ZO_gradint(splited_block, old_loss, local_backward_args=True)
            # Update single conv layer
            else:
                ZO_grad, pre_activ, mask = self.get_layer_actv_ZO_gradint(splited_block, conv_idx, old_loss, local_backward_args=True)
            
            ##### Update gradient
            batch_sz = ZO_grad.shape[0]
            
            if splited_block.type == nn.Linear:
                splited_block.block.weight.grad = torch.matmul(ZO_grad.T, pre_activ) / batch_sz  # average over all batch!
                splited_block.block.bias.grad = torch.mean(ZO_grad, dim=0)
            elif splited_block.type == QuantizedMbBlock:
                ### Block-wise ZO estimation        
                if conv_idx == None:
                    if splited_block.block.q_add is not None:
                        ZO_grad = ZO_grad * splited_block.block.q_add.scale_x2 / splited_block.block.q_add.scale_y

                    grad_x = ZO_grad 
                    for idx in range(len(splited_block.block.conv)-1, -1, -1):
                        layer_input = splited_block.block.conv[:idx](pre_activ)
                        grad_x, grad_w, grad_bias = splited_block.block.conv[idx].local_backward(input=layer_input, grad_output=grad_x, binary_mask=splited_block.block.conv[idx].binary_mask)
                        
                        splited_block.block.conv[idx].weight.grad = grad_w
                        splited_block.block.conv[idx].bias.grad = grad_bias 
                ### layer-wise ZO estimation        
                else:
                    # grad_x, grad_w, grad_bias = splited_block.block.conv[conv_idx].local_backward(input=pre_activ, grad_output=ZO_grad, binary_mask=splited_block.block.conv[conv_idx].binary_mask)
                    grad_x, grad_w, grad_bias = splited_block.block.conv[conv_idx].local_backward(input=pre_activ, grad_output=ZO_grad, binary_mask=mask.bool())

                    if configs.train_config.layerwise_update is None:
                        if splited_block.block.conv[conv_idx].weight.grad is None:
                            splited_block.block.conv[conv_idx].weight.grad = grad_w
                            splited_block.block.conv[conv_idx].bias.grad = grad_bias
                        else:
                            splited_block.block.conv[conv_idx].weight.grad += grad_w
                            splited_block.block.conv[conv_idx].bias.grad += grad_bias
                        # splited_block.block.conv[conv_idx].out_grad = ZO_grad
                    else:
                        ##### layerwise update
                        this_layer = splited_block.block.conv[conv_idx]
                        lr = self.param_lr
                        this_layer.weight.data.sub_( (lr * grad_w / this_layer.scale_w.view(-1, 1, 1, 1) ** 2).round().clamp(- 2 ** (this_layer.w_bit - 1), 2 ** (this_layer.w_bit - 1) - 1) )
                        this_layer.bias.data.sub_( (lr * grad_bias / (this_layer.scale_x * this_layer.scale_w) ** 2).round().clamp(- 2 ** (4*this_layer.w_bit - 1), 2 ** (4*this_layer.w_bit - 1) - 1) )

            else:
                raise NotImplementedError('Unknown block type')      
        return None

    def get_layer_actv_ZO_gradint(self, splited_block, conv_idx, old_loss, local_backward_args=False):
        assert splited_block.type == QuantizedMbBlock
        block_in = self.obj_fn(ending_idx=splited_block.idx, return_loss_reduction='no_loss')
        pre_activ = splited_block.block.conv[:conv_idx](block_in)
        post_actv = splited_block.block.conv[conv_idx](pre_activ)

        if configs.train_config.layerwise_update and self.estimate_method == 'forward':
            _, old_loss = self.obj_fn(starting_idx=splited_block.idx, input=block_in, return_loss_reduction='none')

        if isinstance(self.sigma, dict):
            sigma = self.sigma['actv']
        else:
            sigma = self.sigma
            
        if type(sigma) is float:
            q_sigma = sigma / splited_block.block.conv[conv_idx].scale_y.view(-1, 1, 1, 1)
            q_sigma = q_sigma.round()
        elif type(sigma) is int:
            q_sigma = sigma
        else:
            raise ValueError('Unknown sigma type')

        if configs.train_config.ZO_grad_prune_ratio is not None:
            ZO_grad_prune_ratio = configs.train_config.ZO_grad_prune_ratio
            mask = torch.zeros_like(post_actv, dtype=torch.bool)
            
            ### Depthwise filter magnitude top-k sparsity
            # dw_channelwise = splited_block.block.conv[conv_idx+1].weight.abs().sum([1,2,3])
            # topk_dim = int((1.0-ZO_grad_prune_ratio) * dw_channelwise.numel())
            # _, indices = torch.topk(dw_channelwise, topk_dim)
            # mask[:,indices,:,:] = True
            
            ### Output actv magnitude top-k sparsity
            # topk_dim = int((1.0-ZO_grad_prune_ratio) * post_actv.numel())
            # _, indices = torch.topk((post_actv-splited_block.block.conv[conv_idx].zero_y).flatten(), topk_dim)
            # mask.view(-1)[indices] = True

            ### Output actv magnitude top-k sparsity, batch-wise
            if configs.train_config.prune_method == 'top-k-param':
                batch_sz = post_actv.shape[0]
                topk_dim = int((1.0-ZO_grad_prune_ratio) * (post_actv.numel() / batch_sz))
                for b in range(batch_sz):
                    _, indices = torch.topk((post_actv[b]-splited_block.block.conv[conv_idx].zero_y).flatten(), topk_dim)
                    mask[b].view(-1)[indices] = True
                mask = mask * splited_block.block.conv[conv_idx].binary_mask
            
            elif configs.train_config.prune_method == 'random-k-param':
                ratio = 1.0-ZO_grad_prune_ratio
                mask = torch.bernoulli(ratio*splited_block.block.conv[conv_idx].binary_mask)
            
            ### Output actv magnitude top-k sparsity, channel-wise
            elif configs.train_config.prune_method == 'top-k-channel':
                batch_sz = post_actv.shape[0]
                topk_dim = int((1.0-ZO_grad_prune_ratio) * (post_actv.size(1)))
                for b in range(batch_sz):
                    _, indices = torch.topk(torch.linalg.norm((post_actv[b]-splited_block.block.conv[conv_idx].zero_y), dim=(1,2)), topk_dim)
                    mask[b,indices,:,:] = True
                mask = mask * splited_block.block.conv[conv_idx].binary_mask
            
            elif configs.train_config.prune_method == 'random-k-channel':
                ratio = 1.0-ZO_grad_prune_ratio
                mask = torch.bernoulli(ratio*torch.ones(tuple(post_actv.size()[0:2]), device=post_actv.device))
                mask = mask.unsqueeze(-1).unsqueeze(-1) * splited_block.block.conv[conv_idx].binary_mask
            
        else:
            mask = splited_block.block.conv[conv_idx].binary_mask
            # mask = torch.ones_like(post_actv)
        
        splited_block.block.conv[conv_idx].actv_mask = mask

        post_actv_shape = tuple(post_actv.shape)
        batch_sz = post_actv_shape[0]
        post_actv = post_actv.view(batch_sz, -1)
        mask = mask.view(batch_sz, -1)

        ZO_grad = torch.zeros_like(post_actv, device=self.device)
        if self.sample_method == 'coord_basis':
            for i in range(post_actv.shape[1]):
                org_post_actv = post_actv[:, i].int()
                post_actv[:, i] = post_actv[:, i] + mask[:, i] * sigma
                if splited_block.type == QuantizedMbBlock:
                    a_bit = splited_block.block.conv[conv_idx].a_bit
                    post_actv.data = post_actv.data.clamp(- 2 ** (a_bit - 1), 2 ** (a_bit - 1) - 1)
                
                pos_distance = post_actv[:, i] - org_post_actv

                block_out = splited_block.block.conv[conv_idx+1:](post_actv.view(post_actv_shape))
                block_out = splited_block.block.forward_q_add(block_in, block_out)

                _, pos_loss = self.obj_fn(starting_idx=splited_block.idx+1, input=block_out, return_loss_reduction='none')
                self.forward_counter += 1
                post_actv[:, i] = org_post_actv

                if self.estimate_method == 'forward':
                    for batch_idx in range(batch_sz):
                        if ((pos_loss[batch_idx] - old_loss[batch_idx]) != 0) & (pos_distance[batch_idx] != 0):
                            ZO_grad[batch_idx,i] = (pos_loss[batch_idx] - neg_loss[batch_idx]) / pos_distance[batch_idx]
                        else:
                            ZO_grad[batch_idx,i] = 0

                elif self.estimate_method == 'antithetic':
                    post_actv[:, i] = post_actv[:, i] - mask[:, i] * sigma
                    if splited_block.type == QuantizedMbBlock:
                        a_bit = splited_block.block.conv[conv_idx].a_bit
                        post_actv.data = post_actv.data.clamp(- 2 ** (a_bit - 1), 2 ** (a_bit - 1) - 1)
                    
                    neg_distance = org_post_actv - post_actv[:, i]
                    
                    block_out = splited_block.block.conv[conv_idx+1:](post_actv.view(post_actv_shape))
                    block_out = splited_block.block.forward_q_add(block_in, block_out)

                    _, neg_loss = self.obj_fn(starting_idx=splited_block.idx+1, input=block_out, return_loss_reduction='none')
                    self.forward_counter += 1
                    
                    for batch_idx in range(batch_sz):
                        if ((pos_loss[batch_idx] - neg_loss[batch_idx]) != 0) & ((pos_distance[batch_idx]+neg_distance[batch_idx]) != 0):
                            ZO_grad[batch_idx,i] = (pos_loss[batch_idx] - neg_loss[batch_idx]) / (pos_distance[batch_idx]+neg_distance[batch_idx])
                        else:
                            ZO_grad[batch_idx,i] = 0

                    post_actv[:, i] = org_post_actv
                else:
                    raise NotImplementedError('Unknown estimate method')
            
            ZO_grad = (ZO_grad / batch_sz).view(post_actv_shape)
            mask = mask.view(post_actv_shape)
        elif self.sample_method == 'bernoulli':         
            org_post_actv = post_actv.int()

            # if hasattr(configs.ZO_Estim, 'n_sample_distri'):
            #     n_sample_distri = configs.ZO_Estim.n_sample_distri
            #     if n_sample_distri == 'uniform':
            #         n_sample = self.n_sample / 42
            #     elif n_sample_distri == 'dim':
            #         n_sample = int((self.n_sample * splited_block.block.conv[conv_idx].out_dimension / self.ZO_dimension))
            # else:
            #     n_sample = self.n_sample
            
            n_sample = self.n_sample
                  
            for i in range(n_sample):
                if hasattr(configs.ZO_Estim, 'sync_batch_perturb') and configs.ZO_Estim.sync_batch_perturb:
                    u = mask * torch.tile(self._sample_unit_sphere_quantized(post_actv.shape[-1], self.sample_method, self.device).unsqueeze(0), (batch_sz, 1))
                else:
                    u = mask * self._sample_unit_sphere_quantized(post_actv.shape, self.sample_method, self.device)

                post_actv = post_actv + u * q_sigma

                if splited_block.type == QuantizedMbBlock:
                    a_bit = splited_block.block.conv[conv_idx].a_bit
                    post_actv.data = post_actv.data.clamp(- 2 ** (a_bit - 1), 2 ** (a_bit - 1) - 1)
                
                # pos_distance = post_actv[:, i] - org_post_actv

                block_out = splited_block.block.conv[conv_idx+1:](post_actv.view(post_actv_shape))
                block_out = splited_block.block.forward_q_add(block_in, block_out)

                _, pos_loss = self.obj_fn(starting_idx=splited_block.idx+1, input=block_out, return_loss_reduction='none')
                self.forward_counter += 1
                post_actv = org_post_actv

                if self.estimate_method == 'forward':
                    ZO_grad += (pos_loss - old_loss).view(-1,1) / sigma * u

                elif self.estimate_method == 'antithetic':
                    post_actv = post_actv - u * q_sigma

                    if splited_block.type == QuantizedMbBlock:
                        a_bit = splited_block.block.conv[conv_idx].a_bit
                        post_actv.data = post_actv.data.clamp(- 2 ** (a_bit - 1), 2 ** (a_bit - 1) - 1)
                    
                    # pos_distance = post_actv[:, i] - org_post_actv

                    block_out = splited_block.block.conv[conv_idx+1:](post_actv.view(post_actv_shape))
                    block_out = splited_block.block.forward_q_add(block_in, block_out)
                
                    _, neg_loss = self.obj_fn(starting_idx=splited_block.idx+1, input=block_out, return_loss_reduction='none')
                    self.forward_counter += 1
                    
                    ZO_grad += (pos_loss - neg_loss).view(-1,1) / 2.0 / sigma * u

                    post_actv = org_post_actv
              
            ZO_grad = (ZO_grad / n_sample / batch_sz).view(post_actv_shape)
            mask = mask.view(post_actv_shape)
        else:
            raise NotImplementedError('Unknown sample method')
        
        if type(sigma) is float:
            ### scale to theta_bar's gradient
            ZO_grad = ZO_grad * splited_block.block.conv[conv_idx].scale_y
        
        ### ZO gradient scale adjustment
        if hasattr(configs.ZO_Estim, 'scale') and torch.sum(mask).item() > 0:
            if configs.ZO_Estim.scale == 'sqrt-dim':
                ZO_grad = ZO_grad * math.sqrt((batch_sz * n_sample) / (batch_sz * n_sample + torch.sum(mask).item()/batch_sz - 1))
            elif configs.ZO_Estim.scale == 'dim':
                ZO_grad = ZO_grad * ((batch_sz * n_sample) / (batch_sz * n_sample + torch.sum(mask).item()/batch_sz - 1))
                
            elif type(configs.ZO_Estim.scale) is int:
                ZO_grad = ZO_grad / configs.ZO_Estim.scale
            else:
                raise NotImplementedError(f'Unknown {configs.ZO_Estim.scale}')
        
        if local_backward_args == True:
            return ZO_grad, pre_activ, mask
        else:
            return ZO_grad
    
    def get_block_actv_ZO_gradint(self, splited_block, old_loss, local_backward_args=False):
        assert splited_block.type == QuantizedMbBlock

        pre_activ = self.obj_fn(ending_idx=splited_block.idx, return_loss_reduction='no_loss')
        post_actv = splited_block.block(pre_activ)

        if configs.train_config.layerwise_update and self.estimate_method == 'forward':
            _, old_loss = self.obj_fn(starting_idx=splited_block.idx, input=pre_activ, return_loss_reduction='none')

        assert type(self.sigma) is int
        assert hasattr(splited_block.block, 'binary_mask')
        if configs.train_config.ZO_grad_prune_ratio is not None:
            ZO_grad_prune_ratio = configs.train_config.ZO_grad_prune_ratio
            mask = torch.zeros_like(post_actv, dtype=torch.bool)

            topk_dim = int((1.0-ZO_grad_prune_ratio) * post_actv.numel())
            _, indices = torch.topk((post_actv-splited_block.block.conv[-1].zero_y).flatten(), topk_dim)
            mask.view(-1)[indices] = True
            # batch_sz = post_actv.shape[0]
            # topk_dim = int((1.0-ZO_grad_prune_ratio) * (post_actv.numel() / batch_sz))
            # for b in range(batch_sz):
            #     _, indices = torch.topk((post_actv[b]-splited_block.block.conv[-1].zero_y).flatten(), topk_dim)
            #     mask[b].view(-1)[indices] = True
        else:
            mask = splited_block.block.conv[-1].binary_mask.int()
            # mask = torch.ones_like(post_actv)

        post_actv_shape = tuple(post_actv.shape)
        batch_sz = post_actv_shape[0]

        post_actv = post_actv.view(batch_sz, -1)
        mask = mask.view(batch_sz, -1)

        ZO_grad = torch.zeros_like(post_actv, device=self.device)

        if self.sample_method == 'coord_basis':

            for i in range(post_actv.shape[1]):
                org_post_actv = post_actv[:, i].int()

                post_actv[:, i] = post_actv[:, i] + mask[:, i] * self.sigma
                a_bit = splited_block.block.a_bit
                post_actv.data = post_actv.data.clamp(- 2 ** (a_bit - 1), 2 ** (a_bit - 1) - 1)
                
                pos_distance = post_actv[:, i] - org_post_actv

                _, pos_loss = self.obj_fn(starting_idx=splited_block.idx+1, input=post_actv.view(post_actv_shape), return_loss_reduction='none')
                self.forward_counter += 1
                post_actv[:, i] = org_post_actv

                if self.estimate_method == 'forward':

                    for batch_idx in range(batch_sz):
                        if ((pos_loss[batch_idx] - old_loss[batch_idx]) != 0) & (pos_distance[batch_idx] != 0):
                            ZO_grad[batch_idx,i] = (pos_loss[batch_idx] - neg_loss[batch_idx]) / pos_distance[batch_idx]
                        else:
                            ZO_grad[batch_idx,i] = 0

                elif self.estimate_method == 'antithetic':

                    post_actv[:, i] = post_actv[:, i] - mask[:, i] * self.sigma
                    a_bit = splited_block.block.a_bit
                    post_actv.data = post_actv.data.clamp(- 2 ** (a_bit - 1), 2 ** (a_bit - 1) - 1)
                    
                    neg_distance = org_post_actv - post_actv[:, i]
                
                    _, neg_loss = self.obj_fn(starting_idx=splited_block.idx+1, input=post_actv.view(post_actv_shape), return_loss_reduction='none')
                    self.forward_counter += 1
                    
                    for batch_idx in range(batch_sz):
                        if ((pos_loss[batch_idx] - neg_loss[batch_idx]) != 0) & ((pos_distance[batch_idx]+neg_distance[batch_idx]) != 0):
                            ZO_grad[batch_idx,i] = (pos_loss[batch_idx] - neg_loss[batch_idx]) / (pos_distance[batch_idx]+neg_distance[batch_idx])
                        else:
                            ZO_grad[batch_idx,i] = 0

                    post_actv[:, i] = org_post_actv
                else:
                    raise NotImplementedError('Unknown estimate method')
            
            ZO_grad = (ZO_grad / batch_sz).view(post_actv_shape)
            mask = mask.view(post_actv_shape)
        elif self.sample_method == 'bernoulli':         
            org_post_actv = post_actv.int()

            for i in range(self.n_sample):
                if configs.ZO_Estim.sync_batch_perturb:
                    u = mask * torch.tile(self._sample_unit_sphere_quantized(post_actv.shape[-1], self.sample_method, self.device).unsqueeze(0), (batch_sz, 1))
                else:
                    u = mask * self._sample_unit_sphere_quantized(post_actv.shape, self.sample_method, self.device)

                post_actv = post_actv + u * self.sigma

                a_bit = splited_block.block.a_bit
                post_actv.data = post_actv.data.clamp(- 2 ** (a_bit - 1), 2 ** (a_bit - 1) - 1)
                # pos_distance = post_actv - org_post_actv

                _, pos_loss = self.obj_fn(starting_idx=splited_block.idx+1, input=post_actv.view(post_actv_shape), return_loss_reduction='none')
                self.forward_counter += 1
                post_actv = org_post_actv

                if self.estimate_method == 'forward':
                    ZO_grad += (pos_loss - old_loss).view(-1,1) / self.sigma * u

                elif self.estimate_method == 'antithetic':
                    post_actv = org_post_actv
                    post_actv = post_actv - u * self.sigma

                    a_bit = splited_block.block.a_bit
                    post_actv.data = post_actv.data.clamp(- 2 ** (a_bit - 1), 2 ** (a_bit - 1) - 1)
                    # neg_distance = org_post_actv - post_actv
                
                    _, neg_loss = self.obj_fn(starting_idx=splited_block.idx+1, input=post_actv.view(post_actv_shape), return_loss_reduction='none')
                    self.forward_counter += 1
                    
                    ZO_grad += (pos_loss - neg_loss).view(-1,1) / 2.0 / self.sigma * u

                    post_actv = org_post_actv
              
            ZO_grad = (ZO_grad / self.n_sample / batch_sz).view(post_actv_shape)
            mask = mask.view(post_actv_shape)
        else:
            raise NotImplementedError('Unknown sample method')

        if configs.train_config.ZO_grad_prune_ratio is not None:
            ZO_grad = ZO_grad * 4 / int((1.0-ZO_grad_prune_ratio) * (post_actv.numel() / batch_sz))
        else:
            ZO_grad = ZO_grad / (post_actv.numel() / batch_sz)
        
        if local_backward_args == True:
            return ZO_grad, pre_activ, mask
        else:
            return ZO_grad
          
    def get_single_param_ZO_gradient(self, block_idx, trainable_layer, param, block_in, old_loss, sigma, estimate_method, sample_method, p_scale=False):
        param_dim = param.numel()
        param_shape = param.shape
        
        fp_sigma = sigma

        # if type(sigma) is not int:
        #     L = self.n_sample
        #     d = param.data.numel()
        #     sigma = sigma * math.sqrt(4*L/(L+d-1)) / trainable_layer.scale_w.view(-1, 1, 1, 1)
        #     sigma = sigma.round()
        
        if type(sigma) is not int:
            if param.dim() == 4:
                sigma = sigma / trainable_layer.scale_w.view(-1, 1, 1, 1)
                sigma = sigma.round()
            elif param.dim() == 1:
                sigma = sigma / trainable_layer.scale_x / trainable_layer.scale_w
                sigma = sigma.round()
            
                
        param_ZO_grad = torch.zeros_like(param, device=self.device)
        loss_diff = 0

        if configs.train_config.ZO_grad_prune_ratio is not None:
            ZO_grad_prune_ratio = configs.train_config.ZO_grad_prune_ratio
            mask = torch.zeros_like(param, dtype=torch.bool)

            ### Output actv magnitude top-k sparsity, batch-wise
            if configs.train_config.prune_method == 'top-k-param':
                raise NotImplementedError('top-k-param not implemented yet')
            
            elif configs.train_config.prune_method == 'random-k-param':
                ratio = 1.0-ZO_grad_prune_ratio
                mask = torch.bernoulli(ratio*torch.ones_like(param))
            
            ### Output actv magnitude top-k sparsity, channel-wise
            elif configs.train_config.prune_method == 'top-k-channel':
                raise NotImplementedError('top-k-channel not implemented yet')
            
            elif configs.train_config.prune_method == 'random-k-channel':
                raise NotImplementedError('random-k-channel not implemented yet')
            
        else:
            mask = torch.ones_like(param)
        
        if param.dim() == 4:
            trainable_layer.weight_mask = mask
        elif param.dim() == 1:
            trainable_layer.bias_mask = mask

        if sample_method == 'coord_basis':
            param_vec = param.view(-1)
            param_ZO_grad = param_ZO_grad.view(-1)
            mask = mask.view(-1)
            for i in range(param_dim):
                if mask[i] == 0:
                    pass
                else:
                    # pos
                    param_vec[i] = param_vec[i] + sigma
                    _, pos_loss = self.obj_fn(starting_idx=block_idx, input=block_in, return_loss_reduction='mean')
                    param_vec[i] = param_vec[i] - sigma

                    # neg
                    if estimate_method == 'forward':
                        param_ZO_grad[i] = (pos_loss - old_loss) / sigma
                    elif estimate_method == 'antithetic':
                        param_vec[i] = param_vec[i] - sigma
                        _, neg_loss = self.obj_fn(starting_idx=block_idx, input=block_in, return_loss_reduction='mean')
                        param_vec[i] = param_vec[i] + sigma

                        param_ZO_grad[i] = (pos_loss - neg_loss) / 2 / sigma
                    else:
                        raise NotImplementedError('Unknown estimate method')
        elif sample_method == 'bernoulli':
            old_param = param.clone()
            if hasattr(configs.ZO_Estim, 'n_sample_distri'):
                n_sample_distri = configs.ZO_Estim.n_sample_distri
                if n_sample_distri == 'uniform':
                    n_sample = self.n_sample / 42
                elif n_sample_distri == 'dim':
                    n_sample = int(self.n_sample * param_dim / self.ZO_dimension)
                elif n_sample_distri == 'half':
                    n_sample = int(self.n_sample / 2)
            else:
                n_sample = self.n_sample
            for i in range(n_sample):
                u = self._sample_unit_sphere_quantized(param.shape, sample_method, self.device) * mask
                # u = u / math.sqrt((self.n_sample + u.numel() - 1) / 4 / self.n_sample)
                # pos
                param.add_(u * sigma)
                if type(trainable_layer) == QuantizedConv2d:
                    w_bit = trainable_layer.w_bit
                    if param.dim() == 4:
                        param.data = param.data.clamp(- 2 ** (w_bit - 1), 2 ** (w_bit - 1) - 1)
                    elif param.dim() == 1:
                        param.data = param.data.clamp(- 2 ** (4*w_bit - 1), 2 ** (4*w_bit - 1) - 1)
                _, pos_loss = self.obj_fn(starting_idx=block_idx, input=block_in, return_loss_reduction='mean')
                param.copy_(old_param)

                # neg
                if estimate_method == 'forward':
                    # loss_diff += (pos_loss - old_loss) / fp_sigma / self.n_sample
                    param_ZO_grad += (pos_loss - old_loss)  * u
                elif estimate_method == 'antithetic':
                    param.sub_(u * sigma)
                    if type(trainable_layer) == QuantizedConv2d:
                        w_bit = trainable_layer.w_bit
                        if param.dim() == 4:
                            param.data = param.data.clamp(- 2 ** (w_bit - 1), 2 ** (w_bit - 1) - 1)
                        elif param.dim() == 1:
                            param.data = param.data.clamp(- 2 ** (4*w_bit - 1), 2 ** (4*w_bit - 1) - 1)
                    _, neg_loss = self.obj_fn(starting_idx=block_idx, input=block_in, return_loss_reduction='mean')
                    param.copy_(old_param)

                    # loss_diff += (pos_loss -  neg_loss) / 2 / sigma / self.n_sample
                    # loss_diff += (pos_loss - 2*old_loss + neg_loss) / fp_sigma**2 / self.n_sample
                    param_ZO_grad += (pos_loss - neg_loss) / 2 * u
            
            if type(sigma) is int:
                param_ZO_grad = param_ZO_grad / sigma
            else: 
                # param_ZO_grad = param_ZO_grad * torch.where(sigma != 0, 1 / sigma, sigma)
                ### scale to grad_w_bar
                # param_ZO_grad = param_ZO_grad / sigma
                ### scale to grad_w_bar via chains rule
                param_ZO_grad = param_ZO_grad / fp_sigma
                if param.dim() == 4:
                    param_ZO_grad = param_ZO_grad * trainable_layer.scale_w.view(-1, 1, 1, 1)
                elif param.dim() == 1:
                    param_ZO_grad = param_ZO_grad * trainable_layer.scale_x * trainable_layer.scale_w

            param_ZO_grad = param_ZO_grad / n_sample
            
            if hasattr(configs.ZO_Estim, 'scale'):
                if configs.ZO_Estim.scale == 'sqrt-dim':
                    # param_ZO_grad = param_ZO_grad * math.sqrt(n_sample / (param.numel() - 1))
                    # param_ZO_grad = param_ZO_grad * math.sqrt(n_sample / (self.n_sample + param.numel() + 1))
                    param_ZO_grad = param_ZO_grad * math.sqrt(n_sample / (n_sample + torch.sum(mask).item() - 1))
                elif configs.ZO_Estim.scale == 'dim':
                    param_ZO_grad = param_ZO_grad * (n_sample / (n_sample + torch.sum(mask).item() - 1))
                elif type(configs.ZO_Estim.scale) is int:
                    param_ZO_grad = param_ZO_grad / configs.ZO_Estim.scale
                else:
                    raise NotImplementedError(f'Unknown {configs.ZO_Estim.scale}')

            # if hasattr(configs.ZO_Estim, 'scale'):
            #     batch_sz = block_in.shape[0]
            #     if configs.ZO_Estim.scale == 'sqrt-dim':
            #         # param_ZO_grad = param_ZO_grad * math.sqrt(n_sample / (param.numel() - 1))
            #         # param_ZO_grad = param_ZO_grad * math.sqrt(n_sample / (self.n_sample + param.numel() + 1))
            #         param_ZO_grad = param_ZO_grad * math.sqrt((n_sample*batch_sz) / (n_sample*batch_sz + torch.sum(mask).item() - 1))
            #     elif configs.ZO_Estim.scale == 'dim':
            #         param_ZO_grad = param_ZO_grad * ((n_sample*batch_sz) / (n_sample*batch_sz + torch.sum(mask).item() - 1))
            #     elif type(configs.ZO_Estim.scale) is int:
            #         param_ZO_grad = param_ZO_grad / configs.ZO_Estim.scale
            #     else:
            #         raise NotImplementedError(f'Unknown {configs.ZO_Estim.scale}')
        else:
            return NotImplementedError('sample method not implemented yet')

        param_ZO_grad = param_ZO_grad.view(param_shape)
        
        return param_ZO_grad
    
    
    def get_single_param_ZO_gradient_independent(self, block_idx, trainable_layer, param, block_in, old_loss, sigma, estimate_method, sample_method, p_scale=False):
        param_dim = param.numel()
        param_shape = param.shape
        
        fp_sigma = sigma

        # if type(sigma) is not int:
        #     L = self.n_sample
        #     d = param.data.numel()
        #     sigma = sigma * math.sqrt(4*L/(L+d-1)) / trainable_layer.scale_w.view(-1, 1, 1, 1)
        #     sigma = sigma.round()
        
        if type(sigma) is not int:
            if param.dim() == 4:
                sigma = sigma / trainable_layer.scale_w.view(-1, 1, 1, 1)
                sigma = sigma.round()
            elif param.dim() == 1:
                sigma = sigma / trainable_layer.scale_x / trainable_layer.scale_w
                sigma = sigma.round()
            
                
        param_ZO_grad = torch.zeros_like(param, device=self.device)
        loss_diff = 0

        if configs.train_config.ZO_grad_prune_ratio is not None:
            ZO_grad_prune_ratio = configs.train_config.ZO_grad_prune_ratio
            mask = torch.zeros_like(param, dtype=torch.bool)

            ### Output actv magnitude top-k sparsity, batch-wise
            if configs.train_config.prune_method == 'top-k-param':
                raise NotImplementedError('top-k-param not implemented yet')
            
            elif configs.train_config.prune_method == 'random-k-param':
                ratio = 1.0-ZO_grad_prune_ratio
                mask = torch.bernoulli(ratio*torch.ones_like(param))
            
            ### Output actv magnitude top-k sparsity, channel-wise
            elif configs.train_config.prune_method == 'top-k-channel':
                raise NotImplementedError('top-k-channel not implemented yet')
            
            elif configs.train_config.prune_method == 'random-k-channel':
                raise NotImplementedError('random-k-channel not implemented yet')
            
        else:
            mask = torch.ones_like(param)
        
        if param.dim() == 4:
            trainable_layer.weight_mask = mask
        elif param.dim() == 1:
            trainable_layer.bias_mask = mask

        if sample_method == 'coord_basis':
            raise NotImplementedError('coord_basis not implemented yet')
        elif sample_method == 'bernoulli':
            old_param = param.clone()
            if hasattr(configs.ZO_Estim, 'n_sample_distri'):
                n_sample_distri = configs.ZO_Estim.n_sample_distri
                if n_sample_distri == 'uniform':
                    n_sample = self.n_sample / 42
                elif n_sample_distri == 'dim':
                    n_sample = int(self.n_sample * param_dim / self.ZO_dimension)
                elif n_sample_distri == 'half':
                    n_sample = int(self.n_sample / 2)
            else:
                n_sample = self.n_sample
            
            batch_sz = block_in.shape[0]
            for batch_idx in range(batch_sz):
                inde_input = block_in[batch_idx].unsqueeze(0)
                for i in range(n_sample):
                    u = self._sample_unit_sphere_quantized(param.shape, sample_method, self.device) * mask
                    # u = u / math.sqrt((self.n_sample + u.numel() - 1) / 4 / self.n_sample)
                    # pos
                    param.add_(u * sigma)
                    if type(trainable_layer) == QuantizedConv2d:
                        w_bit = trainable_layer.w_bit
                        if param.dim() == 4:
                            param.data = param.data.clamp(- 2 ** (w_bit - 1), 2 ** (w_bit - 1) - 1)
                        elif param.dim() == 1:
                            param.data = param.data.clamp(- 2 ** (4*w_bit - 1), 2 ** (4*w_bit - 1) - 1)
                    _, pos_loss = self.obj_fn(starting_idx=block_idx, input=inde_input, return_loss_reduction='single_'+str(batch_idx))
                    param.copy_(old_param)

                    # neg
                    if estimate_method == 'forward':
                        # loss_diff += (pos_loss - old_loss) / fp_sigma / self.n_sample
                        param_ZO_grad += (pos_loss - old_loss)  * u
                    elif estimate_method == 'antithetic':
                        param.sub_(u * sigma)
                        if type(trainable_layer) == QuantizedConv2d:
                            w_bit = trainable_layer.w_bit
                            if param.dim() == 4:
                                param.data = param.data.clamp(- 2 ** (w_bit - 1), 2 ** (w_bit - 1) - 1)
                            elif param.dim() == 1:
                                param.data = param.data.clamp(- 2 ** (4*w_bit - 1), 2 ** (4*w_bit - 1) - 1)
                        _, neg_loss = self.obj_fn(starting_idx=block_idx, input=inde_input, return_loss_reduction='single_'+str(batch_idx))
                        param.copy_(old_param)

                        # loss_diff += (pos_loss -  neg_loss) / 2 / sigma / self.n_sample
                        # loss_diff += (pos_loss - 2*old_loss + neg_loss) / fp_sigma**2 / self.n_sample
                        param_ZO_grad += (pos_loss - neg_loss) / 2 * u
            
            ### single sample, the loss was not averaged by batch_sz
            param_ZO_grad = param_ZO_grad / batch_sz
            
            if type(sigma) is int:
                param_ZO_grad = param_ZO_grad / sigma
            else: 
                # param_ZO_grad = param_ZO_grad * torch.where(sigma != 0, 1 / sigma, sigma)
                ### scale to grad_w_bar
                # param_ZO_grad = param_ZO_grad / sigma
                ### scale to grad_w_bar via chains rule
                param_ZO_grad = param_ZO_grad / fp_sigma
                if param.dim() == 4:
                    param_ZO_grad = param_ZO_grad * trainable_layer.scale_w.view(-1, 1, 1, 1)
                elif param.dim() == 1:
                    param_ZO_grad = param_ZO_grad * trainable_layer.scale_x * trainable_layer.scale_w

            ### divided by total number of samples
            # param_ZO_grad = param_ZO_grad / n_sample
            param_ZO_grad = param_ZO_grad / (batch_sz * n_sample)

            if hasattr(configs.ZO_Estim, 'scale'):
                if configs.ZO_Estim.scale == 'sqrt-dim':
                    param_ZO_grad = param_ZO_grad * math.sqrt((n_sample*batch_sz) / (n_sample*batch_sz + torch.sum(mask).item() - 1))
                elif configs.ZO_Estim.scale == 'dim':
                    param_ZO_grad = param_ZO_grad * ((n_sample*batch_sz) / (n_sample*batch_sz + torch.sum(mask).item() - 1))
                elif type(configs.ZO_Estim.scale) is int:
                    param_ZO_grad = param_ZO_grad / configs.ZO_Estim.scale
                else:
                    raise NotImplementedError(f'Unknown {configs.ZO_Estim.scale}')
        else:
            return NotImplementedError('sample method not implemented yet')

        param_ZO_grad = param_ZO_grad.view(param_shape)
        
        return param_ZO_grad
    
    def get_weight_and_bias_ZO_gradient(self, block_idx, trainable_layer, block_in, old_loss):
        dimension = trainable_layer.weight.numel() + trainable_layer.bias.numel() 
        trainable_layer.weight.grad = torch.zeros_like(trainable_layer.weight)
        trainable_layer.bias.grad = torch.zeros_like(trainable_layer.bias)

        if self.sample_method == 'coord_basis':
            raise NotImplementedError
        else:
            u = dict()
            old_param = dict()
            old_param['weight'] = trainable_layer.weight.data.clone()
            old_param['bias'] = trainable_layer.bias.data.clone()
            
            n_sample = self.n_sample
            for i in range(n_sample):
                ### Generate random perturbation with the same shape as the parameter
                u['weight'] = self._sample_unit_sphere_quantized(trainable_layer.weight.shape, self.sample_method, self.device)
                u['bias'] = self._sample_unit_sphere_quantized(trainable_layer.bias.shape, self.sample_method, self.device)
                
                # p_sigma = self.sigma
                
                if isinstance(self.sigma, dict):
                    weight_sigma = self.sigma['weight']
                    bias_sigma = self.sigma['bias']
                else:
                    weight_sigma = self.sigma
                    bias_sigma = self.sigma
                
                ### Add perturbation to the parameter
                # pos
                if type(weight_sigma) is not int:
                    trainable_layer.weight.add_(u['weight'] * (weight_sigma / trainable_layer.scale_w.view(-1, 1, 1, 1)).round() )
                    trainable_layer.bias.add_(u['bias'] * (bias_sigma / trainable_layer.scale_x / trainable_layer.scale_w).round() )
                else:
                    trainable_layer.weight.add_(u['weight'] * weight_sigma)
                    trainable_layer.bias.add_(u['bias'] * bias_sigma)
                
                trainable_layer.weight.data = trainable_layer.weight.data.clamp(- 2 ** (trainable_layer.w_bit - 1), 2 ** (trainable_layer.w_bit - 1) - 1)
                trainable_layer.bias.data = trainable_layer.bias.data.clamp(- 2 ** (4*trainable_layer.w_bit - 1), 2 ** (4*trainable_layer.w_bit - 1) - 1)
                        
                _, pos_loss = self.obj_fn(starting_idx=block_idx, input=block_in, return_loss_reduction='mean')
                
                trainable_layer.weight.copy_(old_param['weight'])
                trainable_layer.bias.copy_(old_param['bias'])

                ### Estimate gradient
                if self.estimate_method == 'forward':
                    if type(weight_sigma) is not int:
                        trainable_layer.weight.grad.add_((pos_loss - old_loss) / weight_sigma / n_sample * u['weight'] * trainable_layer.scale_w.view(-1, 1, 1, 1))
                        trainable_layer.bias.grad.add_((pos_loss - old_loss) / bias_sigma / n_sample * u['bias'] * trainable_layer.scale_x * trainable_layer.scale_w)
                    else:
                        trainable_layer.weight.grad.add_((pos_loss - old_loss) / weight_sigma / n_sample * u['weight'])
                        trainable_layer.bias.grad.add_((pos_loss - old_loss) / bias_sigma / n_sample * u['bias'])
                            
                elif self.estimate_method == 'antithetic':
                    if type(weight_sigma) is not int:
                        trainable_layer.weight.sub_(u['weight'] * (weight_sigma / trainable_layer.scale_w.view(-1, 1, 1, 1)).round())
                        trainable_layer.bias.sub_(u['bias'] * (bias_sigma / trainable_layer.scale_x / trainable_layer.scale_w).round())
                    else:
                        trainable_layer.weight.sub_(u['weight'] * weight_sigma)
                        trainable_layer.bias.sub_(u['bias'] * bias_sigma)
                    
                    trainable_layer.weight.data = trainable_layer.weight.data.clamp(- 2 ** (trainable_layer.w_bit - 1), 2 ** (trainable_layer.w_bit - 1) - 1)
                    trainable_layer.bias.data = trainable_layer.bias.data.clamp(- 2 ** (4*trainable_layer.w_bit - 1), 2 ** (4*trainable_layer.w_bit - 1) - 1)
                    
                    _, neg_loss = self.obj_fn(starting_idx=block_idx, input=block_in, return_loss_reduction='mean')
                    
                    trainable_layer.weight.copy_(old_param['weight'])
                    trainable_layer.bias.copy_(old_param['bias'])

                    if type(weight_sigma) is not int:
                        trainable_layer.weight.grad.add_((pos_loss - neg_loss) / 2 / weight_sigma / n_sample * u['weight'] * trainable_layer.scale_w.view(-1, 1, 1, 1))
                        trainable_layer.bias.grad.add_((pos_loss - neg_loss) / 2 / bias_sigma / n_sample * u['bias'] * trainable_layer.scale_x * trainable_layer.scale_w)
                    else:
                        trainable_layer.weight.grad.add_((pos_loss - neg_loss) / 2 / weight_sigma / n_sample * u['weight'])
                        trainable_layer.bias.grad.add_((pos_loss - neg_loss) / 2 / bias_sigma / n_sample * u['bias'])
                
            if hasattr(configs.ZO_Estim, 'scale'):
                if configs.ZO_Estim.scale == 'sqrt-dim':
                    # param.grad = param.grad * math.sqrt(self.n_sample / (param.numel() - 1))
                    # param.grad = param.grad * math.sqrt(self.n_sample / (self.n_sample + param.numel() + 1))
                    trainable_layer.weight.grad = trainable_layer.weight.grad * math.sqrt(self.n_sample / (self.n_sample + dimension - 1))
                    trainable_layer.bias.grad = trainable_layer.bias.grad * math.sqrt(self.n_sample / (self.n_sample + dimension - 1))
                elif configs.ZO_Estim.scale == 'dim':
                    trainable_layer.weight.grad = trainable_layer.weight.grad * (self.n_sample / (self.n_sample + dimension - 1))
                    trainable_layer.bias.grad = trainable_layer.bias.grad * (self.n_sample / (self.n_sample + dimension - 1))
                elif type(configs.ZO_Estim.scale) is int:
                    trainable_layer.weight.grad = trainable_layer.weight.grad / configs.ZO_Estim.scale
                    trainable_layer.bias.grad = trainable_layer.bias.grad / configs.ZO_Estim.scale
                else:
                    raise NotImplementedError(f'Unknown {configs.ZO_Estim.scale}')
                            
                        
        return None
    
    def get_layer_param_ZO_gradient(self, block_idx, trainable_layer, block_in, old_loss, trainable_param_list, estimate_method, sample_method):        
        for trainable_param_name in trainable_param_list:
            if trainable_param_name == 'weight+bias':
                self.get_weight_and_bias_ZO_gradient(block_idx, trainable_layer, block_in, old_loss)
            else:
                if hasattr(trainable_layer, trainable_param_name) == False:
                    break
                else:
                    param = getattr(trainable_layer, trainable_param_name)
                    if isinstance(self.sigma, dict):
                        sigma = self.sigma[trainable_param_name]
                    else:
                        sigma = self.sigma
                    
                    # if trainable_param_name == 'scale_w':
                    #     self.get_scale_w_ZO_gradient(block_idx, trainable_layer, param, block_in, old_loss, sigma, estimate_method, sample_method)
                    # else:
                    if hasattr(configs.ZO_Estim, 'independent_wp') and configs.ZO_Estim.independent_wp is True:
                        param_ZO_grad = self.get_single_param_ZO_gradient_independent(block_idx, trainable_layer, param, block_in, old_loss, sigma, estimate_method, sample_method)
                    else: 
                        param_ZO_grad = self.get_single_param_ZO_gradient(block_idx, trainable_layer, param, block_in, old_loss, sigma, estimate_method, sample_method)
                    
                    if param.grad is None:
                        param.grad = param_ZO_grad
                    else:
                        param.grad += param_ZO_grad

        return None
    
    def get_param_ZO_gradient(self, old_loss, verbose=False):
        for trainable_layer_name in self.trainable_layer_list:
            trainable_layer_name = trainable_layer_name.split('.')
            block_name = f'{trainable_layer_name[0]}.{trainable_layer_name[1]}'

            for splited_block in self.splited_block_list:
                if splited_block.name == block_name:
                    # splited_layer = splited_block
                    break
                
            if 'conv' in trainable_layer_name:
                conv_idx = int(trainable_layer_name[3])
            else:
                conv_idx = None
            
            block_idx = splited_block.idx
            block_in = self.obj_fn(ending_idx=splited_block.idx, return_loss_reduction='no_loss')
            
            ##### Estimate gradient
            
            if conv_idx == None:
                ##### block
                for conv_idx in range(len(splited_block.block.conv)):
                    ##### Estimate gradient
                    trainable_layer = splited_block.block.conv[conv_idx]             
                    ##### conv update
                    if configs.ZO_Estim.param_update_method == 'layerwise':
                        ##### layerwise update
                        this_layer = splited_block.block.conv[conv_idx]
                        lr = self.param_lr
                        this_layer.weight.data.sub_( (lr * this_layer.weight.grad.data / this_layer.scale_w.view(-1, 1, 1, 1) ** 2).round().clamp(- 2 ** (this_layer.w_bit - 1), 2 ** (this_layer.w_bit - 1) - 1) )
                        this_layer.bias.data.sub_( (lr * this_layer.bias.grad.data / (this_layer.scale_x * this_layer.scale_w) ** 2).round().clamp(- 2 ** (4*this_layer.w_bit - 1), 2 ** (4*this_layer.w_bit - 1) - 1) )
            else:
                ##### layer
                trainable_layer = splited_block.block.conv[conv_idx] 
                self.get_layer_param_ZO_gradient(block_idx, trainable_layer, block_in, old_loss, self.trainable_param_list, self.estimate_method, self.sample_method)

                if configs.ZO_Estim.param_update_method == 'layerwise':
                    ##### layerwise update
                    this_layer = splited_block.block.conv[conv_idx]
                    lr = self.param_lr
                    this_layer.weight.data.sub_( (lr * this_layer.weight.grad.data / this_layer.scale_w.view(-1, 1, 1, 1) ** 2).round().clamp(- 2 ** (this_layer.w_bit - 1), 2 ** (this_layer.w_bit - 1) - 1) )
                    this_layer.bias.data.sub_( (lr * this_layer.bias.grad.data / (this_layer.scale_x * this_layer.scale_w) ** 2).round().clamp(- 2 ** (4*this_layer.w_bit - 1), 2 ** (4*this_layer.w_bit - 1) - 1) )
                    
        return None
    
    def get_mixture_ZO_gradient(self, old_loss, verbose=False):
        split_block_number = configs.ZO_Estim.split_block_number
        
        if self.estimate_method == 'forward':
            _, old_loss_vec = self.obj_fn(return_loss_reduction='none')
        
        for trainable_layer_name in self.trainable_layer_list:
            trainable_layer_name_split = trainable_layer_name.split('.')
            block_name = f'{trainable_layer_name_split[0]}.{trainable_layer_name_split[1]}'
            block_number = int(trainable_layer_name_split[1])

            for splited_block in self.splited_block_list:
                if splited_block.name == block_name:
                    # splited_layer = splited_block
                    break
                
            if 'conv' in trainable_layer_name:
                conv_idx = int(trainable_layer_name_split[3])
            else:
                conv_idx = None
            
            block_idx = splited_block.idx
            block_in = self.obj_fn(ending_idx=splited_block.idx, return_loss_reduction='no_loss')
            
            ##### Estimate gradient
            
            if conv_idx == None:
                raise NotImplementedError
            else:
                ### Weight Perturbation
                if block_number <= split_block_number:
                    ##### layer
                    trainable_layer = splited_block.block.conv[conv_idx] 
                    self.get_layer_param_ZO_gradient(block_idx, trainable_layer, block_in, old_loss, self.trainable_param_list, self.estimate_method, self.sample_method)
                
                ### Activation Perturbation    
                else:
                    if splited_block.type == QuantizedMbBlock:
                        ZO_grad, pre_activ, mask = self.get_layer_actv_ZO_gradint(splited_block, conv_idx, old_loss_vec, local_backward_args=True)
                        grad_x, grad_w, grad_bias = splited_block.block.conv[conv_idx].local_backward(input=pre_activ, grad_output=ZO_grad, binary_mask=mask.bool())
                        
                        # grad_w = grad_w / 10
                        # grad_bias = grad_bias / 10
                        
                        if configs.train_config.layerwise_update is None:
                            if splited_block.block.conv[conv_idx].weight.grad is None:
                                splited_block.block.conv[conv_idx].weight.grad = grad_w
                                splited_block.block.conv[conv_idx].bias.grad = grad_bias
                            else:
                                splited_block.block.conv[conv_idx].weight.grad += grad_w
                                splited_block.block.conv[conv_idx].bias.grad += grad_bias
                            # splited_block.block.conv[conv_idx].out_grad = ZO_grad
                    else:
                        raise NotImplementedError
        
        return None
    
    def create_fwd_hook_get_out_dimension(self):
        def fwd_hook(module, input, output):
            # input is a tuple
            module.out_dimension = output.numel() / output.shape[0]
        return fwd_hook
    
    def create_fwd_hook_save_input(self):
        def fwd_hook(module, input, output):
            # input is a tuple
            module.clean_input = input[0].detach().clone()
            module.clean_binary_mask = module.binary_mask
            module.out_dimension = output.numel() / output.shape[0]
            module.grad_output = None
        return fwd_hook
      
    def create_fwd_hook_add_perturbation(self, sigma):
        def fwd_hook(module, input, output):
            # input is a tuple
            # module.in_value = input[0].detach().clone()
            # output is a tensor
            module.sigma = sigma
            module.u_actv = self._sample_unit_sphere_quantized(output.shape, self.sample_method, self.device)
            output += sigma * module.u_actv * module.binary_mask
            output.data = output.data.clamp(- 2 ** (module.a_bit - 1), 2 ** (module.a_bit - 1) - 1)
        return fwd_hook
      
    def get_all_actv_ZO_gradient(self):
        ### save clean input
        fwd_hook_handle_list = []
        for name, module in self.model.named_modules():
            if any(keyword in name for keyword in self.trainable_layer_list):
                fwd_hook_save_input = self.create_fwd_hook_save_input()
                fwd_hook_handle_list.append(module.register_forward_hook(fwd_hook_save_input))
                
        _, old_loss = self.obj_fn(return_loss_reduction='none')
        
        batch_sz = len(old_loss)
                
        for fwd_hook_handle in fwd_hook_handle_list:
            fwd_hook_handle.remove()  
        
        dimension = self.ZO_dimension
        
        ### get perturbed loss
        fwd_hook_handle_list = []
        for name, module in self.model.named_modules():
            if any(keyword in name for keyword in self.trainable_layer_list):
                if type(self.sigma) is not int:
                    sigma = self.sigma / module.scale_y.view(-1, 1, 1, 1)
                    sigma = sigma.round()
                else:
                    sigma = self.sigma
                fwd_hook_add_perturbation = self.create_fwd_hook_add_perturbation(sigma)
                fwd_hook_handle_list.append(module.register_forward_hook(fwd_hook_add_perturbation))

        for i in range(self.n_sample):
            _, pos_loss = self.obj_fn(return_loss_reduction='none')
            
            if self.estimate_method == 'forward':
                loss_diff = (pos_loss - old_loss) / batch_sz / self.n_sample
            elif self.estimate_method == 'antithetic':
                for name, module in self.model.named_modules():
                    if any(keyword in name for keyword in self.trainable_layer_list):
                        module.sigma = - module.sigma
                
                _, neg_loss = self.obj_fn(return_loss_reduction='none')
                for name, module in self.model.named_modules():
                    if any(keyword in name for keyword in self.trainable_layer_list):
                        module.sigma = - module.sigma
                
                loss_diff = (pos_loss - neg_loss) / 2 / batch_sz / self.n_sample
            else:
                raise NotImplementedError
          
            ### accumulate grad_output
            for name, module in self.model.named_modules():
                if any(keyword in name for keyword in self.trainable_layer_list):
                    if module.grad_output is None:
                        module.grad_output = loss_diff.view(-1,1,1,1) * module.u_actv / self.sigma
                    else:
                        module.grad_output += loss_diff.view(-1,1,1,1) * module.u_actv / self.sigma
        
        if type(self.sigma) is not int:
            for name, module in self.model.named_modules():
                if any(keyword in name for keyword in self.trainable_layer_list):
                    module.grad_output = module.grad_output * module.scale_y
        
        for fwd_hook_handle in fwd_hook_handle_list:
            fwd_hook_handle.remove()      
        
        ### gradient scaling
        if hasattr(configs.ZO_Estim, 'scale'):

            if configs.ZO_Estim.scale == 'sqrt-dim':
                # grad_scale = math.sqrt(self.n_sample / (param.numel() - 1))
                # grad_scale = math.sqrt(self.n_sample / (self.n_sample + param.numel() + 1))
                grad_scale = math.sqrt((batch_sz * self.n_sample) / (batch_sz * self.n_sample + dimension - 1))
            elif configs.ZO_Estim.scale == 'dim':
                grad_scale = (batch_sz * self.n_sample) / (batch_sz * self.n_sample + dimension - 1)
            elif type(configs.ZO_Estim.scale) is int:
                grad_scale = configs.ZO_Estim.scale
            else:
                raise NotImplementedError(f'Unknown {configs.ZO_Estim.scale}')
        else:
            grad_scale = 1
            
        ### gradient estimation
        
        for name, module in self.model.named_modules():
            if any(keyword in name for keyword in self.trainable_layer_list):
                grad_x, grad_w, grad_bias = module.local_backward(input=module.clean_input, grad_output=grad_scale*module.grad_output, binary_mask=module.clean_binary_mask)
                module.weight.grad = grad_w
                module.bias.grad = grad_bias
                
                module.grad_output = None
                module.clean_input = None
                module.clean_binary_mask = None
                module.u_actv = None
                module.sigma = None
        
        return None
                
    
    def get_all_param_ZO_gradient(self, old_loss):
        dimension = self.ZO_dimension
        for name, param in self.model.named_parameters():
            if any(keyword in name for keyword in self.trainable_layer_list):
                param.grad = torch.zeros_like(param)
                
        if isinstance(self.sigma, dict):
            weight_sigma = self.sigma['weight']
            bias_sigma = self.sigma['bias']
        else:
            weight_sigma = self.sigma
            bias_sigma = self.sigma
        
        if self.sample_method == 'coord_basis':
            raise NotImplementedError
        else:
            u = dict()
            old_param = dict()
            n_sample = self.n_sample
            for i in range(n_sample):
                ### Generate random perturbation with the same shape as the parameter
                for name, module in self.model.named_modules():
                    if any(keyword in name for keyword in self.trainable_layer_list):
                        u[name+'.weight'] = self._sample_unit_sphere_quantized(module.weight.shape, self.sample_method, self.device)
                        u[name+'.bias'] = self._sample_unit_sphere_quantized(module.bias.shape, self.sample_method, self.device)
                        old_param[name+'.weight'] = module.weight.data.clone()
                        old_param[name+'.bias'] = module.bias.data.clone()
                # for name, param in self.model.named_parameters():
                #     if any(keyword in name for keyword in self.trainable_layer_list):
                #         u[name] = self._sample_unit_sphere_quantized(param.shape, self.sample_method, self.device)
                
                ### Add perturbation to the parameter
                # pos
                for name, module in self.model.named_modules():
                    if any(keyword in name for keyword in self.trainable_layer_list):
                        if type(weight_sigma) is not int:
                            module.weight.add_(u[name+'.weight'] * (weight_sigma / module.scale_w.view(-1, 1, 1, 1)).round() )
                            module.bias.add_(u[name+'.bias'] * (bias_sigma / module.scale_x / module.scale_w).round() )
                        else:
                            module.weight.add_(u[name+'.weight'] * weight_sigma)
                            module.bias.add_(u[name+'.bias'] * bias_sigma)
                        
                        module.weight.data = module.weight.data.clamp(- 2 ** (module.w_bit - 1), 2 ** (module.w_bit - 1) - 1)
                        module.bias.data = module.bias.data.clamp(- 2 ** (4*module.w_bit - 1), 2 ** (4*module.w_bit - 1) - 1)
                        
                # for name, param in self.model.named_parameters():
                #     if any(keyword in name for keyword in self.trainable_layer_list):
                #         param.add_(u[name] * p_sigma)
                    
                _, pos_loss = self.obj_fn()
                
                for name, module in self.model.named_modules():
                    if any(keyword in name for keyword in self.trainable_layer_list):
                        module.weight.copy_(old_param[name+'.weight'])
                        module.bias.copy_(old_param[name+'.bias'])

                ### Estimate gradient
                if self.estimate_method == 'forward':
                    for name, module in self.model.named_modules():
                        if any(keyword in name for keyword in self.trainable_layer_list):
                            if type(weight_sigma) is not int:
                                module.weight.grad.add_((pos_loss - old_loss) / weight_sigma / n_sample * u[name+'.weight'] * module.scale_w.view(-1, 1, 1, 1))
                                module.bias.grad.add_((pos_loss - old_loss) / bias_sigma / n_sample * u[name+'.bias'] * module.scale_x * module.scale_w)
                            else:
                                module.weight.grad.add_((pos_loss - old_loss) / weight_sigma / n_sample * u[name+'.weight'])
                                module.bias.grad.add_((pos_loss - old_loss) / bias_sigma / n_sample * u[name+'.bias'])
                            
                    # for name, param in self.model.named_parameters():
                    #     if any(keyword in name for keyword in self.trainable_layer_list):
                    #         param.sub_(u[name] * p_sigma)
                    #         param.grad += (pos_loss - old_loss) / self.sigma / n_sample * u[name]
                elif self.estimate_method == 'antithetic':
                    for name, module in self.model.named_modules():
                        if any(keyword in name for keyword in self.trainable_layer_list):
                            if type(weight_sigma) is not int:
                                module.weight.sub_(u[name+'.weight'] * (weight_sigma / module.scale_w.view(-1, 1, 1, 1)).round())
                                module.bias.sub_(u[name+'.bias'] * (bias_sigma / module.scale_x / module.scale_w).round())
                            else:
                                module.weight.sub_(u[name+'.weight'] * weight_sigma)
                                module.bias.sub_(u[name+'.bias'] * bias_sigma)
                            
                            module.weight.data = module.weight.data.clamp(- 2 ** (module.w_bit - 1), 2 ** (module.w_bit - 1) - 1)
                            module.bias.data = module.bias.data.clamp(- 2 ** (4*module.w_bit - 1), 2 ** (4*module.w_bit - 1) - 1)
                                
                    # for name, param in self.model.named_parameters():
                    #     if any(keyword in name for keyword in self.trainable_layer_list):
                    #         param.sub_(u[name] * p_sigma * 2)
                    
                    _, neg_loss = self.obj_fn()
                    
                    for name, module in self.model.named_modules():
                        if any(keyword in name for keyword in self.trainable_layer_list):
                            module.weight.copy_(old_param[name+'.weight'])
                            module.bias.copy_(old_param[name+'.bias'])

                            if type(weight_sigma) is not int:
                                module.weight.grad.add_((pos_loss - neg_loss) / 2 / weight_sigma / n_sample * u[name+'.weight'] * module.scale_w.view(-1, 1, 1, 1))
                                module.bias.grad.add_((pos_loss - neg_loss) / 2 / bias_sigma / n_sample * u[name+'.bias'] * module.scale_x * module.scale_w)
                            else:
                                module.weight.grad.add_((pos_loss - neg_loss) / 2 / weight_sigma / n_sample * u[name+'.weight'])
                                module.bias.grad.add_((pos_loss - neg_loss) / 2 / bias_sigma / n_sample * u[name+'.bias'])
                    # for name, param in self.model.named_parameters():
                    #     if any(keyword in name for keyword in self.trainable_layer_list):
                    #         param.add_(u[name] * p_sigma)
                    #         param.grad += (pos_loss - neg_loss) / 2 / self.sigma / n_sample * u[name]
                
            if hasattr(configs.ZO_Estim, 'scale'):
                for name, param in self.model.named_parameters():
                    if any(keyword in name for keyword in self.trainable_layer_list):
                        if configs.ZO_Estim.scale == 'sqrt-dim':
                            # param.grad = param.grad * math.sqrt(self.n_sample / (param.numel() - 1))
                            # param.grad = param.grad * math.sqrt(self.n_sample / (self.n_sample + param.numel() + 1))
                            param.grad = param.grad * math.sqrt(self.n_sample / (self.n_sample + dimension - 1))
                        elif configs.ZO_Estim.scale == 'dim':
                            param.grad = param.grad * (self.n_sample / (self.n_sample + dimension - 1))
                        elif type(configs.ZO_Estim.scale) is int:
                            param.grad = param.grad / configs.ZO_Estim.scale
                        else:
                            raise NotImplementedError(f'Unknown {configs.ZO_Estim.scale}')
                            
                        
        return None
    
    def update_obj_fn(self, obj_fn):
        self.obj_fn = obj_fn
    
    def get_forward_cnt(self):
        return self.forward_counter
    
    def update_param_lr(self, param_lr):
        self.param_lr = param_lr
    
    def estimate_grad(self, old_loss):
        
        # self.model.zero_grad()
        
        if self.perturb_method == 'mixture':
            self.estim_grads = self.get_mixture_ZO_gradient(old_loss=old_loss)
        # no old_loss: actv old_loss should use non-reduced loss
        elif self.perturb_method == 'activation':
            self.estim_grads = self.get_actv_ZO_gradient()
        
        elif self.perturb_method == 'param':
            self.estim_grads = self.get_param_ZO_gradient(old_loss=old_loss)
        
        elif self.perturb_method == 'all_param':
            self.estim_grads = self.get_all_param_ZO_gradient(old_loss=old_loss)
        
        elif self.perturb_method == 'all_activation':
            self.estim_grads = self.get_all_actv_ZO_gradient()

        else:
            raise ValueError('Unknown perturb_method')