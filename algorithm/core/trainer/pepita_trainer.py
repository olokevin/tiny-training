import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from quantize.quantized_ops_diff import QuantizedConv2dDiff
import math

from .base_trainer import BaseTrainer
from ..utils.basic import DistributedMetric, accuracy
from ..utils.config import configs
from ..utils.logging import logger
from ..utils import dist

# for conv models
def _compute_delta_w_conv(inp,out_diff,w_shape,stride=1,sqrt=False):
    delta_w = torch.zeros(w_shape, device=inp.device)
    ch_out = w_shape[0] # number of output channels
    size_out = out_diff.shape[-1] # size of output map
    ch_in = w_shape[1] # number of input channels
    size_in = inp.shape[-1] # size of input map
    ks = w_shape[2] # kernel height and width
    bs = out_diff.shape[0]
    cnt = 0
    for r in range(0,size_out): # loop over all the output rows
        for c in range(0,size_out): # loop over all the output columns
            #print(ch_out,size_out,ch_in,size_in,ks)
            #print("r,c",r,c)
            inp_r_start = stride*r
            inp_r_end = stride*r+ks
            inp_c_start = stride*c
            inp_c_end = stride*c+ks
            this_out_diff = out_diff[:,:,r,c]
            this_inp = inp[:,:,inp_r_start:inp_r_end,inp_c_start:inp_c_end]
            partial = ev(this_out_diff, this_inp, bs, ch_in, ch_out, ks).reshape_as(delta_w) # gives the right answer
            delta_w += partial
            cnt += 1
    if sqrt == False:
        delta_w *= 1./cnt          # dw = dw/n
    else:
        delta_w *= 1./math.sqrt(cnt)
    return delta_w

def ev(this_out_diff, this_inp, bs, chin, chout, ks):
    prod_mul = torch.mul(this_out_diff.reshape(bs,chout,1,1,1), this_inp.reshape(bs,1,chin,ks,ks))
    prod_mul = torch.mean(prod_mul,axis=0)  # average across batchsize
    return prod_mul

class PEPITATrainer(BaseTrainer):
    def __init__(self, 
                 model: nn.Module, 
                 data_loader, 
                 criterion, 
                 optimizer, 
                 lr_scheduler,
                 nin: int, 
                 nout: int, 
                 Bstd: float = 0.05,
                 trainable_layer_list: Optional[List[str]] = None,
                 ):
        super().__init__(model, data_loader, criterion, optimizer, lr_scheduler)
        
        self.Bstd = Bstd
        self.nin = nin
        self.nout = nout
        
        # Initialize random feedback matrix F
        self.F = torch.randn(nout, nin, device='cuda')
        self.F = self.F / torch.norm(self.F, dim=1, keepdim=True)
        
        sd = math.sqrt(6/nin)
        self.F = (torch.rand(nout, nin, device='cuda')*2*sd-sd)*Bstd  # mean zero
        
        # Register forward hooks for trainable layers
        self.handles = []
        self.activations = {}
        
        # Find trainable layers (QuantizedConv2dDiff)
        self.trainable_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, QuantizedConv2dDiff):
                if trainable_layer_list is None or name in trainable_layer_list:
                    self.trainable_layers.append(name)
    
    def _fwd_hook_save_activation(self, name: str):
        """Save input and output activations during standard forward pass."""
        def hook(module, input, output):
            self.activations[name] = {
                'input': input[0].detach(),
                'output': output.detach()
            }
        return hook
    
    def _fwd_hook_compute_grad(self, name: str):
        """Compute gradients during modulated forward pass."""
        def hook(module, input, output):
            # Get saved clean activations
            clean_input = self.activations[name]['input']
            clean_output = self.activations[name]['output']
            
            # Compute deltas
            delta_input = input[0] - clean_input
            delta_output = output - clean_output
            
            # Compute gradients for QuantizedConv2dDiff
            if isinstance(module, QuantizedConv2dDiff):
                # Compute weight gradient
                
                inp = (input[0] - module.zero_x) * module.scale_x
                out_diff = delta_output * module.scale_y
                
                weight_grad = _compute_delta_w_conv(
                    inp=inp,
                    out_diff=out_diff,
                    w_shape=module.weight.shape,
                    sqrt=True
                )
                
                # weight_grad = torch.zeros_like(module.weight)
                # for i in range(module.out_channels):
                #     # Process each output channel
                #     channel_delta = delta_output[:, i:i+1]  # [batch, 1, H, W]
                #     # Compute gradient for this filter
                #     grad = F.conv2d(delta_input, channel_delta, padding=module.padding)
                #     weight_grad[i] = grad.squeeze(0)  # [in_channels, kernel_size, kernel_size]
                
                # Compute bias gradient
                bias_grad = delta_output.sum(dim=(0, 2, 3))  # [out_channels]
                
                ### scale the gradients
                weight_grad = weight_grad * module.scale_w.view(-1, 1, 1, 1)
                bias_grad = bias_grad * module.scale_x * module.scale_w
                
                # Scale gradients by batch_size
                batch_size = delta_output.shape[0]
                weight_grad = weight_grad / batch_size
                bias_grad = bias_grad / batch_size
                
                # Save gradients
                # module.weight.grad = weight_grad
                module.bias.grad = bias_grad
        return hook
    
    def _project_error(self, error: torch.Tensor) -> torch.Tensor:
        """Project error to input space using random matrix F."""
        # error: [batch, nout]
        # F: [nout, nin]
        # Projected: [batch, nin]
        return torch.matmul(error, self.F)
    
    
    def train_one_epoch(self, epoch):
        self.model.train()
        self.data_loader['train'].sampler.set_epoch(epoch)

        train_loss = DistributedMetric('train_loss')
        train_top1 = DistributedMetric('train_top1')
        
        with tqdm(total=len(self.data_loader['train']),
                 desc='Train Epoch #{}'.format(epoch),
                 disable=dist.rank() > 0 or configs.ray_tune) as t:
            for batch_idx, (images, labels) in enumerate(self.data_loader['train']):
                images, labels = images.cuda(), labels.cuda()
                
                # Register hooks for standard forward pass
                save_handles = []
                for name, module in self.model.named_modules():
                    if name in self.trainable_layers:
                        handle = module.register_forward_hook(self._fwd_hook_save_activation(name))
                        save_handles.append(handle)
                
                # Standard forward pass
                with torch.no_grad():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Remove hooks after standard forward pass
                for handle in save_handles:
                    handle.remove()
                
                # Convert targets to one-hot encoding
                targets_one_hot = torch.zeros_like(outputs)
                targets_one_hot.scatter_(1, labels.unsqueeze(1).long(), 1)
                
                # Compute error and project to input space
                error = F.softmax(outputs, dim=1) - targets_one_hot
                
                ### INT8 qunatized scale
                # projected_error = self._project_error(error)
                projected_error = torch.matmul(error, self.F).view_as(images)
                
                # Modulated forward pass
                modulated_inputs = images / 127.5 + projected_error
                quant_scale = torch.max(modulated_inputs.abs()).item() / 127
                modulated_inputs = (modulated_inputs / quant_scale).round().clamp(-128, 127)
                
                # Register gradient computation hooks
                grad_handles = []
                for name, module in self.model.named_modules():
                    if name in self.trainable_layers:
                        handle = module.register_forward_hook(self._fwd_hook_compute_grad(name))
                        grad_handles.append(handle)
                
                # Forward pass with gradient computation
                with torch.no_grad():
                    self.model(modulated_inputs)
                
                # Remove gradient computation hooks
                for handle in grad_handles:
                    handle.remove()
                
                # Update parameters
                if hasattr(self.optimizer, 'pre_step'):  # for SGDScale optimizer
                    self.optimizer.pre_step(self.model)

                self.optimizer.step()

                if hasattr(self.optimizer, 'post_step'):  # for SGDScaleInt optimizer
                    self.optimizer.post_step(self.model)
                
                self.optimizer.zero_grad()  # or self.net.zero_grad()
                
                # Update metrics
                acc1 = accuracy(outputs, labels, topk=(1,))[0]
                train_loss.update(loss, images.shape[0])
                train_top1.update(acc1.item(), images.shape[0])

                t.set_postfix({
                    'loss': train_loss.avg.item(),
                    'top1': train_top1.avg.item(),
                    'batch_size': images.shape[0],
                    'img_size': images.shape[2],
                    'lr': self.optimizer.param_groups[0]['lr'],
                })
                t.update()

                # Update learning rate if needed
                if configs.run_config.iteration_decay == 1:
                    self.lr_scheduler.step()

        return {
            'train/top1': round(train_top1.avg.item(), 3),
            'train/loss': round(train_loss.avg.item(), 3),
            'train/lr': round(self.optimizer.param_groups[0]['lr'], 5),
        }
    
    def validate(self, data_set='val'):
        self.model.eval()
        val_criterion = self.criterion

        val_loss = DistributedMetric('val_loss')
        val_top1 = DistributedMetric('val_top1')

        with torch.no_grad():
            with tqdm(total=len(self.data_loader[data_set]),
                      desc='Validate',
                      disable=dist.rank() > 0 or configs.ray_tune) as t:
                for images, labels in self.data_loader[data_set]:
                    images, labels = images.cuda(), labels.cuda()
                    # compute output
                    output = self.model(images)
                    loss = val_criterion(output, labels)
                    val_loss.update(loss, images.shape[0])
                    acc1 = accuracy(output, labels, topk=(1,))[0]
                    val_top1.update(acc1.item(), images.shape[0])

                    t.set_postfix({
                        'loss': val_loss.avg.item(),
                        'top1': val_top1.avg.item(),
                        'batch_size': images.shape[0],
                        'img_size': images.shape[2],
                    })
                    t.update()
        return {
            'val/top1': round(val_top1.avg.item(), 3),
            'val/loss': round(val_loss.avg.item(), 3),
        }
        
    def cleanup(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = [] 