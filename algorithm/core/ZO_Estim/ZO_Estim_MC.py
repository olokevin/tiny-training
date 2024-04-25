from typing import Callable

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from quantize.quantized_ops_diff import QuantizedConv2dDiff as QuantizedConv2d
from quantize.quantized_ops_diff import QuantizedMbBlockDiff as QuantizedMbBlock
from quantize.quantized_ops_diff import _TruncateActivationRange

from scipy.stats import qmc
from .ZO_Estim_entry import split_model, split_named_model
from .QMC_sampler import sphere_n, coord_basis, block_mask_generator, layer_mask_generator

from ..utils.config import configs
from ..utils.logging import logger

DEBUG = None
# DEBUG = True

class SplitedBlock(nn.Module):
    def __init__(self, idx, name, block):
        super().__init__()
        self.idx = idx
        self.name = name
        self.block = block
        self.type = type(block)
        self.grad = None
    
    def update_grad(self, grad):
        self.grad = grad

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

        quantize_method: str = 'None',  # 'None', 'u_fp-grad_fp', 'u_fp-grad_int', 'u_int-grad_int', 'u_int-grad_fp'
        mask_method: str = 'None',
        estimate_method: str = 'forward',
        perturb_method: str = 'batch',
        sample_method: str = 'gaussian',
        prior_method: str = 'None'
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
        elif trainable_layer_list == 'all':
            for splited_block in self.splited_block_list:
                if type(splited_block.block) is QuantizedMbBlock:
                    for conv_idx in range(len(splited_block.block.conv)):
                        self.trainable_layer_list.append(f'{splited_block.name}.conv.{conv_idx}')
        elif 'last' in trainable_layer_list:
            n_block = int(trainable_layer_list.split('-')[1])
            for splited_block in self.splited_block_list[::-1]:
                if type(splited_block.block) is QuantizedMbBlock:
                    if n_block <= 0:
                        break
                    for conv_idx in range(len(splited_block.block.conv)):
                        self.trainable_layer_list.append(f'{splited_block.name}.conv.{conv_idx}')
                        n_block -= 1
                        if n_block <= 0:
                            break
        elif 'first' in trainable_layer_list:
            n_block = int(trainable_layer_list.split('-')[1])
            for splited_block in self.splited_block_list:
                if type(splited_block.block) is QuantizedMbBlock:
                    if n_block <= 0:
                        break
                    for conv_idx in range(len(splited_block.block.conv)):
                        self.trainable_layer_list.append(f'{splited_block.name}.conv.{conv_idx}')
                        n_block -= 1
                        if n_block <= 0:
                            break
        else:
            raise ValueError('Not supported trainable_layer_list')
        
        
        self.trainable_splited_block_list = []
        for splited_block in self.splited_block_list:
            if splited_block.name in self.trainable_layer_list:
                self.trainable_splited_block_list.append(splited_block)

        self.quantize_method = quantize_method
        
        self.estimate_method = estimate_method
        self.perturb_method = perturb_method
        self.sample_method = sample_method
        self.prior_method = prior_method

        self.device = next(self.model.parameters()).device
        self.dtype = next(self.model.parameters()).dtype


        # self.ZO_dimension = sum(p.numel() for p in filter(lambda x: x.requires_grad, self.model.parameters()))
        self.ZO_dimension = sum(p.numel() for name, p in self.model.named_parameters() if name in self.trainable_param_list)
        print('ZO_dimension=', self.ZO_dimension)

        if 'block' in mask_method:
            self.mask_method = mask_method.split('-')[0]
            self.n_blocks = int(mask_method.split('-')[1])
        elif mask_method == 'layer':
            self.mask_method = mask_method
            self.n_blocks = len(self.trainable_param_list)
            print('n_blocks= ', self.n_blocks)
        else:
            self.mask_method = 'None'
        
        if self.sample_method == 'coord_basis':
            self.n_sample = self.ZO_dimension
        
        self.forward_counter = 0

        self.mask = None
        self.mask_generator = None

        self.sampler = None

        self.prior_mean = None
        self.prior_std = torch.zeros(self.ZO_dimension, device=self.device)

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
        
        if sample_method == 'bernoulli':
            sample = torch.ones(shape, device=device) - 2*torch.bernoulli(0.5*torch.ones(shape, device=device))
        else:
            return NotImplementedError('Unlnown sample method', self.sample_method)
        
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
                # Add perturbation to the parameter
                param.data.add_(perturbation)
                u_idx += param_len 
  
    def _init_mask_generator(self):
        if self.mask_method == 'block':
            return block_mask_generator(self.ZO_dimension, self.n_blocks, self.device)
        elif self.mask_method == 'layer':
            return layer_mask_generator(self.model, self.ZO_dimension, self.trainable_param_list, self.device)
        else:
            return None
    
    def get_mask(self):
        if self.mask_generator is not None:
            return next(self.mask_generator)
        else:
            return None
    
    def apply_mask(self, mask):
        if mask is not None:
            if mask.numel() != self.ZO_dimension:
                raise ValueError('the length of the mask should be the same as the dimension')
            else:
                self.mask = mask

    def apply_prior(self, prior_mean=None, prior_std=None): 
        if prior_mean is not None:
            if prior_mean.numel() != self.ZO_dimension:
                raise ValueError('the length of the mask should be the same as the dimension')
            else:
                self.prior_mean = prior_mean
        if prior_std is not None:
            if prior_std.numel() != self.ZO_dimension:
                raise ValueError('the length of the mask should be the same as the dimension')
            else:
                self.prior_std = prior_std
    
    def update_prior_mean(self, grad_vec):
        if grad_vec is not None:
            if 'past_grad' in self.prior_method:
                self.prior_mean = self.prior_mean = None
            elif 'mov_avg' in self.prior_method:
                momentum = 0.9
                self.prior_mean.mul_(momentum).add_(1 - momentum, F.normalize(grad_vec, p=2, dim=0))

    def get_batch_ZO_gradient(self, old_loss, verbose=False):
        dimension = self.ZO_dimension
        device = self.device
        ZO_grad = torch.zeros(dimension, device=device)

        self.sampler = self._init_sampler(dimension=dimension)
        self.mask_generator = self._init_mask_generator()
        if self.mask_generator is not None:
            n_blocks = self.n_blocks
        else:
            n_blocks = 1
        
        loss_diff_seq = torch.zeros(self.n_sample, device=device)

        for i in range(n_blocks):
            mask = self.get_mask()
            self.apply_mask(mask)

            for n in range(self.n_sample):
                # print(n)
                if 'perturb' in self.prior_method:
                    if n == 0:
                        u = torch.nn.functional.normalize(self.prior_mean, p=2, dim=0)
                    else:
                        u = self._sample_unit_sphere(dimension, device)
                elif 'neighbor' in self.prior_method:
                    # u = self.prior_mean + torch.randn(dimension, device=device) * self.prior_std
                    # u = self.prior_mean
                    u = self.prior_mean - self.prior_std * 0.1
                    # u = self.prior_mean - self.prior_std * 0.1 + torch.randn(dimension, device=device) * 0.01
                    u = torch.nn.functional.normalize(u, p=2, dim=0)
                else:
                    u = self._sample_unit_sphere(dimension, device)
                
                if self.mask is not None:
                    u = u * self.mask
                
                # Add perturbation
                self._add_params_perturbation(self.sigma, u)
                
                _, pos_loss = self.obj_fn()
                self.forward_counter += 1
                # remove perturbation
                self._add_params_perturbation(self.sigma, -u)

                if self.estimate_method == 'one_point':
                    loss_diff = pos_loss
                elif self.estimate_method == 'forward':
                    loss_diff = pos_loss - old_loss
                elif self.estimate_method == 'antithetic':
                    # neg perturbation
                    self._add_params_perturbation(self.sigma, -u)

                    _, neg_loss = self.obj_fn()
                    self.forward_counter += 1
                    # remove perturbation
                    self._add_params_perturbation(self.sigma, u)

                    loss_diff = (pos_loss - neg_loss) / 2

                loss_diff = loss_diff / self.sigma

                loss_diff_seq[n] = loss_diff
                
                # ZO_grad = ZO_grad + dimension * loss_diff * u / self.n_sample
                if self.sample_method == 'coord_basis':
                    ZO_grad = ZO_grad + loss_diff * u
                else:
                    ZO_grad = ZO_grad + loss_diff * u / self.n_sample

        if verbose == False:
            return ZO_grad
        else:
            return ZO_grad, loss_diff_seq
    
    
    def get_actv_ZO_gradient(self, verbose=False):
        for trainable_layer_name in self.trainable_layer_list:
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
                ZO_grad, pre_activ, mask = self.get_block_actv_ZO_gradint(splited_block, local_backward_args=True)
            # Update single conv layer
            else:
                ZO_grad, pre_activ, mask = self.get_layer_actv_ZO_gradint(splited_block, conv_idx, local_backward_args=True)
            
            ##### Update gradient
            batch_sz = ZO_grad.shape[0]
            
            if splited_block.type == nn.Linear:
                splited_block.block.weight.grad = torch.matmul(ZO_grad.T, pre_activ) / batch_sz  # average over all batch!
                splited_block.block.bias.grad = torch.mean(ZO_grad, dim=0)
            elif splited_block.type == QuantizedMbBlock:
                if conv_idx == None:
                    if DEBUG:
                        FO_grad = splited_block.block.out_grad
                        FO_grad = FO_grad * mask

                        if splited_block.block.q_add is not None:
                            scale_y = splited_block.block.q_add.scale_y
                            scale_x = splited_block.block.q_add.scale_x1
                            effective_scale = splited_block.block.conv[-1].effective_scale
                        else:
                            scale_y = splited_block.block.conv[-1].y_scale
                            scale_x = splited_block.block.conv[-1].x_scale
                            effective_scale = splited_block.block.conv[-1].effective_scale

                        print('\nGrad Norm')
                        print('FO_grad:', torch.linalg.norm(FO_grad))
                        print('ZO_grad:', torch.linalg.norm(ZO_grad))

                        print('\nZO grad estimate')
                        print('cos sim grad_output', F.cosine_similarity(FO_grad.view(-1), ZO_grad.view(-1), dim=0))

                        # normalize_FO_grad = F.normalize(FO_grad.view(batch_sz, -1), p=2, dim=1)
                        # normalize_ZO_grad = F.normalize(ZO_grad.view(batch_sz, -1), p=2, dim=1)
                        # relative_error = torch.norm(normalize_FO_grad - normalize_ZO_grad, dim=1).mean()
                        # print('relative_error: ', relative_error)
                        # print('relative_error / √d:', relative_error/math.sqrt(FO_grad.numel()))

                        print('\nscaled grad')
                        # print('FO_grad / scale_y:', torch.linalg.norm(FO_grad / scale_y))
                        # print('FO_grad / scale_y**2:', torch.linalg.norm(FO_grad / scale_y**2))

                        print('ZO_grad * scale_y:', torch.linalg.norm(ZO_grad * scale_y))
                        print('ZO_grad * scale_x:', torch.linalg.norm(ZO_grad * scale_x))
                        print('ZO_grad * effective_scale:', torch.linalg.norm(ZO_grad * effective_scale.view(1, -1, 1, 1).cuda()))

                else:
                    effective_scale = splited_block.block.conv[conv_idx].effective_scale.view(1, -1, 1, 1).cuda()
                    # ZO_grad = ZO_grad * mask
                    # ZO_grad = ZO_grad * effective_scale
                    grad_x, grad_w, grad_bias = splited_block.block.conv[conv_idx].local_backward(input=pre_activ, grad_output=ZO_grad)
                    
                    if DEBUG:
                        FO_grad = splited_block.block.conv[conv_idx].out_grad
                        FO_grad = FO_grad * mask

                        FO_grad_x, FO_grad_w, FO_grad_bias = splited_block.block.conv[conv_idx].local_backward(input=pre_activ, grad_output=FO_grad)
                        true_grad_w = splited_block.block.conv[conv_idx].weight.grad
                        true_grad_bias = splited_block.block.conv[conv_idx].bias.grad
                        if conv_idx == 0:
                            true_grad_x = self.splited_block_list[splited_block.idx-1].block.out_grad
                        else:
                            true_grad_x = splited_block.block.conv[conv_idx-1].out_grad

                        print('\n function of local_backward')
                        print('cos sim grad_w', F.cosine_similarity(FO_grad_w.view(-1), true_grad_w.view(-1), dim=0))
                        print('cos sim grad_bias', F.cosine_similarity(FO_grad_bias.view(-1), true_grad_bias.view(-1), dim=0))
                        print('cos sim grad_x', F.cosine_similarity(FO_grad_x.view(-1), true_grad_x.view(-1), dim=0))

                        mean_FO_grad_x, mean_FO_grad_w, mean_FO_grad_bias = splited_block.block.conv[conv_idx].local_backward(input=torch.mean(pre_activ, dim=0).unsqueeze(0), grad_output=torch.mean(FO_grad, dim=0).unsqueeze(0))
                        print('\n function of batch_meaned local_backward')
                        print('cos sim grad_w', F.cosine_similarity(mean_FO_grad_w.view(-1), true_grad_w.view(-1), dim=0))
                        print('cos sim grad_bias', F.cosine_similarity(mean_FO_grad_bias.view(-1), true_grad_bias.view(-1), dim=0))

                        print('\n ZO grad estimate')
                        print('cos sim grad_output', F.cosine_similarity(FO_grad.view(-1), ZO_grad.view(-1), dim=0))
                        pruned_FO_grad=FO_grad[ZO_grad!=0]
                        pruned_ZO_grad=ZO_grad[ZO_grad!=0]
                        print('FO_grad non_zero:', torch.linalg.norm(pruned_FO_grad))
                        print('ZO_grad non_zero:', torch.linalg.norm(pruned_ZO_grad))
                        print('cos sim grad_output non_zero', F.cosine_similarity(pruned_FO_grad.view(-1), pruned_ZO_grad.view(-1), dim=0))

                        print('\n Grad Norm')
                        print('FO_grad:', torch.linalg.norm(FO_grad))
                        print('ZO_grad:', torch.linalg.norm(ZO_grad))
                        print('FO_grad / effective_scale:', torch.linalg.norm(FO_grad / effective_scale))
                        print('FO_grad / effective_scale**2:', torch.linalg.norm(FO_grad / effective_scale**2))
                        print('ZO_grad * effective_scale:', torch.linalg.norm(ZO_grad * effective_scale))
                        print('ZO_grad / effective_scale:', torch.linalg.norm(ZO_grad / effective_scale))

                    splited_block.block.conv[conv_idx].weight.grad = grad_w
                    splited_block.block.conv[conv_idx].bias.grad = grad_bias
            else:
                raise NotImplementedError('Unknown block type')      
        return None

    def get_layer_actv_ZO_gradint(self, splited_block, conv_idx, local_backward_args=False):
        assert splited_block.type == QuantizedMbBlock
        block_in = self.obj_fn(ending_idx=splited_block.idx, return_loss_reduction='no_loss')
        pre_activ = splited_block.block.conv[:conv_idx](block_in)
        post_actv = splited_block.block.conv[conv_idx](pre_activ)
        copy_post_actv = post_actv.clone()

        assert type(self.sigma) is int
        mask = splited_block.block.conv[conv_idx].binary_mask.int()
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
                if splited_block.type == QuantizedMbBlock:
                    a_bit = splited_block.block.conv[conv_idx].a_bit
                    post_actv.clamp(- 2 ** (a_bit - 1), 2 ** (a_bit - 1) - 1)
                
                pos_distance = post_actv[:, i] - org_post_actv

                block_out = splited_block.block.conv[conv_idx+1:](post_actv.view(post_actv_shape))
                if splited_block.block.q_add is not None:
                    block_out = splited_block.block.q_add(block_in, block_out)
                block_out = _TruncateActivationRange.apply(block_out, splited_block.block.a_bit)

                _, pos_loss = self.obj_fn(starting_idx=splited_block.idx+1, input=block_out, return_loss_reduction='none')
                self.forward_counter += 1

                if self.estimate_method == 'forward':
                    post_actv[:, i] = org_post_actv
                    _, old_loss = self.obj_fn(return_loss_reduction='none')

                    for batch_idx in range(batch_sz):
                        if ((pos_loss[batch_idx] - old_loss[batch_idx]) != 0) & (pos_distance[batch_idx] != 0):
                            ZO_grad[batch_idx,i] = (pos_loss[batch_idx] - neg_loss[batch_idx]) / pos_distance[batch_idx]
                        else:
                            ZO_grad[batch_idx,i] = 0

                    post_actv[:, i] = org_post_actv
                elif self.estimate_method == 'antithetic':
                    post_actv[:, i] = org_post_actv
                    post_actv[:, i] = post_actv[:, i] - mask[:, i] * self.sigma
                    if splited_block.type == QuantizedMbBlock:
                        a_bit = splited_block.block.conv[conv_idx].a_bit
                        post_actv.clamp(- 2 ** (a_bit - 1), 2 ** (a_bit - 1) - 1)
                    
                    neg_distance = org_post_actv - post_actv[:, i]
                    
                    block_out = splited_block.block.conv[conv_idx+1:](post_actv.view(post_actv_shape))
                    if splited_block.block.q_add is not None:
                        block_out = splited_block.block.q_add(block_in, block_out)
                    block_out = _TruncateActivationRange.apply(block_out, splited_block.block.a_bit)

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
            
            ZO_grad = ZO_grad.view(post_actv_shape)
            mask = mask.view(post_actv_shape)
        else:
            raise NotImplementedError('Unknown sample method')
        
        if local_backward_args == True:
            return ZO_grad, pre_activ, mask
        else:
            return ZO_grad
    
    def get_block_actv_ZO_gradint(self, splited_block, local_backward_args=False):
        assert splited_block.type == QuantizedMbBlock

        pre_activ = self.obj_fn(ending_idx=splited_block.idx, return_loss_reduction='no_loss')
        post_actv = splited_block.block(pre_activ)

        assert type(self.sigma) is int
        assert hasattr(splited_block.block, 'binary_mask')
        mask = splited_block.block.binary_mask.int()
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
                post_actv.clamp(- 2 ** (a_bit - 1), 2 ** (a_bit - 1) - 1)
                
                pos_distance = post_actv[:, i] - org_post_actv

                _, pos_loss = self.obj_fn(starting_idx=splited_block.idx+1, input=post_actv.view(post_actv_shape), return_loss_reduction='none')
                self.forward_counter += 1

                if self.estimate_method == 'forward':
                    post_actv[:, i] = org_post_actv
                    _, old_loss = self.obj_fn(return_loss_reduction='none')

                    for batch_idx in range(batch_sz):
                        if ((pos_loss[batch_idx] - old_loss[batch_idx]) != 0) & (pos_distance[batch_idx] != 0):
                            ZO_grad[batch_idx,i] = (pos_loss[batch_idx] - neg_loss[batch_idx]) / pos_distance[batch_idx]
                        else:
                            ZO_grad[batch_idx,i] = 0

                elif self.estimate_method == 'antithetic':
                    post_actv[:, i] = org_post_actv

                    post_actv[:, i] = post_actv[:, i] - mask[:, i] * self.sigma
                    a_bit = splited_block.block.a_bit
                    post_actv.clamp(- 2 ** (a_bit - 1), 2 ** (a_bit - 1) - 1)
                    
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
            
            ZO_grad = ZO_grad.view(post_actv_shape)
            mask = mask.view(post_actv_shape)
        elif self.sample_method == 'bernoulli':         
            org_post_actv = post_actv[:, i].int()

            for i in range(self.n_sample):
                u = mask * self._sample_unit_sphere_quantized(post_actv_shape, self.sample_method, self.device)

                post_actv = post_actv + u * self.sigma

                # a_bit = splited_block.block.a_bit
                # post_actv.clamp(- 2 ** (a_bit - 1), 2 ** (a_bit - 1) - 1)
                # pos_distance = post_actv - org_post_actv

                _, pos_loss = self.obj_fn(starting_idx=splited_block.idx+1, input=post_actv.view(post_actv_shape), return_loss_reduction='none')
                self.forward_counter += 1

                if self.estimate_method == 'forward':
                    post_actv[:, i] = org_post_actv
                    _, old_loss = self.obj_fn(return_loss_reduction='none')
                    
                    ZO_grad += (pos_loss - old_loss) / self.sigma * u

                elif self.estimate_method == 'antithetic':
                    post_actv[:, i] = org_post_actv
                    post_actv[:, i] = post_actv[:, i] - u * self.sigma

                    # a_bit = splited_block.block.a_bit
                    # post_actv.clamp(- 2 ** (a_bit - 1), 2 ** (a_bit - 1) - 1)
                    # neg_distance = org_post_actv - post_actv[:, i]
                
                    _, neg_loss = self.obj_fn(starting_idx=splited_block.idx+1, input=post_actv.view(post_actv_shape), return_loss_reduction='none')
                    self.forward_counter += 1
                    
                    ZO_grad += (pos_loss - old_loss) / self.sigma * u

                    post_actv[:, i] = org_post_actv
              
            ZO_grad = ZO_grad.view(post_actv_shape) / self.n_sample
            mask = mask.view(post_actv_shape)


        else:
            raise NotImplementedError('Unknown sample method')
        
        if local_backward_args == True:
            return ZO_grad, pre_activ, mask
        else:
            return ZO_grad

    # def get_break_ZO_grad(self):
    #     for i in range(len(self.trainable_splited_block_list)):
    #         splited_block = self.trainable_splited_block_list[i]
    #         with torch.no_grad():
    #             ZO_grad = self.get_block_actv_ZO_gradint(splited_block)
            
    #         if i>0:
    #             detach_idx = self.trainable_splited_block_list[i-1].idx
    #         else:
    #             detach_idx = None
    #         output = self.obj_fn(ending_idx=splited_block.idx, return_loss_reduction='no_loss', detach_idx=detach_idx)
    #         # output.grad = ZO_grad
    #         output.backward(ZO_grad)
        
    #     _, loss = self.obj_fn(detach_idx=self.trainable_splited_block_list[-1].idx)
    #     loss.backward()
        
    #     return None

    def get_break_ZO_grad(self):
        for i in range(len(self.trainable_splited_block_list)):
            splited_block = self.trainable_splited_block_list[i]

            if splited_block.block.q_add is not None:
                scale_y = splited_block.block.q_add.scale_y
            else:
                scale_y = splited_block.block.conv[-1].y_scale
              
            with torch.no_grad():
                ZO_grad = self.get_block_actv_ZO_gradint(splited_block) * scale_y
            
            if i==0:
                output = self.obj_fn(ending_idx=splited_block.idx+1, return_loss_reduction='no_loss') 
            else:
                output = self.obj_fn(starting_idx=self.trainable_splited_block_list[i-1].idx+1, ending_idx=splited_block.idx+1, input=output, return_loss_reduction='no_loss')
            
            output.backward(ZO_grad)
            output = output.detach()
        
        _, loss = self.obj_fn(starting_idx=self.trainable_splited_block_list[-1].idx+1, input=output)
        loss.backward()
        
        return None    

    def get_single_param_ZO_gradint(self, splited_block, param, block_in, old_loss, sigma, estimate_method, sample_method):
        param_dim = param.numel()
        param_vec = param.view(-1)
        param_shape = param.shape

        param_ZO_grad = torch.zeros_like(param_vec, device=self.device)

        # sigma = torch.mean(param)
        # print(torch.mean(param))

        if sample_method == 'coord_basis':
            for i in range(param_dim):
                old_param_vec = param_vec[i] * 1
                # pos
                param_vec[i] = param_vec[i] + sigma
                _, pos_loss = self.obj_fn(starting_idx=splited_block.idx, input=block_in, return_loss_reduction='mean')
                param_vec[i] = param_vec[i] - sigma

                # neg
                if estimate_method == 'forward':
                    param_ZO_grad[i] = (pos_loss - old_loss) / sigma
                elif estimate_method == 'antithetic':
                    param_vec[i] = param_vec[i] - sigma
                    _, neg_loss = self.obj_fn(starting_idx=splited_block.idx, input=block_in, return_loss_reduction='mean')
                    param_vec[i] = param_vec[i] + sigma

                    param_ZO_grad[i] = (pos_loss - neg_loss) / 2 / sigma
                else:
                    raise NotImplementedError('Unknown estimate method')
        else:
            return NotImplementedError('sample method not implemented yet')
        
        param_ZO_grad = param_ZO_grad.view(param_shape)
        return param_ZO_grad
    
    
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
                trainable_layer = splited_block.block.conv[conv_idx] 
            else:
                conv_idx = None
                trainable_layer = splited_block.block
            
            block_in = self.obj_fn(ending_idx=splited_block.idx, return_loss_reduction='no_loss')
            
            ##### Estimate gradient
            for trainable_param_name in self.trainable_param_list:
                trainable_param = getattr(trainable_layer, trainable_param_name)
                if isinstance(self.sigma, dict):
                    sigma = self.sigma[trainable_param_name]
                else:
                    sigma = self.sigma
                trainable_param_ZO_grad = self.get_single_param_ZO_gradint(splited_block, trainable_param, block_in, old_loss, sigma, self.estimate_method, self.sample_method)
                if type(trainable_param) is nn.Parameter:
                    trainable_param.grad = trainable_param_ZO_grad
                else:
                    lr = configs.run_config.base_lr
                    trainable_param = trainable_param - lr * trainable_param_ZO_grad

        
        return None

    
    # logger.info(f'layer {splited_block.name}.conv.{conv_idx}, out-of-range: {torch.sum(splited_block.block.conv[conv_idx].OOR_mask)}/{torch.numel(splited_block.block.conv[conv_idx].OOR_mask)} = {torch.sum(splited_block.block.conv[conv_idx].OOR_mask)/torch.numel(splited_block.block.conv[conv_idx].OOR_mask)}')
    # logger.info(f'layer {splited_block.name}.conv.{conv_idx}, scale_grad_norm: {torch.norm(effective_scale_grad)}, scale_norm: {torch.norm(old_effective_scale)}')
    
    def update_obj_fn(self, obj_fn):
        self.obj_fn = obj_fn
    
    def get_forward_cnt(self):
        return self.forward_counter
    
    def estimate_grad(self):
        
        if DEBUG:
            pass
        else:
            self.model.zero_grad()
        
        if self.perturb_method == 'batch':
            outputs, old_loss = self.obj_fn()
            self.estim_grads = self.get_batch_ZO_gradient(old_loss)
        
        elif self.perturb_method == 'activation':
            outputs, old_loss = self.obj_fn()
            self.estim_grads = self.get_actv_ZO_gradient()
        
        elif self.perturb_method == 'param':
            outputs, old_loss = self.obj_fn()
            self.estim_grads = self.get_param_ZO_gradient(old_loss)
      
        elif self.perturb_method == 'single':
            outputs, old_loss = self.obj_fn(row=-1)
            self.estim_grads = self.get_ZO_gradient_single(old_loss)
            old_loss = torch.mean(old_loss)
        else:
            raise ValueError('Unknown perturb_method')
        
        # if self.prior_method != 'None':
        #     self.update_prior_mean(self.estim_grads)
        
        return outputs, old_loss, self.estim_grads
    
    def update_grad(self):
        if self.perturb_method == 'batch':
            u_idx = 0
            for name, param in self.model.named_parameters():
                if name in self.trainable_param_list:
                    # extract cooresponding grad vector
                    param_len = param.numel()
                    param_shape = param.shape
                    param_grad = self.estim_grads[u_idx : u_idx+param_len].reshape(param_shape)
                    # Update param.grad
                    # if 'u_fp' in self.quantize_method:

                    # name_list = name.split('.')
                    # w_scale = torch.tensor(self.model[int(name_list[0])][int(name_list[1])].conv[int(name_list[3])].w_scale, device=self.device).view(-1, 1, 1, 1)
                    # param_grad = param_grad * w_scale

                    if 'grad_int' in self.quantize_method:
                        param_grad = param_grad.round()
                    
                    param.grad = param_grad
                    u_idx += param_len
        else:
            pass
    

    # def get_ZO_gradient_single(self, old_loss, verbose=False):
    #     dimension = self.ZO_dimension
    #     device = self.device
    #     batch_sz = self.obj_fn(row=0, get_batch_sz=True)
    #     ZO_grad = torch.zeros(dimension, device=device)

    #     self.sampler = self._init_sampler(dimension=dimension)

    #     if self.sample_method == 'coord_basis':
    #         N_ZO = dimension
    #     else:
    #         N_ZO = self.n_sample

    #     loss_diff_seq = torch.zeros(N_ZO, device=device)

    #     for n in range(N_ZO):
    #         for i_row in range(batch_sz):
    #             u = self._sample_unit_sphere(dimension, device)
    #             # Add perturbation
    #             self._add_params_perturbation(self.sigma, u)
                
    #             self.model.zero_grad()
    #             with torch.no_grad():
    #                 _, pos_loss = self.obj_fn(row=i_row)
    #             self.forward_counter += 1
    #             # remove perturbation
    #             self._add_params_perturbation(self.sigma, -u)

    #             if self.estimate_method == 'one_point':
    #                 loss_diff = pos_loss
    #             elif self.estimate_method == 'forward':
    #                 loss_diff = pos_loss - old_loss[i_row]
    #             elif self.estimate_method == 'antithetic':
    #                 # neg perturbation
    #                 self._add_params_perturbation(self.sigma, -u)
    #                 self.model.zero_grad()
    #                 with torch.no_grad():
    #                     _, neg_loss = self.obj_fn(row=i_row)
    #                 self.forward_counter += 1
    #                 # remove perturbation
    #                 self._add_params_perturbation(self.sigma, u)

    #                 loss_diff = (pos_loss - neg_loss) / 2
                
    #             # loss_diff_seq[n] = loss_diff / self.sigma

    #             ZO_grad = ZO_grad + dimension * loss_diff * u / self.sigma / batch_sz / self.n_sample
    #             # ZO_grad = ZO_grad + math.sqrt(dimension) * loss_diff * u / self.sigma / batch_sz / self.n_sample
    #             # ZO_grad = ZO_grad + loss_diff * u / self.sigma / batch_sz / self.n_sample

    #     if verbose == False:
    #         return ZO_grad
    #     else:
    #         return ZO_grad, loss_diff_seq