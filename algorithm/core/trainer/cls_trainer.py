from tqdm import tqdm
import math
import random
import torch
import torch.nn.functional as F

from .base_trainer import BaseTrainer
from ..utils.basic import DistributedMetric, accuracy
from ..utils.config import configs
from ..utils.logging import logger
from ..utils import dist

from core.ZO_Estim.ZO_Estim_entry import build_obj_fn, split_model, split_named_model
from quantize.quantized_ops_diff import QuantizedMbBlockDiff as QuantizedMbBlock
from quantize.quantized_ops_diff import QuantizedConv2dDiff as QuantizedConv2d
from quantize.quantized_ops_diff import _TruncateActivationRange

PARAM_GRAD_DEBUG = None
# PARAM_GRAD_DEBUG = True

OUT_GRAD_DEBUG = None
# OUT_GRAD_DEBUG = True

def save_grad(layer):
    def hook(grad):
        layer.out_grad = grad
    return hook
  
def fwd_hook_save_value(module, input, output):
    module.in_value = input[0].detach().clone()
    module.out_value = output.detach().clone()

def bwd_hook_save_grad(module, grad_input, grad_output):
    module.in_grad = grad_input[0].detach().clone()
    module.out_grad = grad_output[0].detach().clone()

class ClassificationTrainer(BaseTrainer):
    def validate(self, data_set='val'):
        self.model.eval()
        val_criterion = self.criterion  # torch.nn.CrossEntropyLoss()

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
            'val/top1': val_top1.avg.item(),
            'val/loss': val_loss.avg.item(),
        }

    def train_one_epoch(self, epoch):
        self.model.train()
        self.data_loader['train'].sampler.set_epoch(epoch)

        train_loss = DistributedMetric('train_loss')
        train_top1 = DistributedMetric('train_top1')
        
        self.optimizer.zero_grad()
        
        if OUT_GRAD_DEBUG:
            for block in self.model[1]:
                for layer in block.conv:
                    if hasattr(layer, 'weight'): 
                        layer.register_forward_hook(fwd_hook_save_value)
                        layer.register_backward_hook(bwd_hook_save_grad)

        ##### Layerwise gradient estimaiton alignment #####
        if PARAM_GRAD_DEBUG:
            images, labels = next(iter(self.data_loader['train']))
            images, labels = images.cuda(), labels.cuda()
            self.optimizer.zero_grad() # clear the previous gradients
            
            FO_grad_list = []
            ZO_grad_list = []
            
            ### BP gradient
            output = self.model(images)
            loss = self.criterion(output, labels)
            # backward and update
            loss.backward()

            # partial update config
            if configs.backward_config.enable_backward_config:
                from core.utils.partial_backward import apply_backward_config
                apply_backward_config(self.model, configs.backward_config)
            
            for layer in self.model.modules():
                if isinstance(layer, QuantizedConv2d):
                    layer.FO_grad = layer.weight.grad.data 
                    # layer.FO_grad = layer.bias.grad.data 
                    FO_grad_list.append(layer.FO_grad.view(-1))
            
            ### ZO gradient
            self.optimizer.zero_grad()
            with torch.no_grad():
                output = self.model(images)
                loss = self.criterion(output, labels)

                obj_fn = build_obj_fn(configs.ZO_Estim.obj_fn_type, data=images, target=labels, model=self.model, criterion=self.criterion)
                self.ZO_Estim.update_obj_fn(obj_fn)
                self.ZO_Estim.update_param_lr(self.optimizer.param_groups[0]['lr'])
                self.ZO_Estim.estimate_grad(old_loss=loss)

                self.ZO_Estim.update_grad()
            
            for layer in self.model.modules():
                if isinstance(layer, QuantizedConv2d):
                    layer.ZO_grad = layer.weight.grad.data
                    # layer.ZO_grad = layer.bias.grad.data
            
            ### ZO gradient independnet perturbation
            # self.optimizer.zero_grad()
            # with torch.no_grad():
            #     batch_sz = images.size(0)
            #     for i in range(batch_sz):
            #         self.optimizer.zero_grad()
            #         images_i = images[i].unsqueeze(0)
            #         labels_i = labels[i].unsqueeze(0)
            #         output = self.model(images_i)
            #         loss = self.criterion(output, labels_i)

            #         obj_fn = build_obj_fn(configs.ZO_Estim.obj_fn_type, data=images_i, target=labels_i, model=self.model, criterion=self.criterion)
            #         self.ZO_Estim.update_obj_fn(obj_fn)
            #         self.ZO_Estim.update_param_lr(self.optimizer.param_groups[0]['lr'])
            #         self.ZO_Estim.estimate_grad(old_loss=loss)

            #         self.ZO_Estim.update_grad()
                    
            #         for layer in self.model.modules():
            #             if isinstance(layer, QuantizedConv2d):
            #                 if not hasattr(layer, 'ZO_grad'):
            #                     layer.ZO_grad = torch.zeros_like(layer.weight.grad.data)
            #                 layer.ZO_grad += layer.weight.grad.data / batch_sz
            
            
            """
                Layer selection
            """
            # for name, layer in self.model.named_modules():
            #     if isinstance(layer, QuantizedConv2d):
            #         print(name)

            # print('||G||')
            # for layer in self.model.modules():
            #     if isinstance(layer, QuantizedConv2d):
            #         G_W_ratio = torch.norm(layer.weight.grad)  
            #         print(f'{G_W_ratio}') 
            
            # print('||G / s||')
            # for layer in self.model.modules():
            #     if isinstance(layer, QuantizedConv2d):
            #         G_W_ratio = torch.norm(layer.weight.grad / layer.scale_w.view(-1,1,1,1))   
            #         print(f'{G_W_ratio}')                        
            
            # print('||G|| / ||W||')
            # for layer in self.model.modules():
            #     if isinstance(layer, QuantizedConv2d):
            #         G_W_ratio = torch.norm(layer.weight.grad) / torch.norm(layer.weight)
            #         print(f'{G_W_ratio}')
            
            # print('||G / s|| / ||W * s||')
            # for layer in self.model.modules():
            #     if isinstance(layer, QuantizedConv2d):
            #         G_W_ratio = torch.norm(layer.weight.grad / layer.scale_w.view(-1,1,1,1)) / torch.norm(layer.weight * layer.scale_w.view(-1,1,1,1))
            #         print(f'{G_W_ratio}')
            
            # print('||G / s^2|| / ||W||')
            # for layer in self.model.modules():
            #     if isinstance(layer, QuantizedConv2d):
            #         G_W_ratio = torch.norm(layer.weight.grad / layer.scale_w.view(-1,1,1,1)**2) / torch.norm(layer.weight)
            #         print(f'{G_W_ratio}')
            
            # print('||G|| / ||W * s||')
            # for layer in self.model.modules():
            #     if isinstance(layer, QuantizedConv2d):
            #         G_W_ratio = torch.norm(layer.weight.grad) / torch.norm(layer.weight * layer.scale_w.view(-1,1,1,1))
            #         print(f'{G_W_ratio}')
            
            # for block in self.model[1]:
            #     for layer in block.conv:
            #         if hasattr(layer, 'weight'):  
                        
            #             G_W_ratio = torch.norm(layer.weight.grad / (layer.weight + 1e-6))
            #             G_W_ratio = torch.norm((layer.weight.grad / layer.scale_w.view(-1,1,1,1)) / ((layer.weight + 1e-6) * layer.scale_w.view(-1,1,1,1)))
                        
            #             print(f'{G_W_ratio}')
            
            print('layer cos sim')
            for layer in self.model.modules():
                if isinstance(layer, QuantizedConv2d):
                    ZO_grad_list.append(layer.ZO_grad.view(-1))
                    print(f'{F.cosine_similarity(layer.FO_grad.view(-1), layer.ZO_grad.view(-1), dim=0)}')
            
            print('layer ZO/FO norm ratio')
            for layer in self.model.modules():
                if isinstance(layer, QuantizedConv2d):
                    print(f'{torch.linalg.norm(layer.ZO_grad.view(-1)) / torch.linalg.norm(layer.FO_grad.view(-1))}')
            
            print('===== No gradient norm scaling =====')
            print('layer MSE')
            for layer in self.model.modules():
                if isinstance(layer, QuantizedConv2d):
                    print(f'{torch.linalg.norm(layer.ZO_grad.view(-1) - layer.FO_grad.view(-1)) ** 2}')
            
            print('===== Scale by √d =====')            
            print('layer MSE')
            for layer in self.model.modules():
                if isinstance(layer, QuantizedConv2d):
                    # scale = math.sqrt(self.ZO_Estim.n_sample / (self.ZO_Estim.n_sample + layer.weight.numel() - 1))
                    if self.ZO_Estim.perturb_method == 'activation':
                        dimension = layer.out_dimension
                    elif self.ZO_Estim.perturb_method == 'param':
                        dimension = layer.weight.numel()
                    elif self.ZO_Estim.perturb_method == 'all_activation':
                        dimension = self.ZO_Estim.ZO_dimension
                    elif self.ZO_Estim.perturb_method == 'all_param':
                        dimension = self.ZO_Estim.ZO_dimension
                    scale = math.sqrt((self.ZO_Estim.n_sample * configs.data_provider.base_batch_size) / (self.ZO_Estim.n_sample * configs.data_provider.base_batch_size + dimension - 1))
                    print(f'{torch.linalg.norm(layer.ZO_grad.view(-1) * scale - layer.FO_grad.view(-1)) ** 2}')
          
            print('===== Scale by d =====')
            print('layer MSE')
            for layer in self.model.modules():
                if isinstance(layer, QuantizedConv2d):
                    # scale = self.ZO_Estim.n_sample / (self.ZO_Estim.n_sample + layer.weight.numel() - 1)
                    if self.ZO_Estim.perturb_method == 'activation':
                        dimension = layer.out_dimension
                    elif self.ZO_Estim.perturb_method == 'param':
                        dimension = layer.weight.numel()
                    elif self.ZO_Estim.perturb_method == 'all_activation':
                        dimension = self.ZO_Estim.ZO_dimension
                    elif self.ZO_Estim.perturb_method == 'all_param':
                        dimension = self.ZO_Estim.ZO_dimension
                    scale = (self.ZO_Estim.n_sample * configs.data_provider.base_batch_size) / (self.ZO_Estim.n_sample * configs.data_provider.base_batch_size + dimension - 1)
                    print(f'{torch.linalg.norm(layer.ZO_grad.view(-1) * scale - layer.FO_grad.view(-1)) ** 2}')
            
            FO_grad_vec = torch.cat(FO_grad_list)
            ZO_grad_vec = torch.cat(ZO_grad_list)
            print(f'model wise cos sim {F.cosine_similarity(FO_grad_vec, ZO_grad_vec, dim=0)}')
            print(f'model wise ZO / FO norm {torch.linalg.norm(ZO_grad_vec) / torch.linalg.norm(FO_grad_vec)}')
            print(f'model wise MSE {torch.linalg.norm(ZO_grad_vec-FO_grad_vec) ** 2}')
                
                
        ##### Fisher #####
        if OUT_GRAD_DEBUG:
            for batch_idx, (images, labels) in enumerate(self.data_loader['train']):
                if batch_idx >= configs.run_config.grad_accumulation_steps:
                    break
                else:
                    images, labels = images.cuda(), labels.cuda()
                    self.optimizer.zero_grad() # clear the previous gradients
                    
                    if self.ZO_Estim is None:
                        output = self.model(images)
                        loss = self.criterion(output, labels)
                        # backward and update
                        loss.backward()

                        # partial update config
                        if configs.backward_config.enable_backward_config:
                            from core.utils.partial_backward import apply_backward_config
                            apply_backward_config(self.model, configs.backward_config)
                    else:
                        with torch.no_grad():
                            output = self.model(images)
                            loss = self.criterion(output, labels)

                            obj_fn = build_obj_fn(configs.ZO_Estim.obj_fn_type, data=images, target=labels, model=self.model, criterion=self.criterion)
                            self.ZO_Estim.update_obj_fn(obj_fn)
                            self.ZO_Estim.update_param_lr(self.optimizer.param_groups[0]['lr'])
                            self.ZO_Estim.estimate_grad(old_loss=loss)

                            self.ZO_Estim.update_grad()
                    
                for block in self.model[1]:
                    for layer in block.conv:
                        if hasattr(layer, 'weight'):  
                            ### Fisher Information
                            
                            batch_sz = layer.out_value.size(0)
                            out_channel = layer.out_value.size(1)
                            
                            if not hasattr(layer, 'fisher'):
                                layer.fisher = torch.zeros(out_channel, device=layer.out_value.device)

                            for c in range(out_channel):
                                for n in range(batch_sz):
                                    layer.fisher[c] += ((torch.sum((layer.out_value[n,c,:,:] - layer.zero_y) * layer.out_grad[n,c,:,:])) ** 2) / (2 * batch_sz)
                            
                            # layer_dim = sum(p.numel() for name, p in layer.named_parameters())
                            # layer.fisher += = (torch.sum(layer.out_value * layer.out_grad)) ** 2 / layer_dim
                            
                            print(f'{torch.sum(layer.fisher)}')
        
        with tqdm(total=len(self.data_loader['train']),
                  desc='Train Epoch #{}'.format(epoch),
                  disable=dist.rank() > 0 or configs.ray_tune) as t:
            self.optimizer.zero_grad()
            for batch_idx, (images, labels) in enumerate(self.data_loader['train']):
                images, labels = images.cuda(), labels.cuda()
                # self.optimizer.zero_grad()
                
                if self.ZO_Estim is None:
                    if configs.train_config.layerwise_update is None:
                        output = self.model(images)
                        loss = self.criterion(output, labels)
                        # backward and update
                        loss.backward()

                        # partial update config
                        if configs.backward_config.enable_backward_config:
                            from core.utils.partial_backward import apply_backward_config
                            apply_backward_config(self.model, configs.backward_config)
                    else:
                        ##### Layerwise update #####
                        for layer in self.model.modules():
                            if isinstance(layer, torch.nn.Conv2d):
                                layer.weight.requires_grad = False
                                layer.bias.requires_grad = False
                        
                        lr = self.optimizer.param_groups[0]['lr']

                        if configs.train_config.layerwise_update == 'all':
                            layer_name_list = configs.train_config.layerwise_update_layer_list
                        elif configs.train_config.layerwise_update == 'one':
                            layer_name_list = [random.choice(configs.train_config.layerwise_update_layer_list),]
                        
                        # layerwise update each conv layer
                        for layer_name in layer_name_list:
                            name_list = layer_name.split('.')
                            this_layer = self.model[int(name_list[0])][int(name_list[1])].conv[int(name_list[3])]
                            this_layer.weight.requires_grad = True
                            this_layer.bias.requires_grad = True

                            output = self.model(images)
                            loss = self.criterion(output, labels)
                            # backward and update
                            loss.backward()

                            this_layer.weight.data.sub_( (lr * this_layer.weight.grad.data / this_layer.scale_w.view(-1, 1, 1, 1) ** 2).round().clamp(- 2 ** (this_layer.w_bit - 1), 2 ** (this_layer.w_bit - 1) - 1) )
                            this_layer.bias.data.sub_( (lr * this_layer.bias.grad.data / (this_layer.scale_x * this_layer.scale_w) ** 2).round().clamp(- 2 ** (4*this_layer.w_bit - 1), 2 ** (4*this_layer.w_bit - 1) - 1) )

                            self.optimizer.zero_grad()

                            this_layer.weight.requires_grad = False
                            this_layer.bias.requires_grad = False
                        
                        # update classifier layer (other layers requires_grad = False)
                        output = self.model(images)
                        loss = self.criterion(output, labels)
                        # backward and update
                        loss.backward()       
 
                else:
                    ##### break BP #####
                    if configs.ZO_Estim.fc_bp == 'break_BP':
                        obj_fn = build_obj_fn(configs.ZO_Estim.obj_fn_type, data=images, target=labels, model=self.model, criterion=self.criterion)
                        self.ZO_Estim.update_obj_fn(obj_fn)
                        
                        with torch.no_grad():
                            output = self.model(images)
                            loss = self.criterion(output, labels)
                        
                        self.ZO_Estim.get_break_ZO_grad()
                    
                    ##### partial BP #####
                    elif configs.ZO_Estim.fc_bp == 'partial_BP':
                        output = self.model(images)
                        loss = self.criterion(output, labels)
                        # backward and update
                        loss.backward()

                        if PARAM_GRAD_DEBUG:
                            name_list = self.ZO_Estim.trainable_layer_list[0].split('.')
                            if 'conv' in name_list:
                                conv_idx = int(name_list[3])
                            else:
                                conv_idx = 2
                            this_layer = self.model[int(name_list[0])][int(name_list[1])].conv[conv_idx]
                            # this_layer = self.model[int(name_list[0])][int(name_list[1])].conv[conv_idx].normalization_layer
                            FO_weight_grad = this_layer.weight.grad.data
                            FO_bias_grad = this_layer.bias.grad.data
                            
                            if OUT_GRAD_DEBUG:
                                FO_out_grad = this_layer.out_grad

                        # partial update config
                        if configs.backward_config.enable_backward_config:
                            from core.utils.partial_backward import apply_backward_config
                            apply_backward_config(self.model, configs.backward_config)
                        
                        with torch.no_grad():
                            obj_fn = build_obj_fn(configs.ZO_Estim.obj_fn_type, data=images, target=labels, model=self.model, criterion=self.criterion)
                            self.ZO_Estim.update_obj_fn(obj_fn)
                            self.ZO_Estim.update_param_lr(self.optimizer.param_groups[0]['lr'])
                            self.ZO_Estim.estimate_grad(old_loss=loss)

                            self.ZO_Estim.update_grad()

                            if PARAM_GRAD_DEBUG:
                                ZO_weight_grad = this_layer.weight.grad.data
                                ZO_bias_grad = this_layer.bias.grad.data
                                
                                if OUT_GRAD_DEBUG:
                                    ZO_out_grad = this_layer.out_grad
                    
                    ##### NO BP #####
                    elif configs.ZO_Estim.fc_bp == 'cls_only':
                        obj_fn = build_obj_fn(configs.ZO_Estim.obj_fn_type, data=images, target=labels, model=self.model, criterion=self.criterion)

                        output, loss = obj_fn(detach_idx=-3)
                        loss.backward()
                        
                        with torch.no_grad():
                            self.ZO_Estim.update_obj_fn(obj_fn)
                            self.ZO_Estim.update_param_lr(self.optimizer.param_groups[0]['lr'])
                            self.ZO_Estim.estimate_grad(old_loss=loss)

                            self.ZO_Estim.update_grad()

                    elif configs.ZO_Estim.fc_bp == False:
                        with torch.no_grad():
                            output = self.model(images)
                            loss = self.criterion(output, labels)

                            obj_fn = build_obj_fn(configs.ZO_Estim.obj_fn_type, data=images, target=labels, model=self.model, criterion=self.criterion)
                            self.ZO_Estim.update_obj_fn(obj_fn)
                            self.ZO_Estim.update_param_lr(self.optimizer.param_groups[0]['lr'])
                            self.ZO_Estim.estimate_grad(old_loss=loss)

                            self.ZO_Estim.update_grad()
                
                if PARAM_GRAD_DEBUG:
                    """
                        Single Layer similarity
                    """
                    print('\nWeight Norm')
                    print('cos sim', F.cosine_similarity(FO_weight_grad.view(-1), ZO_weight_grad.view(-1), dim=0))
                    print('FO_weight_grad norm:', torch.linalg.norm(FO_weight_grad))
                    print('ZO_weight_grad norm:', torch.linalg.norm(ZO_weight_grad))
                    print(f'FO * COS / ZO: {torch.linalg.norm(FO_weight_grad) * F.cosine_similarity(FO_weight_grad.view(-1), ZO_weight_grad.view(-1), dim=0) / torch.linalg.norm(ZO_weight_grad)}')
                    
                    if hasattr(this_layer, 'weight_mask'):
                        ZO_weight_grad = ZO_weight_grad * this_layer.weight_mask
                        FO_weight_grad = FO_weight_grad * this_layer.weight_mask
                        print('masked cos sim', F.cosine_similarity(ZO_weight_grad.view(-1), FO_weight_grad.view(-1), dim=0))
                        print('masked FO_weight_grad norm:', torch.linalg.norm(FO_weight_grad))
                        print('masked ZO_weight_grad norm:', torch.linalg.norm(ZO_weight_grad))
                        print(f'masked FO * COS / ZO: {torch.linalg.norm(FO_weight_grad) * F.cosine_similarity(ZO_weight_grad.view(-1), FO_weight_grad.view(-1), dim=0) / torch.linalg.norm(ZO_weight_grad)}')

                    # print('cos sim FO/scale ZO', F.cosine_similarity((FO_weight_grad / this_layer.scale_w.view(-1,1,1,1)).view(-1), ZO_weight_grad.view(-1), dim=0))
                    # print('cos sim FO ZO/scale', F.cosine_similarity(FO_weight_grad.view(-1), (ZO_weight_grad / this_layer.scale_w.view(-1,1,1,1)).view(-1), dim=0))
                    # print('cos sim FO*scale ZO', F.cosine_similarity((FO_weight_grad * this_layer.scale_w.view(-1,1,1,1)).view(-1), ZO_weight_grad.view(-1), dim=0))
                    # print('cos sim FO ZO*scale', F.cosine_similarity(FO_weight_grad.view(-1), (ZO_weight_grad * this_layer.scale_w.view(-1,1,1,1)).view(-1), dim=0))
                    
                    # ratio = 0.1
                    # topk_dim = int(FO_weight_grad.numel() * ratio)

                    # _, topk_indices = torch.topk(ZO_weight_grad.abs().flatten(), topk_dim)
                    # rand_indices = torch.randperm(FO_weight_grad.numel())[:topk_dim]

                    # topk_FO_grad = torch.zeros_like(FO_weight_grad.view(-1))
                    # topk_FO_grad[topk_indices] = FO_weight_grad.view(-1)[topk_indices]
                    # topk_ZO_grad = torch.zeros_like(ZO_weight_grad.view(-1))
                    # topk_ZO_grad[topk_indices] = ZO_weight_grad.view(-1)[topk_indices]
                    # print(f'top {ratio} cos sim: {F.cosine_similarity(topk_FO_grad, topk_ZO_grad, dim=0)}')
                    
                    # rand_FO_grad = torch.zeros_like(FO_weight_grad.view(-1))
                    # rand_FO_grad[rand_indices] = FO_weight_grad.view(-1)[rand_indices]
                    # rand_ZO_grad = torch.zeros_like(ZO_weight_grad.view(-1))
                    # rand_ZO_grad[rand_indices] = ZO_weight_grad.view(-1)[rand_indices]
                    # print(f'rand {ratio} cos sim: {F.cosine_similarity(rand_FO_grad, rand_ZO_grad, dim=0)}')

                    print('\nBias Norm')
                    print('cos sim', F.cosine_similarity(FO_bias_grad.view(-1), ZO_bias_grad.view(-1), dim=0))
                    print('FO_bias_grad norm:', torch.linalg.norm(FO_bias_grad))

                    print('ZO_bias_grad norm:', torch.linalg.norm(ZO_bias_grad))
                    
                    if hasattr(this_layer, 'bias_mask'):
                        ZO_bias_grad = ZO_bias_grad * this_layer.bias_mask
                        FO_bias_grad = FO_bias_grad * this_layer.bias_mask
                        print('masked cos sim', F.cosine_similarity(ZO_bias_grad.view(-1), FO_bias_grad.view(-1), dim=0))
                        print('masked FO_bias_grad norm:', torch.linalg.norm(FO_bias_grad))
                        print('masked ZO_bias_grad norm:', torch.linalg.norm(ZO_bias_grad))
                        print(f'masked FO * COS / ZO: {torch.linalg.norm(FO_bias_grad) * F.cosine_similarity(ZO_bias_grad.view(-1), FO_bias_grad.view(-1), dim=0) / torch.linalg.norm(ZO_bias_grad)}')
                    
                    if OUT_GRAD_DEBUG:
                        print('\nOut Grad Norm')
                        print('cos sim', F.cosine_similarity(FO_out_grad.view(-1), ZO_out_grad.view(-1), dim=0))
                        print('FO_out_grad norm:', torch.linalg.norm(FO_out_grad))
                        print('ZO_out_grad norm:', torch.linalg.norm(ZO_out_grad))
                        print(f'FO * COS / ZO: {torch.linalg.norm(FO_out_grad) * F.cosine_similarity(FO_out_grad.view(-1), ZO_out_grad.view(-1), dim=0) / torch.linalg.norm(ZO_out_grad)}')
                        
                        if hasattr(this_layer, 'perturb_mask'):
                            FO_out_grad = FO_out_grad * this_layer.perturb_mask
                            ZO_out_grad = ZO_out_grad * this_layer.perturb_mask
                            print('masked cos sim', F.cosine_similarity(FO_out_grad.view(-1), ZO_out_grad.view(-1), dim=0))
                            print('masked FO_out_grad norm:', torch.linalg.norm(FO_out_grad))
                            print('masked ZO_out_grad norm:', torch.linalg.norm(ZO_out_grad))
                            print(f'masked FO * COS / ZO: {torch.linalg.norm(FO_out_grad) * F.cosine_similarity(FO_out_grad.view(-1), ZO_out_grad.view(-1), dim=0) / torch.linalg.norm(ZO_out_grad)}')


                
                # The gradients are computed for each mini-batch by calling loss.backward(). 
                # This adds the gradients to the existing values instead of replacing them.
                if configs.run_config.grad_accumulation_steps > 1 and (batch_idx + 1) % configs.run_config.grad_accumulation_steps != 0:
                    pass
                # do SGD step
                else:
                    if hasattr(self.optimizer, 'pre_step'):  # for SGDScale optimizer
                        self.optimizer.pre_step(self.model)

                    self.optimizer.step()

                    if hasattr(self.optimizer, 'post_step'):  # for SGDScaleInt optimizer
                        self.optimizer.post_step(self.model)
                    
                    self.optimizer.zero_grad()  # or self.net.zero_grad()

                # after one step
                train_loss.update(loss, images.shape[0])
                acc1 = accuracy(output, labels, topk=(1,))[0]
                train_top1.update(acc1.item(), images.shape[0])

                t.set_postfix({
                    'loss': train_loss.avg.item(),
                    'top1': train_top1.avg.item(),
                    'batch_size': images.shape[0],
                    'img_size': images.shape[2],
                    'lr': self.optimizer.param_groups[0]['lr'],
                })
                t.update()

                # after step (NOTICE that lr changes every step instead of epoch)
                if configs.run_config.iteration_decay == 1:
                    self.lr_scheduler.step()    

                # if self.ZO_Estim is not None:
                #     train_info_dict = {
                #         'train/top1': train_top1.avg.item(),
                #         'train/loss': train_loss.avg.item(),
                #         'train/lr': self.optimizer.param_groups[0]['lr'],
                #     }
                #     logger.info(f'epoch:{epoch} batch:{batch_idx}: f{train_info_dict}')
        
        return {
            'train/top1': train_top1.avg.item(),
            'train/loss': train_loss.avg.item(),
            'train/lr': self.optimizer.param_groups[0]['lr'],
        }