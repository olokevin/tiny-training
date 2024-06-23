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

class ClassificationTrainer(BaseTrainer):
    def validate(self):
        self.model.eval()
        val_criterion = self.criterion  # torch.nn.CrossEntropyLoss()

        val_loss = DistributedMetric('val_loss')
        val_top1 = DistributedMetric('val_top1')

        with torch.no_grad():
            with tqdm(total=len(self.data_loader['val']),
                      desc='Validate',
                      disable=dist.rank() > 0 or configs.ray_tune) as t:
                for images, labels in self.data_loader['val']:
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

        with tqdm(total=len(self.data_loader['train']),
                  desc='Train Epoch #{}'.format(epoch),
                  disable=dist.rank() > 0 or configs.ray_tune) as t:
            for batch_idx, (images, labels) in enumerate(self.data_loader['train']):
                images, labels = images.cuda(), labels.cuda()
                self.optimizer.zero_grad()
                
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
                        if OUT_GRAD_DEBUG:
                            splited_named_modules = split_named_model(self.model)
                            # for name, block in splited_named_modules.items():
                            #     print(name, block)

                            split_modules_list = split_model(self.model)
                            print(split_modules_list)

                            x = images
                            activations = []
                            for name, block in splited_named_modules.items():
                                if type(block) == QuantizedMbBlock:
                                    idx=0
                                    out = block.conv[idx](x)
                                    activations.append((out, name+'.conv.0', block.conv[0]))
                                    for conv_layer in block.conv[1:]:
                                        idx += 1
                                        out = conv_layer(out)
                                        activations.append((out, name+'.conv.'+str(idx), conv_layer))
                                    if block.q_add is not None:
                                        if block.residual_conv is not None:
                                            x = block.residual_conv(x)
                                        out = block.q_add(x, out)
                                        # No normalization for residual block  
                                        x = _TruncateActivationRange.apply(out, block.q_add.scale_y, block.q_add.zero_y, block.a_bit, None)
                                    else:
                                        x = out
                                else:
                                    x = block(x)

                                activations.append((x, name, block))
                            
                            for (activation, name, layer) in activations:
                                activation.register_hook(save_grad(layer))
                            
                            output = x
                        else:
                            output = self.model(images)
                        # confirmed. the output is the same
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
                    print('\nWeight Norm')
                    print('cos sim', F.cosine_similarity(FO_weight_grad.view(-1), ZO_weight_grad.view(-1), dim=0))
                    print('FO_weight_grad norm:', torch.linalg.norm(FO_weight_grad))
                    print('ZO_weight_grad norm:', torch.linalg.norm(ZO_weight_grad))

                    # print('cos sim FO/scale ZO', F.cosine_similarity((FO_weight_grad / this_layer.scale_w.view(-1,1,1,1)).view(-1), ZO_weight_grad.view(-1), dim=0))
                    # print('cos sim FO ZO/scale', F.cosine_similarity(FO_weight_grad.view(-1), (ZO_weight_grad / this_layer.scale_w.view(-1,1,1,1)).view(-1), dim=0))
                    # print('cos sim FO*scale ZO', F.cosine_similarity((FO_weight_grad * this_layer.scale_w.view(-1,1,1,1)).view(-1), ZO_weight_grad.view(-1), dim=0))
                    # print('cos sim FO ZO*scale', F.cosine_similarity(FO_weight_grad.view(-1), (ZO_weight_grad * this_layer.scale_w.view(-1,1,1,1)).view(-1), dim=0))
                    ratio = 0.1
                    topk_dim = int(FO_weight_grad.numel() * ratio)

                    _, FO_indices = torch.topk(FO_weight_grad.abs().flatten(), topk_dim)
                    _, ZO_indices = torch.topk(ZO_weight_grad.abs().flatten(), topk_dim)

                    topk_FO_grad = FO_weight_grad.view(-1)[FO_indices]
                    topk_ZO_grad = ZO_weight_grad.view(-1)[ZO_indices]

                    print(f'top {ratio} cos sim: {F.cosine_similarity(topk_FO_grad, topk_ZO_grad, dim=0)}')
                    print(f'top {ratio} FO norm: {torch.linalg.norm(topk_FO_grad)}')
                    print(f'top {ratio} ZO norm: {torch.linalg.norm(topk_ZO_grad)}')

                    print('\nBias Norm')
                    print('cos sim', F.cosine_similarity(FO_bias_grad.view(-1), ZO_bias_grad.view(-1), dim=0))
                    print('FO_bias_grad norm:', torch.linalg.norm(FO_bias_grad))

                    print('ZO_bias_grad norm:', torch.linalg.norm(ZO_bias_grad))

                if hasattr(self.optimizer, 'pre_step'):  # for SGDScale optimizer
                    self.optimizer.pre_step(self.model)

                self.optimizer.step()

                if hasattr(self.optimizer, 'post_step'):  # for SGDScaleInt optimizer
                    self.optimizer.post_step(self.model)

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
