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
            'val/top1': round(val_top1.avg.item(), 3),
            'val/loss': round(val_loss.avg.item(), 3),
        }

    def train_one_epoch(self, epoch):
        self.model.train()
        self.data_loader['train'].sampler.set_epoch(epoch)

        train_loss = DistributedMetric('train_loss')
        train_top1 = DistributedMetric('train_top1')
        
        self.optimizer.zero_grad()
        
        with tqdm(total=len(self.data_loader['train']),
                  desc='Train Epoch #{}'.format(epoch),
                  disable=dist.rank() > 0 or configs.ray_tune) as t:
            self.optimizer.zero_grad()
            for batch_idx, (images, labels) in enumerate(self.data_loader['train']):
                images, labels = images.cuda(), labels.cuda()
                # self.optimizer.zero_grad()
                
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
                    ##### partial BP #####
                    if configs.ZO_Estim.fc_bp == 'partial_BP':
                        output = self.model(images)
                        loss = self.criterion(output, labels)
                        # backward and update
                        loss.backward()

                        # partial update config
                        if configs.backward_config.enable_backward_config:
                            from core.utils.partial_backward import apply_backward_config
                            apply_backward_config(self.model, configs.backward_config)
                        
                        with torch.no_grad():
                            obj_fn = build_obj_fn(configs.ZO_Estim.obj_fn_type, data=images, target=labels, model=self.model, criterion=self.criterion)
                            self.ZO_Estim.update_obj_fn(obj_fn)
                            self.ZO_Estim.update_param_lr(self.optimizer.param_groups[0]['lr'])
                            self.ZO_Estim.estimate_grad(old_loss=loss)
                    
                    ##### NO BP #####
                    elif configs.ZO_Estim.fc_bp == 'cls_only':
                        obj_fn = build_obj_fn(configs.ZO_Estim.obj_fn_type, data=images, target=labels, model=self.model, criterion=self.criterion)

                        output, loss = obj_fn(detach_idx=-3)
                        loss.backward()
                        
                        with torch.no_grad():
                            self.ZO_Estim.update_obj_fn(obj_fn)
                            self.ZO_Estim.update_param_lr(self.optimizer.param_groups[0]['lr'])
                            self.ZO_Estim.estimate_grad(old_loss=loss)

                    elif configs.ZO_Estim.fc_bp == False:
                        with torch.no_grad():
                            output = self.model(images)
                            loss = self.criterion(output, labels)

                            obj_fn = build_obj_fn(configs.ZO_Estim.obj_fn_type, data=images, target=labels, model=self.model, criterion=self.criterion)
                            self.ZO_Estim.update_obj_fn(obj_fn)
                            self.ZO_Estim.update_param_lr(self.optimizer.param_groups[0]['lr'])
                            self.ZO_Estim.estimate_grad(old_loss=loss)
                
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
        
        return {
            'train/top1': round(train_top1.avg.item(), 3),
            'train/loss': round(train_loss.avg.item(), 3),
            'train/lr': round(self.optimizer.param_groups[0]['lr'], 5),
        }