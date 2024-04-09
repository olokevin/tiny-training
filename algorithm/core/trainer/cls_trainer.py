from tqdm import tqdm
import torch
import torch.nn.functional as F

from .base_trainer import BaseTrainer
from ..utils.basic import DistributedMetric, accuracy
from ..utils.config import configs
from ..utils.logging import logger
from ..utils import dist

from core.ZO_Estim.ZO_Estim_entry import build_obj_fn

def split_model(model):
    modules = []
    for m in model.children():
        if isinstance(m, (torch.nn.Sequential,)):
            modules += split_model(m)
        else:
            modules.append(m)
    return modules

def split_named_model(model, parent_name=''):
    named_modules = {}
    for name, module in model.named_children():
    # for name, module in model.named_modules():    # Error: non-stop recursion
        if isinstance(module, torch.nn.Sequential):
            named_modules.update(split_named_model(module, parent_name + name + '.'))
        elif hasattr(module, 'conv') and isinstance(module.conv, torch.nn.Sequential):
            named_modules.update(split_named_model(module.conv, parent_name + name + '.conv.'))
        else:
            named_modules[parent_name + name] = module
    return named_modules

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
                  desc='Train Epoch #{}'.format(epoch + 1),
                  disable=dist.rank() > 0 or configs.ray_tune) as t:
            for _, (images, labels) in enumerate(self.data_loader['train']):
                images, labels = images.cuda(), labels.cuda()
                self.optimizer.zero_grad()

                # output_sizes = []
                # splited_models = split_model(self.model)
                # x = images
                # output_sizes.append(x.size())
                # for layer in splited_models:
                #     x = layer(x)
                #     output_sizes.append(x.size())
                #     # print the name and the size of the output
                #     print(f'Output size: {x.size()}')
                
                output_sizes = []
                splited_named_models = split_named_model(self.model)
                x = images
                # for name, layer in splited_named_models.items():
                #     x = layer(x)
                #     output_sizes.append(x.size())
                #     print(f'{name}, {x.size()}')
                
                split_modules_list = list(splited_named_models.items())
                for i, (name, layer) in enumerate(split_modules_list):
                    x = layer(x)
                    output_sizes.append(x.size())
                    print(f'{i}, {name}, {x.size()}')

                if self.ZO_Estim is not None and configs.ZO_Estim.fc_bp == False:
                    pass
                else:
                    output = self.model(images)
                    loss = self.criterion(output, labels)
                    # backward and update
                    loss.backward()
                    name_list = self.ZO_Estim.trainable_param_list[0].split('.')
                    FO_grad = self.model[int(name_list[0])][int(name_list[1])].conv[int(name_list[3])].weight.grad.data
                    FO_grad = getattr(self.model[int(name_list[0])][int(name_list[1])].conv[int(name_list[3])], name_list[4]).grad.data

                    # partial update config
                    if configs.backward_config.enable_backward_config:
                        from core.utils.partial_backward import apply_backward_config
                        apply_backward_config(self.model, configs.backward_config)
                
                if self.ZO_Estim is not None:
                    with torch.no_grad():
                        obj_fn = build_obj_fn(configs.ZO_Estim.obj_fn_type, data=images, target=labels, model=self.model, criterion=self.criterion)
                        self.ZO_Estim.update_obj_fn(obj_fn)
                        output, loss, grads = self.ZO_Estim.estimate_grad()

                        self.ZO_Estim.update_grad()
                        name_list = self.ZO_Estim.trainable_param_list[0].split('.')
                        ZO_grad = self.model[int(name_list[0])][int(name_list[1])].conv[int(name_list[3])].weight.grad.data

                name_list = self.ZO_Estim.trainable_param_list[0].split('.')
                w_scale = torch.tensor(self.model[int(name_list[0])][int(name_list[1])].conv[int(name_list[3])].w_scale).view(-1, 1, 1, 1).cuda()
                scale_FO_grad = FO_grad / w_scale
                scale_FO_grad_2 = FO_grad / w_scale ** 2
                scale_ZO_grad = ZO_grad / w_scale
                scale_ZO_grad_2 = ZO_grad / w_scale ** 2
                print('FO_grad norm:', torch.linalg.norm(FO_grad))
                print('scale_FO_grad norm:', torch.linalg.norm(scale_FO_grad))
                print('scale_FO_grad_2 norm:', torch.linalg.norm(scale_FO_grad_2))

                print('ZO_grad norm:', torch.linalg.norm(ZO_grad))
                print('scale_ZO_grad norm:', torch.linalg.norm(scale_ZO_grad))
                print('scale_ZO_grad_2 norm:', torch.linalg.norm(scale_ZO_grad_2))
                
                print('FO_grad-ZO_grad error norm:', torch.linalg.norm(FO_grad-ZO_grad))
                cos_sim_FO_ZO = F.cosine_similarity(FO_grad.view(-1), ZO_grad.view(-1), dim=0)
                print('cos sim', cos_sim_FO_ZO)
                        
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
                self.lr_scheduler.step()

                # train_info_dict = {
                #     'train/top1': train_top1.avg.item(),
                #     'train/loss': train_loss.avg.item(),
                #     'train/lr': self.optimizer.param_groups[0]['lr'],
                # }
                # logger.info(f'epoch {epoch}: f{train_info_dict}')
        
        return {
            'train/top1': train_top1.avg.item(),
            'train/loss': train_loss.avg.item(),
            'train/lr': self.optimizer.param_groups[0]['lr'],
        }
