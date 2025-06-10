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

class FFTrainer(BaseTrainer):
    """
    Trainer implementing the Forward-Forward algorithm.
    Only layers named in `trainable_layer_list` are updated; others are frozen.
    Supports quantized conv2d modules with integrated quantized ReLU by dequantizing activations before computing goodness.

    Args:
        model (nn.Module): the network to train
        data_loader (dict): must contain 'train' DataLoader
        criterion (nn.BCEWithLogitsLoss): loss for positive vs. negative goodness
        optimizer (torch.optim.Optimizer): optimizer operating on trainable parameters
        lr_scheduler: learning-rate scheduler (optional)
        trainable_layer_list (list[str]): list of module names to train (e.g., ['conv1', 'layer2.0.conv2'])
        threshold (float): goodness threshold θ (default=0.0)
    """
    def __init__(
        self,
        model: nn.Module,
        data_loader,
        criterion: nn.BCEWithLogitsLoss,
        optimizer,
        lr_scheduler=None,
        trainable_layer_list=None,
        threshold: float = 0.0,
    ):
        super().__init__(model, data_loader, criterion, optimizer, lr_scheduler)
        self.threshold = threshold
        self._current_goodness = {name: [] for name in trainable_layer_list}

        # Store trainable layer names
        if trainable_layer_list is None:
            raise ValueError("trainable_layer_list must be provided as list of layer names")
        name_to_module = dict(model.named_modules())
        for name in trainable_layer_list:
            if name not in name_to_module:
                raise ValueError(f"Module name '{name}' not found in model")
        self.trainable_layers = trainable_layer_list

        # Freeze all parameters except those in trainable layers
        trainable_params = set()
        for name in self.trainable_layers:
            module = name_to_module[name]
            trainable_params.update(module.parameters())
        for p in self.model.parameters():
            p.requires_grad = (p in trainable_params)

        # Register hooks to capture per-layer activations' goodness
        self._hooks = []
        for name in self.trainable_layers:
            module = name_to_module[name]
            self._hooks.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, layer_name: str):
        def _hook(module, _inp, out):
            if layer_name not in self.trainable_layers:
                return
                
            batch_size = out.size(0)
            # dequantize activations if quantization params are present
            if hasattr(module, 'effective_scale'):
                # out is int, effective_scale maps to float
                scale = module.effective_scale.view(1, -1, *((1,) * (out.dim() - 2)))
                out_float = out.float() * scale
            elif hasattr(module, 'scale_y') and hasattr(module, 'zero_y'):
                zero = module.zero_y.view(1, -1, *((1,) * (out.dim() - 2)))
                scale = module.scale_y.view(1, -1, *((1,) * (out.dim() - 2)))
                out_float = (out.float() - zero) * scale
            else:
                out_float = out.float()
            g = out_float.view(batch_size, -1).pow(2).sum(dim=1)
            self._current_goodness[layer_name].append(g)
        return _hook

    def train_one_epoch(self, epoch: int):
        self.model.train()
        # set epoch for distributed sampler if available
        if hasattr(self.data_loader['train'].sampler, 'set_epoch'):
            self.data_loader['train'].sampler.set_epoch(epoch)

        train_loss = DistributedMetric('train_loss')
        train_top1 = DistributedMetric('train_top1')
        loader = self.data_loader['train']

        # Progress bar over training data
        with tqdm(total=len(loader), desc=f'Train Epoch #{epoch}', disable=dist.rank() > 0 or configs.ray_tune) as t:
            for batch_idx, (images, labels) in enumerate(loader):
                images, labels = images.cuda(), labels.cuda()

                x_neg = self.generate_negative_batch(images, labels)

                # Positive pass
                self._current_goodness = {name: [] for name in self.trainable_layers}
                outputs = self.model(images)
                pos_g = torch.stack([self._current_goodness[name][0] for name in self.trainable_layers], dim=1)

                # Negative pass
                self._current_goodness = {name: [] for name in self.trainable_layers}
                _ = self.model(x_neg)
                neg_g = torch.stack([self._current_goodness[name][0] for name in self.trainable_layers], dim=1)

                # Build logits and targets
                logits = torch.cat([pos_g - self.threshold, neg_g - self.threshold], dim=0)
                targets = torch.cat([
                    torch.ones_like(pos_g),
                    torch.zeros_like(neg_g)
                ], dim=0)

                loss = self.criterion(logits, targets)
                self.optimizer.zero_grad()
                loss.backward()
                
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

    def generate_negative_batch(self, x, labels=None):
        """
        Simple negative data: shuffles inputs across batch.
        Override for other generation schemes.
        """
        idx = torch.randperm(x.size(0), device=x.device)
        return x[idx]

    def cleanup(self):
        """Remove all registered hooks."""
        for h in getattr(self, '_hooks', []):
            h.remove()
        self._hooks = []
