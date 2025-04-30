import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from quantize.quantized_ops_diff import QuantizedMbBlockDiff

from .base_trainer import BaseTrainer
from ..utils.basic import DistributedMetric, accuracy
from ..utils.config import configs
from ..utils.logging import logger
from ..utils import dist

class PEPITATrainerCore:
    def __init__(
        self,
        model: nn.Module,
        sigma: float = 1.0,
        n_samples: int = 1,
        trainable_layers: Optional[List[str]] = None,
        device: str = 'cuda'
    ):
        self.model = model
        self.sigma = sigma
        self.n_samples = n_samples
        self.device = device
        self.trainable_layers = trainable_layers or []
        
        # Register forward hooks for all QuantizedMbBlockDiff layers
        self.handles = []
        self.activations = {}
        self.gradients = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, QuantizedMbBlockDiff):
                if not trainable_layers or name in trainable_layers:
                    # Register forward hook to store activations
                    handle = module.register_forward_hook(self._save_activation(name))
                    self.handles.append(handle)
                    
                    # Initialize gradients dictionary
                    self.gradients[name] = {
                        'weight': torch.zeros_like(module.conv[0].weight.data),
                        'bias': torch.zeros_like(module.conv[0].bias.data)
                    }
    
    def _save_activation(self, name: str):
        def hook(module, input, output):
            self.activations[name] = {
                'input': input[0].detach(),
                'output': output.detach()
            }
        return hook
    
    def _perturb_activation(self, activation: torch.Tensor, sigma: float) -> torch.Tensor:
        """Perturb activation with random noise."""
        noise = torch.randn_like(activation) * sigma
        return activation + noise
    
    def estimate_gradients(
        self,
        loss_fn: callable,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Estimate gradients using PEPITA method."""
        batch_size = inputs.size(0)
        
        # Forward pass to get original loss and activations
        with torch.no_grad():
            outputs = self.model(inputs)
            original_loss = loss_fn(outputs, targets)
        
        # Reset gradients
        for name in self.gradients:
            self.gradients[name]['weight'].zero_()
            self.gradients[name]['bias'].zero_()
        
        # For each trainable layer
        for name in self.activations:
            layer = dict(self.model.named_modules())[name]
            activation = self.activations[name]['output']
            
            # For each sample
            for _ in range(self.n_samples):
                # Perturb activation
                perturbed_activation = self._perturb_activation(activation, self.sigma)
                
                # Forward pass with perturbed activation
                with torch.no_grad():
                    # Replace the layer's output with perturbed activation
                    layer_output = perturbed_activation
                    
                    # Forward through remaining layers
                    current_module = layer
                    while True:
                        next_module = None
                        for child_name, child in current_module._modules.items():
                            if isinstance(child, QuantizedMbBlockDiff):
                                next_module = child
                                break
                        if next_module is None:
                            break
                        layer_output = next_module(layer_output)
                    
                    # Compute loss with perturbed activation
                    perturbed_loss = loss_fn(layer_output, targets)
                
                # Compute gradient estimate
                delta_loss = perturbed_loss - original_loss
                delta_activation = perturbed_activation - activation
                
                # Accumulate gradients
                self.gradients[name]['weight'] += (delta_loss / self.sigma) * delta_activation.mean(dim=0, keepdim=True)
                self.gradients[name]['bias'] += (delta_loss / self.sigma) * torch.ones_like(activation[0])
        
        # Average gradients over samples
        for name in self.gradients:
            self.gradients[name]['weight'] /= self.n_samples
            self.gradients[name]['bias'] /= self.n_samples
        
        return self.gradients
    
    def update_parameters(self, lr: float):
        """Update model parameters using estimated gradients."""
        for name, module in self.model.named_modules():
            if name in self.gradients:
                # Update weights
                module.conv[0].weight.data -= lr * self.gradients[name]['weight']
                module.conv[0].bias.data -= lr * self.gradients[name]['bias']
                
                # Clamp to quantization range
                w_bit = module.conv[0].w_bit
                module.conv[0].weight.data = module.conv[0].weight.data.round().clamp(
                    -2 ** (w_bit - 1), 2 ** (w_bit - 1) - 1
                )
                module.conv[0].bias.data = module.conv[0].bias.data.round().clamp(
                    -2 ** (4 * w_bit - 1), 2 ** (4 * w_bit - 1) - 1
                )
    
    def cleanup(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []

class PEPITATrainer(BaseTrainer):
    def __init__(self, model: nn.Module, data_loader, criterion, optimizer, lr_scheduler):
        super().__init__(model, data_loader, criterion, optimizer, lr_scheduler)
        
        # Initialize PEPITA trainer core
        self.pepita_trainer = PEPITATrainerCore(
            model=self.model,
            sigma=configs.pepita.sigma,
            n_samples=configs.pepita.n_samples,
            trainable_layers=configs.pepita.trainable_layer_list,
            device='cuda'
        )

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

    def train_one_epoch(self, epoch):
        self.model.train()
        self.data_loader['train'].sampler.set_epoch(epoch)

        train_loss = DistributedMetric('train_loss')
        train_top1 = DistributedMetric('train_top1')
        
        self.optimizer.zero_grad()

        with tqdm(total=len(self.data_loader['train']),
                 desc='Train Epoch #{}'.format(epoch),
                 disable=dist.rank() > 0 or configs.ray_tune) as t:
            for batch_idx, (images, labels) in enumerate(self.data_loader['train']):
                images, labels = images.cuda(), labels.cuda()
                
                # Estimate gradients using PEPITA
                gradients = self.pepita_trainer.estimate_gradients(self.criterion, images, labels)
                
                # Update parameters using PEPITA
                self.pepita_trainer.update_parameters(self.optimizer.param_groups[0]['lr'])
                
                # Forward pass to compute metrics
                with torch.no_grad():
                    output = self.model(images)
                    loss = self.criterion(output, labels)

                # Update metrics
                acc1 = accuracy(output, labels, topk=(1,))[0]
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

    def cleanup(self):
        """Cleanup hooks and resources"""
        self.pepita_trainer.cleanup()
        super().cleanup() 