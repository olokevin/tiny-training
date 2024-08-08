from .vision import *
from ..utils.config import configs
from .vision.transform import *
import os
import torch
import torchvision
import pyvww

import numpy as np
from robustbench.data import load_cifar10c
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from .cub2011 import Cub2011

__all__ = ['build_dataset']

class CustomVisionDataset(torchvision.datasets.VisionDataset):
    def __init__(self, tensor_dataset, transform=None, target_transform=None):
        super(CustomVisionDataset, self).__init__(root=None, transform=transform, target_transform=target_transform)
        self.tensor_dataset = tensor_dataset

    def __len__(self):
        return len(self.tensor_dataset)

    def __getitem__(self, index):
        sample, target = self.tensor_dataset[index]
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, target


def build_dataset():
    configs.data_provider.root = os.path.expanduser(configs.data_provider.root)
    if configs.data_provider.dataset == 'image_folder':
        dataset = ImageFolder(
            root=configs.data_provider.root,
            transforms=ImageTransform(),
        )
    elif configs.data_provider.dataset == 'visualwakewords':
        dataset = {
            'train': pyvww.pytorch.VisualWakeWordsClassification(root="/home/yequan/dataset/vww_raw/coco_dataset/train2014", 
                    annFile="/home/yequan/dataset/vww_raw/coco_dataset/annotations/person_keypoints_train2014.json", transform=ImageTransform()['train'],),
            'val': pyvww.pytorch.VisualWakeWordsClassification(root="/home/yequan/dataset/vww_raw/coco_dataset/val2014", 
                    annFile="/home/yequan/dataset/vww_raw/coco_dataset/annotations/person_keypoints_val2014.json", transform=ImageTransform()['val'],),
        }
    elif configs.data_provider.dataset == 'imagenet':
        dataset = {
            'train': torchvision.datasets.ImageNet(configs.data_provider.root, split='train',
                                                  transform=ImageTransform()['train']),
            'val': torchvision.datasets.ImageNet(configs.data_provider.root, split='val',
                                                transform=ImageTransform()['val']),
        }
    elif configs.data_provider.dataset == 'cifar10':
        dataset = {
            'train': torchvision.datasets.CIFAR10(configs.data_provider.root, train=True,
                                                  transform=ImageTransform()['train'], download=True),
            'val': torchvision.datasets.CIFAR10(configs.data_provider.root, train=False,
                                                transform=ImageTransform()['val'], download=True),
        }
    elif configs.data_provider.dataset == 'cifar100':
        dataset = {
            'train': torchvision.datasets.CIFAR100(configs.data_provider.root, train=True,
                                                   transform=ImageTransform()['train'], download=True),
            'val': torchvision.datasets.CIFAR100(configs.data_provider.root, train=False,
                                                 transform=ImageTransform()['val'], download=True),
        }

    elif configs.data_provider.dataset == 'cars':
        dataset = {
            'train': torchvision.datasets.StanfordCars(configs.data_provider.root, split='train', transform=ImageTransform()['train'], download=False),
            'val': torchvision.datasets.StanfordCars(configs.data_provider.root, split='test', transform=ImageTransform()['val'], download=False),
        }
    elif configs.data_provider.dataset == 'cub':
        dataset = {
            'train': Cub2011(configs.data_provider.root, train=True, download=True, transform=ImageTransform()['train']),
            'val': Cub2011(configs.data_provider.root, train=False, download=True, transform=ImageTransform()['val']),
        }
    elif configs.data_provider.dataset == 'aircraft':
        dataset = {
            'train': torchvision.datasets.FGVCAircraft(configs.data_provider.root, split='train',
                                                  transform=ImageTransform()['train'], download=True),
            'val': torchvision.datasets.FGVCAircraft(configs.data_provider.root, split='val',
                                                transform=ImageTransform()['val'], download=True),
        }
    elif configs.data_provider.dataset == 'flowers':
        dataset = {
            'train': torchvision.datasets.Flowers102(configs.data_provider.root, split='train', transform=ImageTransform()['train'], download=True),
            'val': torchvision.datasets.Flowers102(configs.data_provider.root, split='test', transform=ImageTransform()['val'], download=True)
        }
    elif configs.data_provider.dataset == 'food':
        dataset = {
            'train': torchvision.datasets.Food101(configs.data_provider.root, split='train', transform=ImageTransform()['train'], download=True),
            'val': torchvision.datasets.Food101(configs.data_provider.root, split='test', transform=ImageTransform()['val'], download=True)
        }
    elif configs.data_provider.dataset == 'pets':
        dataset = {
            'train': torchvision.datasets.OxfordIIITPet(configs.data_provider.root, split='trainval', transform=ImageTransform()['train'], download=True),
            'val': torchvision.datasets.OxfordIIITPet(configs.data_provider.root, split='test', transform=ImageTransform()['val'], download=True)
        }
    elif configs.data_provider.dataset == 'cifar10-c':
        corruption_type = configs.data_provider.corruption_type
        severity = configs.data_provider.severity
        train_n = configs.data_provider.train_n

        x_corr, y_corr = load_cifar10c(
            10000, severity, os.path.expanduser(configs.data_provider.root), False, [corruption_type]
        )

        labels = {}
        num_classes = int(max(y_corr)) + 1
        for i in range(num_classes):
            labels[i] = [ind for ind, n in enumerate(y_corr) if n == i]
        num_ex = train_n // num_classes
        tr_idxs = []
        val_idxs = []
        test_idxs = []
        for i in range(len(labels.keys())):
            np.random.shuffle(labels[i])
            tr_idxs.append(labels[i][:num_ex])
            val_idxs.append(labels[i][num_ex:num_ex+10])
            # tr_idxs.append(labels[i][:num_ex+10])
            test_idxs.append(labels[i][num_ex+10:num_ex+100])
        tr_idxs = np.concatenate(tr_idxs)
        val_idxs = np.concatenate(val_idxs)
        test_idxs = np.concatenate(test_idxs)
        
        # train_data = TensorDataset(x_corr[tr_idxs], y_corr[tr_idxs])
        # val_data = TensorDataset(x_corr[val_idxs], y_corr[val_idxs])
        # test_data = TensorDataset(x_corr[test_idxs], y_corr[test_idxs])

        # Extract the list of transforms
        train_transforms = ImageTransform()['train']
        transform_list = train_transforms.transforms
        filtered_transforms = [t for t in transform_list if not isinstance(t, transforms.ToTensor)]
        train_transforms = transforms.Compose(filtered_transforms)

        val_transforms = ImageTransform()['val']
        transform_list = val_transforms.transforms
        filtered_transforms = [t for t in transform_list if not isinstance(t, transforms.ToTensor)]
        val_transforms = transforms.Compose(filtered_transforms)

        dataset = {
            'train': CustomVisionDataset(TensorDataset(x_corr[tr_idxs], y_corr[tr_idxs]), transform=train_transforms),
            'val': CustomVisionDataset(TensorDataset(x_corr[val_idxs], y_corr[val_idxs]), transform=val_transforms),
            'test': CustomVisionDataset(TensorDataset(x_corr[test_idxs], y_corr[test_idxs]), transform=val_transforms),
        }
      
    
    elif configs.data_provider.dataset == 'imagenet-c':
        corruption_type = configs.data_provider.corruption_type
        severity = configs.data_provider.severity
        train_n = configs.data_provider.train_n

        data_root = os.path.expanduser(configs.data_provider.root)
        image_dir = os.path.join(data_root, 'imagenet-c', corruption_type, str(severity))
        # dataset = ImageFolder(image_dir, transform=transforms.ToTensor())
        dataset = ImageFolder(image_dir)
        indices = list(range(len(dataset.imgs))) #50k examples --> 50 per class
        assert train_n <= 20000
        labels = {}
        y_corr = dataset.targets
        for i in range(max(y_corr)+1):
            labels[i] = [ind for ind, n in enumerate(y_corr) if n == i] 
        num_ex = train_n // (max(y_corr)+1)
        tr_idxs = []
        val_idxs = []
        test_idxs = []
        for i in range(len(labels.keys())):
            np.random.shuffle(labels[i])
            tr_idxs.append(labels[i][:num_ex])
            val_idxs.append(labels[i][num_ex:num_ex+10])
            # tr_idxs.append(labels[i][:num_ex+10])
            test_idxs.append(labels[i][num_ex+10:num_ex+20])
        tr_idxs = np.concatenate(tr_idxs)
        val_idxs = np.concatenate(val_idxs)
        test_idxs = np.concatenate(test_idxs)

        dataset = {
            'train': CustomVisionDataset(Subset(dataset, tr_idxs), transform=ImageTransform()['train']),
            'val': CustomVisionDataset(Subset(dataset, val_idxs), transform=ImageTransform()['val']),
            'test': CustomVisionDataset(Subset(dataset, test_idxs), transform=ImageTransform()['test']),
        } 
    
    else:
        raise NotImplementedError(configs.data_provider.dataset)
  
    if configs.data_provider.num_samples_per_class is not None:
        trainset = dataset['train']
        indices = []
        for i in range(configs.data_provider.num_classes):  
            class_indices = torch.where(torch.tensor(trainset._labels) == i)[0]
            subset_indices = class_indices[:configs.data_provider.num_samples_per_class]  
            indices.extend(subset_indices)

        dataset['train'] = Subset(trainset, indices)
        
        valset = dataset['val']
        indices = []
        for i in range(configs.data_provider.num_classes):  
            class_indices = torch.where(torch.tensor(valset._labels) == i)[0]
            subset_indices = class_indices[:configs.data_provider.num_samples_per_class]  
            indices.extend(subset_indices)

        dataset['val'] = Subset(valset, indices)
      
    return dataset
