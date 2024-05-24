from .vision import *
from ..utils.config import configs
from .vision.transform import *
import os
import torchvision
import pyvww

__all__ = ['build_dataset']


def build_dataset():
    configs.data_provider.root = os.path.expanduser(configs.data_provider.root)
    if configs.data_provider.dataset == 'image_folder':
        dataset = ImageFolder(
            root=configs.data_provider.root,
            transforms=ImageTransform(),
        )
    elif configs.data_provider.dataset == 'aircraft':
        dataset = {
            'train': torchvision.datasets.FGVCAircraft(configs.data_provider.root, split='train',
                                                  transform=ImageTransform()['train'], download=True),
            'val': torchvision.datasets.FGVCAircraft(configs.data_provider.root, split='val',
                                                transform=ImageTransform()['val'], download=True),
        }
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
    elif configs.data_provider.dataset == 'aircraft':
        dataset = {
            'train': torchvision.datasets.FGVCAircraft(configs.data_provider.root, split='train', annotation_level='family', transform=ImageTransform()['train'], download=True),
            'val': torchvision.datasets.FGVCAircraft(configs.data_provider.root, split='test', annotation_level='family', transform=ImageTransform()['val'], download=True),
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
    else:
        raise NotImplementedError(configs.data_provider.dataset)

    return dataset
