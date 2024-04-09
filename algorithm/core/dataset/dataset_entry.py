from .vision import *
from ..utils.config import configs
from .vision.transform import *
import torchvision
import pyvww

__all__ = ['build_dataset']


def build_dataset():
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
        dataset = ImageNet(root=configs.data_provider.root,
                       transforms=ImageTransform(), )
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
    elif configs.data_provider.dataset == 'imagehog':
        dataset = ImageHog(
            root=configs.data_provider.root,
            transforms=ImageTransform(),
        )
    else:
        raise NotImplementedError(configs.data_provider.dataset)

    return dataset
