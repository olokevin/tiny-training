# Poor Man's Training on MCUs: A Memory-Efficient Quantized Back-Propagation-Free Approach

We provide the code to simulate the BP-free on-device training on GPU servers.
## Setups

**Environment setup.** We recommend using Anaconda to set up the environment. Please find an example set up below:

```bash
conda create -n mcunetv3 python=3.8
conda activate mcunetv3
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install easydict
pip install timm
```

**Dataset preparation**. 

1. For CIFAR-10-C, please download [here](https://zenodo.org/records/2535967) and update the path under `data_provider.root`. default dataset path" `~/dataset`
2. For FGVC, we use `torchvision.datasets` to automatically download and prepare the datasets

## Usage
Zeroth-order training on CIFAR-10-C dataset: 

`cd algorithm/` 

`python train_cls.py configs/cifar_10c.yaml`

default: gaussian_noise corruption with severity 5

Zeroth-order training on FGVC dataset: 

`cd algorithm/` 

`python train_cls.py configs/fgvc.yaml`

