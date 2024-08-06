
### Transfer
CUDA_VISIBLE_DEVICES=2 python train_cls.py configs/transfer.yaml
CUDA_VISIBLE_DEVICES=2 nohup python train_cls.py configs/transfer.yaml >/dev/null 2>&1 &

### Corruption
CUDA_VISIBLE_DEVICES=1 python train_cls.py configs/corruption.yaml
CUDA_VISIBLE_DEVICES=2 nohup python train_cls.py configs/corruption.yaml >/dev/null 2>&1 &

### VWW
python train_cls.py configs/vww.yaml
CUDA_VISIBLE_DEVICES=2 nohup python train_cls.py configs/vww.yaml >/dev/null 2>&1 &