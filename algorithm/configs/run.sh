
### Transfer
CUDA_VISIBLE_DEVICES=2 python train_cls.py configs/transfer.yaml
CUDA_VISIBLE_DEVICES=2 nohup python train_cls.py configs/transfer.yaml >/dev/null 2>&1 &

### Corruption
CUDA_VISIBLE_DEVICES=2 python train_cls.py configs/corruption.yaml
CUDA_VISIBLE_DEVICES=2 nohup python train_cls.py configs/corruption.yaml >/dev/null 2>&1 &

### VWW
python train_cls.py configs/vww.yaml
CUDA_VISIBLE_DEVICES=2 nohup python train_cls.py configs/vww.yaml >/dev/null 2>&1 &

# fp_weight=fp_model.blocks[13].mobile_inverted_conv.inverted_bottleneck.conv.weight.view(-1)
# int_weight=(int_model[1][13].conv[0].weight * int_model[1][13].conv[0].scale_w.view(-1,1,1,1)).view(-1)