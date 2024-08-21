
# ### Transfer
# CUDA_VISIBLE_DEVICES=2 python train_cls.py configs/transfer.yaml
# CUDA_VISIBLE_DEVICES=2 nohup python train_cls.py configs/transfer.yaml >/dev/null 2>&1 &

# ### Corruption
# CUDA_VISIBLE_DEVICES=2 python train_cls.py configs/corruption.yaml
# CUDA_VISIBLE_DEVICES=2 nohup python train_cls.py configs/corruption.yaml >/dev/null 2>&1 &

# ### VWW
# python train_cls.py configs/vww.yaml
# CUDA_VISIBLE_DEVICES=2 nohup python train_cls.py configs/vww.yaml >/dev/null 2>&1 &

# corruption_types=(gaussian_noise impulse_noise shot_noise fog frost snow defocus_blur elastic_transform brightness contrast defocus_blur)
# corruption_types=(gaussian_noise impulse_noise shot_noise fog frost)
corruption_types=(snow elastic_transform brightness contrast defocus_blur)

# Initialize a counter for the GPU index
gpu_index=0
gpu=3

# Iterate through the list of corruption types
for corruption in "${corruption_types[@]}"
do
  # Set the GPU to be used for the current experiment
  # gpu=$((gpu_index % 4))
  
  # Run the experiment with the specified GPU and corruption type
  # CUDA_VISIBLE_DEVICES=$gpu python train_cls.py configs/corruption.yaml --corruption_type $corruption
  CUDA_VISIBLE_DEVICES=$gpu nohup python train_cls.py configs/corruption.yaml --corruption_type $corruption >/dev/null 2>&1 &
  
  # Increment the GPU index for the next experiment
  # gpu_index=$((gpu_index + 1))
done