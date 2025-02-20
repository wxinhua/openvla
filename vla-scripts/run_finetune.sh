# export http_proxy=http://192.168.32.28:18000
# export https_proxy=http://192.168.32.28:18000

export HF_HOME=/media/users/will/huggingface_model
export HUGGINGFACE_HUB_CACHE=/media/users/will/huggingface_model
torchrun --standalone --nnodes 1 --nproc-per-node 1 finetune_new.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /media/users/will/rlds_ur/ \
  --dataset_name ur_dataset \
  --run_root_dir /media/users/will/openvla \
  --adapter_tmp_dir /media/users/will/openvla \
  --lora_rank 32 \
  --batch_size 12 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug False \
  --wandb_project openvla_ur \
  --wandb_entity will \
  --save_steps 5000