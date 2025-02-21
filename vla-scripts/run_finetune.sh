# export http_proxy=http://192.168.32.28:18000
# export https_proxy=http://192.168.32.28:18000

export HF_HOME=/media/users/will/huggingface_model
export HUGGINGFACE_HUB_CACHE=/media/users/will/huggingface_model

gpus=$1
data_root_dir=$2
run_root_dir=$3
dataset_name=$4
batch_size=$5
image_aug=$6
torchrun --standalone --nnodes 1 --nproc-per-node $gpus finetune_new.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir $data_root_dir \
  --dataset_name $dataset_name \
  --run_root_dir $run_root_dir \
  --adapter_tmp_dir /media/users/will/openvla \
  --lora_rank 32 \
  --batch_size $batch_size \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug $image_aug \
  --wandb_project openvla_ur \
  --wandb_entity will \
  --save_steps 5000