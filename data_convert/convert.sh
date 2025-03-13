export CUDA_VISIBLE_DEVICES=0

python data_convert.py 
tfds build --data_dir=/media/users/will/rlds_franka_panda_2/flip_the_cup_upright