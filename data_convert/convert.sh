export CUDA_VISIBLE_DEVICES=0

python data_convert.py 
tfds build --data_dir=/media/users/will/rlds_ur/pick_place_bread_ur