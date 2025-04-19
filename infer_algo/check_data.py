import h5py
import numpy as np
import os
from infer import OpenVLA_Infer

os.environ["TRANSFORMERS_OFFLINE"] = "1"

def check_data(episode_path, model, task_name):
    with h5py.File(episode_path, 'r') as original_file:
        #######single-arm ur
        joint_position = original_file['puppet/arm_joint_position'][:]
        hand_joint_position = original_file['puppet/hand_joint_position'][:]
        joint_position = np.concatenate([joint_position, hand_joint_position], axis=1)
        print(f"joint positon shape: {joint_position.shape}")

        # images_1 = original_file['observations/rgb_images/camera_front'][:]
        images_2 = original_file['observations/rgb_images/camera_left'][:]
        # images_3 = original_file['observations/rgb_images/camera_top'][:]
        # images_4 = original_file['observations/rgb_images/camera_wrist_left'][:]

        last_index = 49
        for i in range(50, len(images_2)):  
            action = joint_position[i][:6]# - joint_position[last_index][:6]
            #joint_position[i][6] = np.where(joint_position[i][6]>= 0.5, 0.98, 0.02)
            action = np.append(action, joint_position[i][6]).astype('float32')
            # if sum(abs(action)) < 0.06 and joint_position[i][6] - joint_position[last_index][6] < 0.1:
            #     continue
            obs = {
                'images': {
                    # 'front': images_1[i],
                    'left': images_2[i],
                    # 'top': images_3[i],
                    # 'wrist_left': images_4[i]
                },
                'state': joint_position[last_index].astype('float32')
            }

            pred_action = model.infer(obs, task_name)
            # 使用字符串格式化打印到小数点后两位
            action_str = ', '.join([f"{a:.2f}" for a in action])
            print(f"step: {i}, action: [{action_str}]")
            print(f"step: {i}, pred action: {pred_action}")
            l1_loss = np.mean(np.abs(pred_action - action))
            print(f"step: {i}, L1 loss: {l1_loss:.2f}")
            
            last_index = i
    
    return None

# 加载模型
ckpt_path = '/media_bd/users/will/openvla/ur_std_station_1/ur_put_corn_into_the_pot/pretrained_model+gen_dataset+b12+lr-0.0005+lora-r32+dropout-0.0--image_aug--30000_chkpt'
model = OpenVLA_Infer(checkpoint_path=ckpt_path)
task_name = 'ur_put_corn_into_the_pot'

# 指定 episode_path
episode_path = '/media_bd/users/will/vla_dataset/ur_std_station_1/ur_put_corn_into_the_pot/success_episodes/val/0307_142248/data/trajectory.hdf5'

# 调用 check_data 函数
check_data(episode_path, model, task_name)