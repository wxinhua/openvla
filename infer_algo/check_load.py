from infer import OpenVLA_Infer
import numpy as np
from h5_loader import ReadH5Files
import cv2
import h5py
import os
import numpy as np
from collections import defaultdict
import torch
path = "/media/users/will/openvla/31f090d05236101ebfc381b61c674dd4746d4ce0+ur_dataset+b12+lr-0.0005+lora-r32+dropout-0.0--20000_chkpt"
openvla = OpenVLA_Infer(path)
Robot_Info_Dict = {}
Robot_Info_Dict['ur'] = {
    'camera_sensors': ['rgb_images'],
    'camera_names': ['camera_front','camera_left','camera_top','camera_wrist_left'],
    'arms': ['puppet','master'],
    # 'controls': ['joint_position', 'end_effector'],
    'controls': ['joint_position'],
    'use_robot_base': False
}

robot_infor = Robot_Info_Dict['ur']
read_h5files = ReadH5Files(robot_infor)
episode_path = "/media/cross_embodiment_data/ur/pick_place_bread_ur/success_episodes/val/0213_182221/data/trajectory.hdf5"
_, control_dict, _, _, _ = read_h5files.execute(episode_path, camera_frame=0, use_depth_image=False)
episode_qpos = control_dict['puppet']['joint_position'][:][:,:6]
episode_hand = control_dict['puppet']['joint_position'][:][:,6:]
episode_action = control_dict['puppet']['joint_position'][:]
episode_len = len(episode_qpos)
for index in range(1):
    image_dict, _, _, _, _ = read_h5files.execute(episode_path, camera_frame=index, use_depth_image=False)
    _, fake_front_image = cv2.imencode('.jpg', image_dict[robot_infor['camera_sensors'][0]]['camera_front'])
    _, fake_left_image = cv2.imencode('.jpg', image_dict[robot_infor['camera_sensors'][0]]['camera_left'])
    _, fake_top_image = cv2.imencode('.jpg', image_dict[robot_infor['camera_sensors'][0]]['camera_top'])
    _, fake_left_wrist_image = cv2.imencode('.jpg', image_dict[robot_infor['camera_sensors'][0]]['camera_wrist_left'])
    
    fake_obs = {
        'images': {
            'front': fake_front_image,
            'left': fake_left_image,
            'top': fake_top_image,
            'wrist_left': fake_left_wrist_image
        },
        # 'arm_joints': episode_qpos[index],
        # 'hand_joints': episode_hand[index]
    }

#obs = {'images': {'front': np.random.rand(480,640,3), 'left': np.random.rand(480,640,3), 'top': np.random.rand(480,640,3), 'wrist_left': np.random.rand(480,640,3)}}
task_name = 'pick_place_bread_ur'
ground_action = torch.from_numpy(episode_action[min(index+1,episode_len-1)]).unsqueeze(0)
action = openvla.infer(fake_obs, task_name)
print(action)
print(ground_action)