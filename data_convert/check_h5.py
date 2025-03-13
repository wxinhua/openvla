import h5py
import numpy as np
episode_path = '/media/users/wk/IL_research/datasets/202502/h5_data/franka_dual_fr3/dual_move_the_apple_from_pink_plate_to_white_plate/success_episodes/train/0217_110702/data/trajectory.hdf5'
with h5py.File(episode_path, 'r') as original_file:
    #single-arm ur
    # joint_position = original_file['puppet/joint_position'][:]
    # images_1 = original_file['observations/rgb_images/camera_front'][:]
    # images_2 = original_file['observations/rgb_images/camera_left'][:]
    # images_3 = original_file['observations/rgb_images/camera_top'][:]
    # images_4 = original_file['observations/rgb_images/camera_wrist_left'][:]
    
    # dual-arms franka
    arm_joint_position_left = original_file['puppet/arm_joint_position_left'][:]
    arm_joint_position_right = original_file['puppet/arm_joint_position_right'][:]
    print(f"arm_joint_position_left shape is:{arm_joint_position_left.shape}")
    joint_position = np.concatenate([arm_joint_position_left, arm_joint_position_right], axis=1)
    print(f"joint_position shape is:{joint_position.shape}")

    left_action = joint_position[1][:7] - joint_position[0][:7]
    print(f"left_action shape is:{left_action.shape}")
    left_action = np.append(left_action, joint_position[1][7]).astype('float32')
    print(f"left_action shape is:{left_action.shape}")
    right_action = joint_position[1][8:15] - joint_position[0][8:15]
    print(f"right_action shape is:{right_action.shape}")
    right_action = np.append(right_action, joint_position[1][15]).astype('float32')
    print(f"right_action shape is:{right_action.shape}")
    action = np.concatenate([left_action, right_action], axis=0)
    print(f"action shape is:{action.shape}")
