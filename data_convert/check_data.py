import h5py
import numpy as np

def check_data(episode_path):
    with h5py.File(episode_path, 'r') as original_file:
        #######single-arm ur
        joint_position = original_file['puppet/joint_position'][:]
        hand_joint_position = original_file['puppet/hand_joint_position'][:]
        joint_position = np.concatenate([joint_position, hand_joint_position], axis=1)
        print(f"joint positon shape: {joint_position.shape}")

        images_1 = original_file['observations/rgb_images/camera_front'][:]
        last_index = 0
        for i in range(1, len(images_1)):  
            action = joint_position[i][:6] - joint_position[last_index][:6]
            action = np.append(action, joint_position[i][6]).astype('float32')
            if sum(abs(action)) < 0.06 and joint_position[i][6] - joint_position[last_index][6] < 0.1:
                continue
            print(f"step: {i}, action: {action}")
            last_index = i
    
    return None

episode_path = '/media/users/wk/IL_research/datasets/202502/h5_data/ur_std_station_1/ur_put_steamed_bun_on_the_steamer_100/success_episodes/train/0226_154157/data/trajectory.hdf5'
check_data(episode_path)
