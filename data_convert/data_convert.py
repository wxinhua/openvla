from typing import Iterator, Tuple, Any
import glob
import numpy as np
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import h5py
import cv2
import re
import sys


TASK2INS = {
    'pick_place_bread_ur': 'pick up the bread and place it on the plate',
    'dual_move_the_apple_from_pink_plate_to_white_plate': 'move the apple from the pink plate to the white plate',
    'put_corn_in_the_oven': 'pick up the corn and plate it on the plate with left arm, open the oven door with right arm, put the corn in the oven, and close the oven door',
    'ur_put_steamed_bun_on_the_steamer_100': 'pick up the steamed bun and place it on the steamer',
    'flip_the_cup_upright': 'flip the cup upright',
}

class GenDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        # single-arm ur info
        # return self.dataset_info_from_configs(
        #     features=tfds.features.FeaturesDict({
        #         'steps': tfds.features.Dataset({
        #             'observation': tfds.features.FeaturesDict({
        #                 'image': tfds.features.Tensor(
        #                     shape=(224, 224, 12),
        #                     dtype=np.uint8,
        #                     #encoding_format='png',
        #                     doc='Main camera RGB observation.',
        #                 ),
        #                 'state': tfds.features.Tensor(
        #                     shape=(7,),  # Updated shape to reflect combined state
        #                     dtype=np.float32,
        #                     doc='Combined robot state (left and right).',
        #                 )
        #             }),
        #             'action': tfds.features.Tensor(
        #                 shape=(7,),
        #                 dtype=np.float32,
        #                 doc='Robot action.',
        #             ),
        #             'language_instruction': tfds.features.Text(
        #                 doc='Language Instruction.'
        #             ),
        #         })
        #     }))

        # single-arm franka info
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Tensor(
                            shape=(224, 224, 9),
                            dtype=np.uint8,
                            #encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,),  # Updated shape to reflect combined state
                            dtype=np.float32,
                            doc='Combined robot state (left and right).',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(8,),
                        dtype=np.float32,
                        doc='Robot action.',
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                })
            }))
    
        # dual-arms franka info
        # return self.dataset_info_from_configs(
        #     features=tfds.features.FeaturesDict({
        #         'steps': tfds.features.Dataset({
        #             'observation': tfds.features.FeaturesDict({
        #                 'image': tfds.features.Tensor(
        #                     shape=(224, 224, 12),
        #                     dtype=np.uint8,
        #                     #encoding_format='png',
        #                     doc='Main camera RGB observation.',
        #                 ),
        #                 'state': tfds.features.Tensor(
        #                     shape=(16,),  # Updated shape to reflect combined state
        #                     dtype=np.float32,
        #                     doc='Combined robot state (left and right).',
        #                 )
        #             }),
        #             'action': tfds.features.Tensor(
        #                 shape=(16,),
        #                 dtype=np.float32,
        #                 doc='Robot action.',
        #             ),
        #             'language_instruction': tfds.features.Text(
        #                 doc='Language Instruction.'
        #             ),
        #         })
        #     }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        # save_path = sys.argv[2]
        # save_path = '/media/users/will/rlds_ur/pick_place_bread_ur'
        # print(save_path)
        # match = re.search(r'rlds_ur/([^/]+)/', save_path)
        # if match is None:
        #     raise ValueError(f"Invalid save path format: {save_path}")
        # data_name = match.group(1)
        data_name = 'flip_the_cup_upright'
        train_paths = glob.glob(f"/media/users/wk/IL_research/datasets/202502/h5_data/franka_panda_2/{data_name}/success_episodes/train/*/data/trajectory.hdf5")
        val_paths = glob.glob(f"/media/users/wk/IL_research/datasets/202502/h5_data/franka_panda_2/{data_name}/success_episodes/val/*/data/trajectory.hdf5")
        # train_paths = glob.glob(f"/media/users/wk/IL_research/datasets/202502/h5_data/ur_std_station_1/{data_name}/success_episodes/train/*/data/trajectory.hdf5")
        # val_paths = glob.glob(f"/media/users/wk/IL_research/datasets/202502/h5_data/ur_std_station_1/{data_name}/success_episodes/val/*/data/trajectory.hdf5")
        # train_paths = glob.glob(f"/media/users/wk/IL_research/datasets/202502/h5_data/franka_dual_fr3/{data_name}/success_episodes/train/*/data/trajectory.hdf5")
        # val_paths = glob.glob(f"/media/users/wk/IL_research/datasets/202502/h5_data/franka_dual_fr3/{data_name}/success_episodes/val/*/data/trajectory.hdf5")
        return {
            'train': self._generate_examples(train_paths),
            'val': self._generate_examples(val_paths),
        }

    def _generate_examples(self, episode_paths) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def crop_image(image):

            img = cv2.imdecode(image, cv2.IMREAD_COLOR)[:, :, ::-1]
            height, width, _ = img.shape
            square_size = min(width, height)

            center_x, center_y = width // 2, height // 2
            crop_x1 = center_x - square_size // 2
            crop_y1 = center_y - square_size // 2
            cropped_image = img[crop_y1:crop_y1 + square_size, crop_x1:crop_x1 + square_size]

            resized_image = cv2.resize(cropped_image, (224, 224))

            return resized_image
        
        def _parse_example(episode_path):
            task = re.search(r'franka_panda_2/([^/]+)/', episode_path).group(1)

            with h5py.File(episode_path, 'r') as original_file:
                #######single-arm ur
                # joint_position = original_file['puppet/joint_position'][:]
                # hand_joint_position = original_file['puppet/hand_joint_position'][:]
                # joint_position = np.concatenate([joint_position, hand_joint_position], axis=1)
                # print(f"joint positon shape: {joint_position.shape}")
                # images_1 = original_file['observations/rgb_images/camera_front'][:]
                # images_2 = original_file['observations/rgb_images/camera_left'][:]
                # images_3 = original_file['observations/rgb_images/camera_top'][:]
                # images_4 = original_file['observations/rgb_images/camera_wrist_left'][:]
                
                ############ dual-arms franka
                # joint_position = original_file['puppet/arm_joint_position'][:]
                
                # images_1 = original_file['observations/rgb_images/camera_front'][:]
                # images_2 = original_file['observations/rgb_images/camera_left'][:]
                # images_3 = original_file['observations/rgb_images/camera_right'][:]
                # images_4 = original_file['observations/rgb_images/camera_top'][:]
                ############ single-arm franka
                joint_position = original_file['puppet/joint_position'][:]
                images_1 = original_file['observations/rgb_images/camera_left'][:]
                images_2 = original_file['observations/rgb_images/camera_right'][:]
                images_3 = original_file['observations/rgb_images/camera_top'][:]
            episode = []
            last_index = 0
            for i in range(1, len(images_1)):
                
                # action = joint_position[i][:6] - joint_position[last_index][:6]
                # action = np.append(action, joint_position[i][6]).astype('float32')

                action = joint_position[i][:7] - joint_position[last_index][:7]
                action = np.append(action, joint_position[i][7]).astype('float32')
                if sum(abs(action)) < 0.06 and joint_position[i][7] - joint_position[last_index][7] < 0.1:
                    continue

                # left_action = joint_position[i][:7] - joint_position[last_index][:7]
                # left_action = np.append(left_action, joint_position[i][7]).astype('float32')
                # right_action = joint_position[i][8:15] - joint_position[last_index][8:15]
                # right_action = np.append(right_action, joint_position[i][15]).astype('float32')
                # action = np.concatenate([left_action, right_action], axis=0)
                
                # if sum(abs(action)) < 0.06 and joint_position[i][6] - joint_position[last_index][6] < 0.1:
                #     continue
                # if sum(abs(left_action)) < 0.06 and joint_position[i][7] - joint_position[last_index][7] < 0.1 and sum(abs(right_action)) < 0.06 and joint_position[i][15] - joint_position[last_index][15] < 0.1:
                #     continue
                
                # resized_image_1 = crop_image(images_1[last_index])
                # resized_image_2 = crop_image(images_2[last_index])
                # resized_image_3 = crop_image(images_3[last_index])
                # resized_image_4 = crop_image(images_4[last_index])
                # resized_image = np.concatenate([resized_image_1, resized_image_2, resized_image_3, resized_image_4], axis=2)
                
                resized_image_1 = crop_image(images_1[last_index])
                resized_image_2 = crop_image(images_2[last_index])
                resized_image_3 = crop_image(images_3[last_index])
                #resized_image_4 = crop_image(images_4[last_index])
                resized_image = np.concatenate([resized_image_1, resized_image_2, resized_image_3], axis=2)

                state = joint_position[last_index].astype('float32')
                

                episode.append({
                    'observation': {
                        'image': resized_image,
                        'state': state,
                    },
                    'action': action,
                    'language_instruction': TASK2INS[task],
                })
                last_index = i
            sample = {
                'steps': episode,
            }

            return episode_path, sample

        for sample in episode_paths:
            yield _parse_example(sample)