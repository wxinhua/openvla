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
    
}

class URDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Tensor(
                            shape=(224, 224, 12),
                            dtype=np.uint8,
                            #encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,),  # Updated shape to reflect combined state
                            dtype=np.float32,
                            doc='Combined robot state (left and right).',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action.',
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                })
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        # save_path = sys.argv[2]
        # save_path = '/media/users/will/rlds_ur/pick_place_bread_ur'
        # print(save_path)
        # match = re.search(r'rlds_ur/([^/]+)/', save_path)
        # if match is None:
        #     raise ValueError(f"Invalid save path format: {save_path}")
        # data_name = match.group(1)
        data_name = 'pick_place_bread_ur'
        # train_paths = glob.glob(f"/media/cross_embodiment_data/ur/test/trajectory.hdf5")
        # val_paths = glob.glob(f"/media/cross_embodiment_data/ur/test/trajectory.hdf5")
        train_paths = glob.glob(f"/media/cross_embodiment_data/ur/{data_name}/success_episodes/train/*/data/trajectory.hdf5")
        val_paths = glob.glob(f"/media/cross_embodiment_data/ur/{data_name}/success_episodes/val/*/data/trajectory.hdf5")
        return {
            'train': self._generate_examples(train_paths),
            'val': self._generate_examples(val_paths),
        }

    def _generate_examples(self, episode_paths) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def crop_image(image):

            img = cv2.imdecode(image, cv2.IMREAD_COLOR)#[:, :, ::-1]
            height, width, _ = img.shape
            square_size = min(width, height)

            center_x, center_y = width // 2, height // 2
            crop_x1 = center_x - square_size // 2
            crop_y1 = center_y - square_size // 2
            cropped_image = img[crop_y1:crop_y1 + square_size, crop_x1:crop_x1 + square_size]

            resized_image = cv2.resize(cropped_image, (224, 224))

            return resized_image
        
        def _parse_example(episode_path):
            task = re.search(r'ur/([^/]+)/', episode_path).group(1)

            with h5py.File(episode_path, 'r') as original_file:
                #print('1')
                joint_position = original_file['puppet/joint_position'][:]
                images_1 = original_file['observations/rgb_images/camera_front'][:]
                images_2 = original_file['observations/rgb_images/camera_left'][:]
                images_3 = original_file['observations/rgb_images/camera_top'][:]
                images_4 = original_file['observations/rgb_images/camera_wrist_left'][:]

            episode = []
            last_index = 0
            for i in range(1, len(images_1)):
                
                action = joint_position[i][:6] - joint_position[last_index][:6]
                action = np.append(action, joint_position[i][6]).astype('float32')
                
                
                if sum(abs(action)) < 0.06 and joint_position[i][6] - joint_position[last_index][6] < 0.1:
                    continue
                
                resized_image_1 = crop_image(images_1[last_index])
                resized_image_2 = crop_image(images_2[last_index])
                resized_image_3 = crop_image(images_3[last_index])
                resized_image_4 = crop_image(images_4[last_index])
                resized_image = np.concatenate([resized_image_1, resized_image_2, resized_image_3, resized_image_4], axis=2)
                

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