import numpy as np
from openvla_utils import get_vla, get_processor, get_vla_action
import cv2
import torch
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
class OpenVLA_Infer():
    def __init__(self,checkpoint_path):
        #model_family: str = "openvla"                    # Model family
        self.pretrained_checkpoint=checkpoint_path     # Pretrained checkpoint path
        self.load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
        self.load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    def crop_image(self, image):
        #img = cv2.imdecode(image, cv2.IMREAD_COLOR)#[:, :, ::-1]
        img = image
        height, width, _ = img.shape
        square_size = min(width, height)

        center_x, center_y = width // 2, height // 2
        crop_x1 = center_x - square_size // 2
        crop_y1 = center_y - square_size // 2
        cropped_image = img[crop_y1:crop_y1 + square_size, crop_x1:crop_x1 + square_size]

        resized_image = cv2.resize(cropped_image, (224, 224))

        return resized_image

    def infer(self, obs, task_name):
        #prompt
        language_instruction = {'pick_place_bread_ur':'pick up bread and place it on the plate'}
        unnorm_key: str = task_name
        # images
        full_image = obs['images']['front']
        full_image = self.crop_image(full_image)
        for cam_name in ['left','top','wrist_left']:
            cam_img = obs['images'][cam_name]
            #print(f"cam img type is:{type(cam_img)}, shape is:{cam_img.shape}")
            cam_img_resize = self.crop_image(cam_img)
            #print(f"shape is:{cam_img_resize.shape}")
            full_image= np.concatenate([full_image, cam_img_resize], axis=2)
        print(f"full image shape is:{full_image.shape}")

        model = get_vla(self.pretrained_checkpoint, self.load_in_4bit, self.load_in_8bit)
        processor = get_processor(self.pretrained_checkpoint)
        print("ready")
        
        with torch.inference_mode():

            obs = {"full_image": full_image, "state": None}
            action = get_vla_action(
                    model, 
                    processor, 
                    self.pretrained_checkpoint, 
                    obs, 
                    language_instruction[task_name], 
                    unnorm_key, 
                )
            

        return action
        
