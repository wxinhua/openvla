import numpy as np
from openvla_utils import get_vla, get_processor, get_vla_action

class GenerateConfig:
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: str = "/mnt/hpfs/baaiei/lyx/openvla/runs/franka_data_joint/openvla-7b+franka_data_joint+b16+lr-0.0005+lora-r32+dropout-0.0"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
    unnorm_key: str = "slide_close_drawer_1"

def infer() -> None:
    cfg = GenerateConfig()
    model = get_vla(cfg)
    processor = get_processor(cfg)
    print("ready")
    
    while True:
        # implement the following three lines
        img = np.random.randint(0, 255, (224, 224, 12), dtype=np.uint8)
        task_name = "task_name_to_be_filled"
        current_joint = np.random.rand(7) # 6×joints + 1×gripper

        obs = {"full_image": img, "state": None}
        action = get_vla_action(
                model, 
                processor, 
                cfg.pretrained_checkpoint, 
                obs, 
                task_name, 
                cfg.unnorm_key, 
            )
        next_joint = current_joint + action

        # implement the following line
        print(next_joint)
        input("Press any key to continue...")
        
if __name__ == '__main__':
    infer()