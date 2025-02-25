from infer import OpenVLA_Infer
import numpy as np



path = "/media/users/will/openvla/31f090d05236101ebfc381b61c674dd4746d4ce0+ur_dataset+b8+lr-0.0005+lora-r32+dropout-0.0--50_chkpt"
openvla = OpenVLA_Infer(path)
obs = {'images': {'front': np.random.rand(480,640,3), 'left': np.random.rand(480,640,3), 'top': np.random.rand(480,640,3), 'wrist_left': np.random.rand(480,640,3)}}
task_name = 'pick_place_bread_ur'
action = openvla.infer(obs, task_name)
print(action)