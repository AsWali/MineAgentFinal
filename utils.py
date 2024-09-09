import torch
from mineclip.mineagent.batch import Batch
from itertools import product
import numpy as np
from mineclip.mineagent.features.voxel.flattened_voxel_block import VOXEL_BLOCK_NAME_MAP

device = torch.device("mps")
block_type_to_index = {}
index_counter = 1

def get_block_index(block_type):
    global index_counter
    if block_type not in block_type_to_index:
        block_type_to_index[block_type] = index_counter
        index_counter += 1
    return block_type_to_index[block_type]

def preprocess_obs(env_obs, info, action):
    B = len(env_obs)
    vectorized_function = np.vectorize(get_block_index)
    yaw_ = [np.deg2rad(obs["location_stats"]["yaw"]) for obs in env_obs]
    pitch_ = [np.deg2rad(obs["location_stats"]["pitch"]) for obs in env_obs]
    compass = torch.as_tensor(np.hstack([np.sin(yaw_), np.cos(yaw_), np.sin(pitch_), np.cos(pitch_)]), device=device)
    gps = torch.as_tensor(np.vstack([obs["location_stats"]["pos"] for obs in env_obs]), device=device)
    voxels = torch.as_tensor(np.stack(vectorized_function([obs["voxels"]['block_name'] for obs in env_obs])), device=device).reshape((B, 3*3*3))
    rgb = torch.stack([entry['img_feat'] for entry in info]).to(device)
    prompt = torch.stack([entry['prompt_feat'] for entry in info]).to(device)
    if compass.shape != torch.Size((B, 4)):
        print("Compass shape is not correct")
    if gps.shape != torch.Size((B, 3)):
        print("GPS shape is not correct")
    if voxels.shape != torch.Size((B, 3 * 3 * 3)):
        print("Voxels shape is not correct")
    if action.shape != torch.Size((B,)):
        print("Action shape is not correct")
    if rgb.shape != torch.Size((B, 512)):
        print("RGB shape is not correct")
    if prompt.shape != torch.Size((B, 512)):
        print("Prompt shape is not correct")
    
    obs = {
        "rgb": rgb,
        "prompt": prompt,
        "compass": compass,
        "gps": gps,
        "voxels": voxels,
        "prev_action": action,
    }
    return Batch(obs=obs)

def transform_action(value):
    array = [0, 0, 0, 12, 12, 0, 0, 0]
    if value == 0:
        return array
    elif value == 1:
        array[5] = 1
    elif value == 2:
        array[5] = 3
    elif value == 3:
        array[0] = 1
    elif value == 4:
        array[0] = 1
        array[2] = 1
    elif value == 5:
        array[2] = 1
    elif value == 6:
        array[0] = 2
    elif value == 7:
        array[1] = 1
    elif value == 8:
        array[1] = 2
    else:
        combinations = [(x, y) for x, y in product(range(-4, 5), repeat=2) if (x, y) != (0, 0)]
        result_dict = {i+9: tuple(combination) for i, combination in enumerate(combinations)}
        pitch, yaw = result_dict[value]
        array[3] += pitch
        array[4] += yaw
    
    return array