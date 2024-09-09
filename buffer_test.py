import pickle
from tianshou.replay_buffer import ReplayBuffer
import pdb
import torch
from utils import preprocess_obs, transform_action
import numpy as np
buffer_size = 20
device = torch.device('cpu')

total_timesteps = 100
num_updates = total_timesteps // buffer_size
buffer = ReplayBuffer(size=buffer_size, ignore_obs_next=True)
num_envs = 1
old_obs = preprocess_obs(1)
next_done = np.zeros(num_envs)

for update in range(1, num_updates + 1):
  for step in range(0, buffer_size):
        with torch.no_grad():
          action, logprob, _, value = torch.tensor([76]), torch.tensor([0.0540]), torch.tensor([0.0540]), torch.tensor([[0.0540]])
        next_obs, reward, done, info = preprocess_obs(1), torch.tensor([1]), [True], False

        buffer.add(obs=old_obs, act=action, rew= reward, logp=logprob, done=next_done, value=value.flatten(), logprob=logprob)
        next_done = np.array(done)
        # numpy array and cast 
  pdb.set_trace()