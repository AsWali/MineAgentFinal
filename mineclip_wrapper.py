from gym import Wrapper
from mineclip import MineCLIP
from omegaconf import OmegaConf
import hashlib
import torch

class MineClipWrapper(Wrapper):
  def __init__(self, env, prompts, num_envs, device):
    super().__init__(env)
    self.env = env
    self.device = device
    self.num_envs = num_envs
    self.prompts = prompts
    
    # Initialize MineClip
    cfg = OmegaConf.load("/Users/asrorwali/Documents/MineAgent/configs/mineclip.yaml")

    OmegaConf.set_struct(cfg, False)
    ckpt = cfg.pop("ckpt")
    OmegaConf.set_struct(cfg, True)

    assert (
        hashlib.md5(open(ckpt.path, "rb").read()).hexdigest() == ckpt.checksum
    ), "broken ckpt"

    self.model = MineCLIP(**cfg).to(self.device)
    self.model.load_ckpt(ckpt.path, strict=True)
    self.model.eval()

    with torch.no_grad():
      self.prompt_feats = self.model.encode_text(prompts) # B, 512

  def step(self, action):
    next_state, reward, done, info = self.env.step(action)
    with torch.no_grad():
      img_feats = self.model.forward_image_features(torch.from_numpy(next_state['rgb']).to(self.device))

    info = tuple({**item, 'prompt_feat': self.prompt_feats[i]} for i, item in enumerate(info))
    info = tuple({**item, 'img_feat': img_feats[i]} for i, item in enumerate(info))
      
    return next_state, reward, done, info
  
  def reset(self):
    next_state = self.env.reset()
    with torch.no_grad():
      img_feats = self.model.forward_image_features(torch.from_numpy(next_state['rgb']).to(self.device))

    info = tuple({} for _ in range(self.num_envs))
    info = tuple({**item, 'prompt_feat': self.prompt_feats[i]} for i, item in enumerate(info))
    info = tuple({**item, 'img_feat': img_feats[i]} for i, item in enumerate(info))

    return next_state, info
  
  def forward_image_features(self, img):
    return self.model.forward_image_features(img)
  
  def is_successful(self):
    return self.env.is_successful()
