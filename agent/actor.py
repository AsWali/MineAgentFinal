import torch.nn as nn
from mineclip.utils import build_mlp

class Actor(nn.Module):
  def __init__(self, cfg, network_output_dim, device):
    super().__init__()

    self.policy = build_mlp(
      input_dim=network_output_dim,
      output_dim=cfg['output_dim'],
      hidden_dim=cfg['hidden_dim'],
      hidden_depth=cfg['hidden_depth'],
      activation="relu",
      norm_type=None,
    ).to(device)
  
  def forward(self, x):
    return self.policy(x)