import torch.nn as nn
from .actor import Actor
from .critic import Critic
from mineclip import SimpleFeatureFusion
from mineclip.mineagent import features as F
from mineclip.mineagent.actor.distribution import Categorical

class MineAgent(nn.Module):
    def __init__(self, cfg, device):
      super().__init__()
      feature_net_kwargs = cfg.feature_net_kwargs

      feature_net = {}
      for k, v in feature_net_kwargs.items():
          v = dict(v)
          cls = v.pop("cls")
          cls = getattr(F, cls)
          feature_net[k] = cls(**v, device=device)

      feature_fusion_kwargs = cfg.feature_fusion
      self.network = SimpleFeatureFusion(
          feature_net, **feature_fusion_kwargs, device=device
      )
      # Network returns return x, None

      self.actor = Actor(cfg.actor, self.network.output_dim, device)
      self.critic = Critic(cfg.critic, self.network.output_dim, device)

    def get_value(self, obs):
        hidden, _ = self.network(obs.obs)
        return self.critic(hidden)

    def get_action_and_value(self, obs, action=None):
        hidden, _ = self.network(obs.obs)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), logits

        
