from __future__ import annotations

import minedojo
from minedojo.sim.inventory import InventoryItem
import numpy as np
from gym import Wrapper
from collections import deque

class DenseMilkCowEnv(Wrapper):
    def __init__(
        self,
        step_penalty: float | int,
        nav_reward_scale: float | int,
        success_reward: float | int,
    ):
        max_spawn_range = 10
        distance_to_axis = int(max_spawn_range / np.sqrt(2))
        spawn_range_low = (-distance_to_axis, 1, -distance_to_axis)
        spawn_range_high = (distance_to_axis, 1, distance_to_axis)

        env = minedojo.make(
            "harvest",
            target_names=["milk_bucket"],
            target_quantities=1,
            reward_weights={
                "milk_bucket": success_reward,
            },
            initial_inventory=[
                InventoryItem(slot=0, name="bucket", variant=None, quantity=1)
            ],
            initial_mobs=["cow"],
            initial_mob_spawn_range_low=spawn_range_low,
            initial_mob_spawn_range_high=spawn_range_high,
            image_size=(160, 256),
            world_seed=123,
            specified_biome="sunflower_plains",
            fast_reset=True,
            use_voxel=True,
            use_lidar=True,
            lidar_rays=[
                    (np.pi * pitch / 180, np.pi * yaw / 180, 999)
                    for pitch in np.arange(-30, 30, 10)
                    for yaw in np.arange(-30, 30, 10)
            ],
        )
        super().__init__(env)

        # reset cmds, call before `env.reset()`
        self._reset_cmds = ["/kill @e[type=!player]", "/clear", "/kill @e[type=item]"]

        self._episode_len = 200
        self._elapsed_steps = 0
        self._first_reset = True

        self._entity = "cow"
        assert step_penalty >= 0, f"penalty must be non-negative"
        self._step_penalty = step_penalty
        self._nav_reward_scale = nav_reward_scale

        self._consecutive_distances = deque(maxlen=2)
        self._distance_min = np.inf

    def reset(self, **kwargs):
        self._elapsed_steps = 0

        if not self._first_reset:
            for cmd in self._reset_cmds:
                self.env.unwrapped.execute_cmd(cmd)
            self.unwrapped.set_time(6000)
            self.unwrapped.set_weather("clear")
        self._first_reset = False

        self._consecutive_distances.clear()
        self._distance_min = np.inf

        obs = super().reset(**kwargs)

        entity_in_sight, distance = self._find_distance_to_entity_if_in_sight(obs)
        if entity_in_sight:
            distance = self._distance_min = min(distance, self._distance_min)
            self._consecutive_distances.append(distance)
        else:
            self._consecutive_distances.append(0)

        return obs

    def step(self, action):
        obs, _reward, done, info = super().step(action)

        # nav reward
        entity_in_sight, distance = self._find_distance_to_entity_if_in_sight(obs)
        nav_reward = 0
        if entity_in_sight:
            distance = self._distance_min = min(distance, self._distance_min)
            self._consecutive_distances.append(distance)
            nav_reward = self._consecutive_distances[0] - self._consecutive_distances[1]
        nav_reward = max(0, nav_reward)
        nav_reward *= self._nav_reward_scale
        # reset distance min if attacking the entity because entity will run away

        # total reward
        reward =  nav_reward - self._step_penalty + _reward

        self._elapsed_steps += 1
        if self._elapsed_steps >= self._episode_len:
            done = True

        info['success'] = self.is_successful
        return obs, reward, done, info
    
    def _find_distance_to_entity_if_in_sight(self, obs):
        in_sight, min_distance = False, None
        entities, distances = (
            obs["rays"]["entity_name"],
            obs["rays"]["entity_distance"],
        )
        entity_idx = np.where(entities == self._entity)[0]
        if len(entity_idx) > 0:
            in_sight = True
            min_distance = np.min(distances[entity_idx])
        return in_sight, min_distance