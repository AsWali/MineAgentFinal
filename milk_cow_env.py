import minedojo
from gym import Wrapper

class MilkCowEnv(Wrapper):
    def __init__(
        self,
        success_reward=200,
    ):
        image_size = (160, 256)
        env = minedojo.make(
            "harvest_milk_with_empty_bucket_and_cow",
            image_size=image_size,
            world_seed=123,
            specified_biome="sunflower_plains",
        )
        super().__init__(env)
        self.success_reward = success_reward
        # reset cmds, call before `env.reset()`
        self._reset_cmds = ["/kill @e[type=!player]", "/clear", "/kill @e[type=item]"]

        self._episode_len = 200
        self._elapsed_steps = 0
        self._first_reset = True

    def reset(self, **kwargs):
        self._elapsed_steps = 0

        if not self._first_reset:
            for cmd in self._reset_cmds:
                self.env.unwrapped.execute_cmd(cmd)
            self.unwrapped.set_time(6000)
            self.unwrapped.set_weather("clear")
        self._first_reset = False
        
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._episode_len:
            done = True
        if self.is_successful:
            reward = reward * self.success_reward
        return obs, reward, done, info