import random

class FakeEnvironment:
  def __init__(self, state_space_size, action_space_size, episode_length):
      self.state_space_size = state_space_size
      self.action_space_size = action_space_size
      self.episode_length = episode_length
      self.current_step = 0

  def reset(self):
      # Return a random initial state
      return self._random_state()

  def step(self, action):
      # Validate action
      assert 0 <= action < self.action_space_size, "Invalid action"

      # Return random next state, reward, and done flag
      next_state = self._random_state()
      reward = self._random_reward()
      done = False
      success = False
      if random.random() < 0.01:  # 10% chance of ending the episode
          done = True
          success = True
      
      self.current_step += 1
      if self.current_step >= self.episode_length:
          done = True
          self.current_step = 0
      return next_state, reward, done, success

  def _random_state(self):
      # Generate a random state
      return [random.random() for _ in range(self.state_space_size)]

  def _random_reward(self):
      # Generate a random reward
      return random.uniform(-1, 1)  # Example: reward in range [-1, 1]

  def _random_done(self):
      # Randomly decide if the episode is done
      return random.choice([True, False])