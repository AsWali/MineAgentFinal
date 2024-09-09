import numpy as np

class RunningBufferStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, value):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self):
        if self.n < 2:
            return 0.0
        return self.M2 / (self.n - 1)

    @property
    def stddev(self):
        return self.variance ** 0.5
    
class SelfImitationLearning:
  def __init__(self, buffer_size, si_threshold):
    self.buffer_size = buffer_size
    self.buffer = []
    self.episode = []
    self.gamma = 0.99
    self.running_stats = RunningBufferStats()
    self.si_threshold = si_threshold

  def add_episode_to_buffer(self, success):
    self.buffer.append((self.episode[0][2], success, self.episode))
    self.running_stats.update(self.episode[0][2])

    self.episode = []
    if len(self.buffer) > self.buffer_size:
      self.buffer.pop(0)

  def step(self, state, action, reward, done, success):
    breakpoint()
    self.episode.append((state, action, reward))
    if done:
      episodic_returns = self.discounted_rewards([r for _, _, r in self.episode], self.gamma)      
      self.episode = [(s, a, episodic_return) for (s, a, r), episodic_return in zip(self.episode, episodic_returns)]
      if success:
        self.add_episode_to_buffer(success)
      elif self.episode[0][2] > self.threshold():
        self.add_episode_to_buffer(success)
      else:
        self.episode = []

  def threshold(self):
    return self.running_stats.mean + self.si_threshold * self.running_stats.stddev
  
  def sample(self):
    probabilities = self.update_weights()
    idx = np.random.choice(len(self.buffer), p=probabilities)
    return self.buffer[idx]
  
  def update_weights(self, fraction_for_unsuccessful=0.1):
    returns = np.array([x[0] for x in self.buffer])
    successes = np.array([x[1] for x in self.buffer])

    probabilities = np.zeros(len(successes))
    num_successes = sum(successes)

    if num_successes > 0:
        equal_prob_success = (1 - fraction_for_unsuccessful) / num_successes
        probabilities[successes] = equal_prob_success

    sum_returns_unsuccessful = sum(returns) - sum(returns[successes])
    if sum_returns_unsuccessful > 0:
        probabilities[~successes] = fraction_for_unsuccessful * returns[~successes] / sum_returns_unsuccessful

    probabilities /= np.sum(probabilities)
    probabilities[-1] = 1.0 - np.sum(probabilities[:-1])
    return probabilities

  def buffered_items(self):
    return len(self.buffer)

  def discounted_rewards(self, rewards, gamma):
    discounted_rewards = []
    discounted_reward = 0
    for reward in reversed(rewards):
      discounted_reward = reward + gamma * discounted_reward
      discounted_rewards.insert(0, discounted_reward)
    return discounted_rewards