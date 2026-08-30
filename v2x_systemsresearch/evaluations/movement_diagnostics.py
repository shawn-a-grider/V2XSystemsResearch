import gymnasium as gym

class FeatureTrackerWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.total_episodes = 0 
        self.event_count = {'red light violation': 0, 'hard braking events': 0}

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        self.hard_brake_episode = False
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info['hard braking']:
            self.hard_brake_episode = True
        if info['red light violation']:
            self.event_count['red light violation'] += 1


        if terminated or truncated:
            self.total_episodes += 1
            if self.hard_brake_episode:
                self.event_count['hard braking events'] += 1

        info['event_count'] = self.event_count
        info['total_episodes'] = self.total_episodes
        return obs, reward, terminated, truncated, info

