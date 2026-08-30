import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env
from data.build_data import rl_event

class SignalDecisionEnv (gym.Env):
    def __init__(self, lookahead):
        super(). __init__()

        self.dt = 0.1
        self.min_acceleration = -26.0
        self.max_acceleration = 13.5
        self.min_velocity = 0.0
        self.max_velocity = 110.0
        self.stop_line = 0.0 
        self.data_path = "data/spat_sample500k.csv"
        self.cycles = rl_event(self.data_path)
        self.all_cycles = [cycle for cycles in self.cycles.values() for cycle in cycles]
        self.lookahead = lookahead * self.dt
        self.future_signal = 0.0

        self.action_space = spaces.Box(
            low = np.array([-1.0]),
            high = np.array([1.0]),
            dtype = np.float32
        )

        self.observation_space = spaces.Box(
            low = np.array([-500.0, 0.0, -26.0, 0.0, 0.0, 0.0]),
            high = np.array([np.inf, 110.0, 13.5, 120.0, 2.0, 2.0]),
            dtype = np.float32
        )

        self.A = np.array([
                    [1, self.dt, 0.5 * self.dt ** 2],
                    [0, 1, self.dt],
                    [0, 0, 1]
                ])

        self.B = np.array([
                    [self.dt ** 3 / 6],
                    [self.dt **2 / 2],
                    [self.dt]
                ], dtype = np.float32)

    def _get_obs(self):
        
        return np.array([
            self.position,
            self.velocity,
            self.acceleration,
            self.signal_age,
            self.signal,
            self.future_signal
        ], dtype = np.float32)


    def update_future_signal(self):
        if self.lookahead == 0:
            self.future_signal = self.signal
            return self.future_signal
        age = self.signal_age + self.lookahead
        if age > self.duration:
            if self.internal_index + 1 == self.max_internal_index:
                return self.future_signal
            self.future_signal = self.all_cycles[self.cycle_index][self.internal_index + 1][0] if self.internal_index != self.max_internal_index else self.future_signal

    def reset_cycle(self):
        self.cycle_index = self.np_random.integers(0, len(self.all_cycles))
        self.internal_index = 0
        self.max_internal_index = len(self.all_cycles[self.cycle_index]) - 1
        self.signal = self.all_cycles[self.cycle_index][self.internal_index][0]
        self.duration = self.all_cycles[self.cycle_index][self.internal_index][1]
        self.signal_age = 0.0
        self.update_future_signal()

    def reset_state(self):
        self.vehicle_state = np.array([
            [self.np_random.uniform(-500, -400)],
            [np.clip(self.np_random.normal(88, 5), self.min_velocity, self.max_velocity)],
            [self.np_random.uniform(-5, 3)]
        ], dtype = np.float32)

    def set_pva(self):
        self.position = self.vehicle_state[0, 0]
        self.velocity = self.vehicle_state[1, 0]
        self.acceleration = self.vehicle_state[2, 0]

    def reset_time(self):
        self.elapsed_time = 0.0 
        self.max_episode_time = 120.0

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)

        self.reset_state()
        self.set_pva()
        self.reset_cycle()
        self.reset_time()
        info = {
            'hard braking': False,
            'red light violation': False
        }
        obs = self._get_obs()

        return obs, info

    def step_jerk(self, *args): #### Actions might include turning movements later as well, so I decided to let step take variable action arguments

        action = args[0]
        scaled_action = (action[0] + 1.0) / 2.0
        min_accel_jerk = (self.min_acceleration - self.acceleration) / self.dt
        max_accel_jerk = (self.max_acceleration - self.acceleration) / self.dt
        min_vel_jerk = 2.0 * (self.min_velocity - self.velocity - self.acceleration * self.dt) / self.dt**2
        max_vel_jerk = 2.0 * (self.max_velocity - self.velocity - self.acceleration * self.dt) / self.dt**2

        min_jerk = max(min_accel_jerk, min_vel_jerk)
        max_jerk = min(max_accel_jerk, max_vel_jerk)

        jerk = min_jerk + scaled_action * (max_jerk - min_jerk)
        reward = -0.001 * jerk**2

        return jerk, reward

    def step_time(self):
        self.elapsed_time += self.dt
        truncated = self.elapsed_time >= self.max_episode_time

        self.signal_age += self.dt
        time_reward = -0.001 * self.dt
        if self.signal_age >= self.duration:
            if self.internal_index== self.max_internal_index:
                self.reset_cycle()
            else: 
                self.internal_index += 1
                self.signal = self.all_cycles[self.cycle_index][self.internal_index][0]
                self.duration = self.all_cycles[self.cycle_index][self.internal_index][1]
                self.signal_age = 0.0

        return truncated, time_reward

    def step_cross(self, previous_position):
        crossed_this_step = previous_position <= self.stop_line and self.position > self.stop_line
        red_light_violation = self.signal == 0 and crossed_this_step
        successful_crossing = self.position >= 10.0 and not red_light_violation

        cross_reward = 0 

        if red_light_violation:
            cross_reward = -100.0
        if successful_crossing:
            cross_reward = 50.0

        terminated = red_light_violation or successful_crossing 

        return terminated, cross_reward, red_light_violation, successful_crossing

    def step(self, action):
        
        jerk, reward = self.step_jerk(action)
        previous_position = self.position
        self.vehicle_state = self.A @ self.vehicle_state + self.B * jerk
        self.set_pva()
        terminated, cross_reward, red_light_violation, successful_crossing = self.step_cross(previous_position)
        truncated, time_reward = self.step_time()
        self.update_future_signal()
        self.hard_braking = self.acceleration < -13.0


        reward += cross_reward 
        reward += time_reward 
        reward -= (self.acceleration**2)
        reward -= 0.1*(jerk**2)

        info = {
            'red light violation': red_light_violation,
            'successful_crossing': successful_crossing,
            'hard braking': self.hard_braking
        }

        return (
            self._get_obs(),
            float(reward),
            bool(terminated),
            bool(truncated),
            info
        )


