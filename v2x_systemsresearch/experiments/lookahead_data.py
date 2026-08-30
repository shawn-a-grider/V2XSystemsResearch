from av_env import SignalDecisionEnv
from movement_diagnostics import FeatureTrackerWrapper
import numpy as np
from stable_baselines3 import PPO
import pandas as pd

def env_factory(lookahead):
    return FeatureTrackerWrapper(SignalDecisionEnv(lookahead))


def structure_data():
    lookaheads = np.array([0, 10, 30, 50, 80, 100], dtype = np.float32)
    red_light_violations = {}
    hard_braking = {}

    for lookahead in lookaheads:
        env = env_factory(lookahead)
        model = PPO('MlpPolicy', env, verbose = 0)
        model.learn(total_timesteps=100_000)
        red_light_violations[f'{lookahead / 10} sec lookahead'] = env.event_count['red light violation']
        hard_braking[f'{lookahead / 10} sec lookahead'] =  env.event_count['hard braking events']

    df = pd.DataFrame({'Red Light Violations': red_light_violations, 'Hard Braking Events': hard_braking})

    return df


df = structure_data()
print(df.head())