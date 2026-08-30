from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from rl_training.av_env import SignalDecisionEnv
from evaluations.movement_diagnostics import FeatureTrackerWrapper
import os

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

env = FeatureTrackerWrapper(SignalDecisionEnv(10.0))

model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)

logger = configure(
    'logs/',
    ['stdout','csv']
)

model.set_logger(logger)
model.learn(total_timesteps=100_000)
print(env.event_count)
print(env.total_episodes)
model.save("models/ppo_signal_agent_10")
