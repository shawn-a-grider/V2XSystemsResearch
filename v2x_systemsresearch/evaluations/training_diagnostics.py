import pandas as pd
import plotly.express as px

class TrainingDiagnostics:
    def __init__(self, log_path):
        self.log_path = log_path
        self.df = pd.read_csv(self.log_path)[['time/total_timesteps', 'rollout/ep_rew_mean', 'train/approx_kl', 'train/entropy_loss', 'train/explained_variance']]
        self.learning_rate = 0.0003
    def plot_rew(self):
        self.fig1 = px.line(self.df, x = 'time/total_timesteps', y = 'rollout/ep_rew_mean', title = 'Reward over Timesteps', labels = {'time/total_timesteps': 'Timestep', 'rollout/ep_rew_mean': 'Mean Episodic Reward'})
        self.fig1.show()
        return self.fig1
    def plot_kl(self):
        self.fig2 = px.line(self.df, x = 'time/total_timesteps', y = 'train/approx_kl', title = 'Kullback-Leibler Divergence over Timesteps', labels = {'time/total_timesteps': 'Timestep', 'train/approx_kl': 'KL Divergence'})
        self.fig2.show()
        return self.fig2
    def plot_entropy_loss(self):
        self.fig3 = px.line(self.df, x = 'time/total_timesteps', y = 'train/entropy_loss', title = 'Entropy Loss over Timesteps', labels = {'time/total_timesteps': 'Timestep', 'train/entropy_loss': 'Entropy Loss'})
        self.fig3.show()
        return self.fig3
    def plot_explained_variance(self):
        self.fig4 = px.line(self.df, x = 'time/total_timesteps', y = 'train/explained_variance', title = 'Explained Variance over Timesteps', labels = {'time/total_timesteps': 'Timestep', 'train/explained_variance': 'Explained Variance'})
        self.fig4.show()
        return self.fig4

