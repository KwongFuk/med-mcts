# ppo/ppo_trainer.py
from stable_baselines3 import PPO

class PPOTrainer:
    def __init__(self, env):
        self.model = PPO("MlpPolicy", env, verbose=1)

    def train(self, timesteps):
        self.model.learn(total_timesteps=timesteps)
