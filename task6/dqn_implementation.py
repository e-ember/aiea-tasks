import gymnasium as gym

from stable_baselines3 import DQN

env = gym.make("CarRacing-v2", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000, log_interval=4)

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()