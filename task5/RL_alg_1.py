# Based off of the Stable Baselines Getting Started documentation:
# https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html

import gymnasium as gym
from stable_baselines3 import A2C

env = gym.make("CarRacing-v2", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)

model = A2C("CnnPolicy", env, tensorboard_log="./rl_logs/", verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")

model.save("rl_car_racing")