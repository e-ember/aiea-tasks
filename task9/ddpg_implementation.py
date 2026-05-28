# import libraries
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import numpy as np

# create gym environment for DDPG algorithm
def make_env():
    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False)
    

    # resize the inputs to 64 x 64 as per the DDPG paper
    env = ResizeObservation(env, shape=(64, 64))

    # NEW CHANGE: Adding Grayscale observation to the environment, in order to simplify model inputs. This was impelmented in the DQN paper but not the DDPG paper.
    env = GrayscaleObservation(env, keep_dim=True)


    # this is to record reward and logs per episode
    env = gym.wrappers.RecordEpisodeStatistics(env)

    os.makedirs("./logs/", exist_ok=True)
    env = Monitor(env, "./logs/ddpg_data")
    return env

env = DummyVecEnv([make_env])
env = VecTransposeImage(env)       
# NEW CHANGE: Stack three frames on each other, similar to the DQN paper, to provide context during training.
env = VecFrameStack(env, n_stack=3)   

noise = OrnsteinUhlenbeckActionNoise(
    mean = np.zeros(3),
    sigma=0.2 * np.ones(3),
    theta=0.15               
)

# create the DDPG model, and create logs for training
model = DDPG("CnnPolicy", env, verbose=1, action_noise=noise, buffer_size=100000, batch_size=64, learning_rate=1e-4, gamma=0.99, tensorboard_log="./ddpg_logs/", tau=0.001)
# train the model for 500,000 timesteps and log every 4 episodes
model.learn(total_timesteps=500000, log_interval=4)

# This is for running the model in the simulation and seeing how it performs.
# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render("human")

# save the model
model.save("new_ddpg_car_racing")

# Note: For few parts of the code, I worked with Claude to fully understand how to make the environment resizable, grayscale, and stack the frames. 
# Made sure to understand what each function was doing before I implemented it, specifically for ResizeObservation() GrayscaleObservation(), and VecFrameStack().