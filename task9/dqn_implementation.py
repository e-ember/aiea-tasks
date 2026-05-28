# import libraries
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack

# create gym environment for DQN algorithm
def make_env():
    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)
    
    # resize the inputs to 84 x 84, and make the env grayscale, as per the Atari DQN paper
    env = ResizeObservation(env, shape=(84, 84))
    env = GrayscaleObservation(env, keep_dim=True)

    # this is to record reward and logs per episode
    env = gym.wrappers.RecordEpisodeStatistics(env)

    os.makedirs("./logs/", exist_ok=True)
    env = Monitor(env, "./logs/dqn_data")
    return env

# in order to follow the Atari DQN paper's procedure of stacking every four frames, we need to 
# wrap the environment and prepare it for DQN CNN. 
env = DummyVecEnv([make_env])
env = VecTransposeImage(env)       
env = VecFrameStack(env, n_stack=4)   

# create the DQN model, and create logs for training
model = DQN("CnnPolicy", env, verbose=1, buffer_size=10000, tensorboard_log="./dqn_logs/")
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
model.save("dqn_car_racing")

# Note: For few parts of the code, I worked with Claude to fully understand how to make the environment resizable, grayscale, and stack the frames. 
# Made sure to understand what each function was doing before I implemented it, specifically for ResizeObservation() GrayscaleObservation(), and VecFrameStack().