## benchmark.py
This is a benchmark for the reward, loss, and episode length through RL algorithm simulations in the Car Racing Gym Environment.

Use: python benchmark.py [MODEL_NAME1] [MODEL_NAME2] ...

Current MODEL_NAME options: DQN, DDPG

Starting and ending simulation reward, loss, and episode length are outputted to the console.

Monitor Data is saved to ./logs/ALG_NAME_data.monitor.csv (Reward and Episode Length)

Logger Data is saved to ./logs/ALG_NAME/progress.csv (Actor/Critic Loss, etc.)

Plots saved to ./logs/ALG_NAME_*.png

## dqn_implementation.py
This is a modified DQN implementation for the Car Racing Gym Environment.
Timesteps to train the model is currently set to 10K, recommended is 500K (model.learn())

## ddpg_implementation.py
This is a modified DDPG implementation for the Car Racing Gym Environment
Timesteps to train the model is currently set to 10K, recommended is 500K (model.learn())

## example_plots/
This folder holds example plots generated for benchmarking DQN and DDPG for the Car Racing Gym Environment
for a short 10K timesteps. Recommended actual timesteps to train the model for is 500K.