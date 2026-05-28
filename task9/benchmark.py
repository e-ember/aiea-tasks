'''
Use: python benchmark.py [MODEL_NAME1] [MODEL_NAME2] ...
Current MODEL_NAME options: DQN, DDPG

End of simulation reward, loss, and episode length are outputted to the console.
Monitor Data is saved to ./logs/ALG_NAME_data.monitor.csv (Reward and Episode Length)
Logger Data is saved to ./logs/ALG_NAME/progress.csv (Actor/Critic Loss, etc.)
Plots saved to ./logs/ALG_NAME_*.png
'''

# import libraries
import sys, subprocess, os

# get command line arguments to run the python scripts for available RL algorithms
args = sys.argv
csv_paths = []
# if DQN in command line arguments, run its python script
if "DQN" in args:
    print("Running DQN implementation...")
    subprocess.run(["python", "dqn_implementation.py"])
    csv_paths.append("./logs/dqn_data.monitor.csv")
# if DDPG in command line arguments, run its python script
if "DDPG" in args:
    print("Running DDPG implementation...")
    subprocess.run(["python", "ddpg_implementation.py"])
    csv_paths.append("./logs/ddpg_data.monitor.csv")

'''
Template for future RL algorithms to add to the benchmark script.
if "ALG_NAME" in args:
    print("Running ALG_NAME implementation...")
    subprocess.run(["python", "ALG_NAME_implementation.py"])
    csv_paths.append("./logs/ALG_NAME_data.monitor.csv")

Changes that need to be added to future RL algorithms
Inside the make_env() function right before returning the environment: 
    os.makedirs("./logs/", exist_ok=True)
    env = Monitor(env, "./logs/ALG_NAME_data")

Between model initiation and model learning:
    log_path = "./logs/ALG_NAME/"
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

Timesteps for each algorithm in the model.learn() line of each python script can be changed for better training results
Recommended timesteps are 500,000
'''

# for each RL algorithm available to run
for path in csv_paths:
    import pandas as pd
    import matplotlib.pyplot as plt

    # read its csv file with the corresponding reward and episode length
    df = pd.read_csv(path, skiprows=1)

    # output final reward and episode length statistics
    name = path.replace("./logs/", "")
    name = path.replace("_data.monitor.csv", " summary").upper()
    print("\n", name)
    print("Starting Reward", df["r"].iloc[0])
    print("Ending Reward", df["r"].iloc[-1])
    print("Starting Episode Length", df["l"].iloc[0])
    print("Ending Episode Length", df["l"].iloc[-1])

    # plot reward and episode length over time and save graphs
    df.plot(x="t", y="r", kind="line")
    plt.title("Reward over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Reward")
    plt.savefig(path.replace("_data.monitor.csv", "_reward.png"))
    plt.clf()

    df.plot(x="t", y="l", kind="line")
    plt.title("Episode Length over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Episode Length")
    plt.savefig(path.replace("_data.monitor.csv", "_episode_length.png"))
    plt.clf()

    # read the monitor metrics csv file and output/save loss statistics
    df_loss = pd.read_csv(path.replace("_data.monitor.csv", "/progress.csv"))
    df_loss.columns = df_loss.columns.str.strip()
    if "train/actor_loss" in df_loss.columns and "train/critic_loss" in df_loss.columns:
        # output starting and ending actor loss
        print("Starting Actor Loss", df_loss["train/actor_loss"].iloc[0])
        print("Ending Actor Loss", df_loss["train/actor_loss"].iloc[-1])

        # create plot for actor loss over time and save graph
        df_loss.plot(x="time/total_timesteps", y="train/actor_loss", kind="line")
        plt.title("Actor Loss over Timesteps")
        plt.xlabel("Total Timesteps")
        plt.ylabel("Actor Loss")
        plt.savefig(path.replace("_data.monitor.csv", "_actor_loss.png"))
        plt.clf()

        # output starting and ending critic loss
        print("Starting Critic Loss", df_loss["train/critic_loss"].iloc[0])
        print("Ending Critic Loss", df_loss["train/critic_loss"].iloc[-1])

        # create plot for critic loss over time and save graph
        df_loss.plot(x="time/total_timesteps", y="train/critic_loss", kind="line")
        plt.title("Critic Loss over Timesteps")
        plt.xlabel("Total Timesteps")
        plt.ylabel("Critic Loss")
        plt.savefig(path.replace("_data.monitor.csv", "_critic_loss.png"))
    
    elif "train/loss" in df_loss.columns:
        # output starting and ending loss
        print("Starting Loss", df_loss["train/loss"].iloc[0])
        print("Ending Loss", df_loss["train/loss"].iloc[-1])

        # create plot for loss over time and save graph
        df_loss.plot(x="time/total_timesteps", y="train/loss", kind="line")
        plt.title("Loss over Timesteps")
        plt.xlabel("Total Timesteps")
        plt.ylabel("Loss")
        plt.savefig(path.replace("_data.monitor.csv", "_loss.png"))

'''
Example Console Output for Benchmarking DQN and DDPG
 ./LOGS/DQN SUMMARY
Starting Reward -41.176471
Ending Reward -50.647335
Starting Episode Length 1000
Ending Episode Length 636
Starting Loss 0.0044750981032848
Ending Loss 0.0016125340480357

 ./LOGS/DDPG SUMMARY
Starting Reward -76.821192
Ending Reward -83.870968
Starting Episode Length 1000
Ending Episode Length 1000
Starting Actor Loss -0.0146701429039239
Ending Actor Loss 0.2952332198619842
Starting Critic Loss 0.1668666303157806
Ending Critic Loss 0.010232669301331

Example Plots Generated can be found in example_plots/
'''