# Read in command line arguments
import sys, subprocess, os
#if command line arguments include DQN, run DQN script
args = sys.argv
csv_paths = []
if "DQN" in args:
    print("Running DQN implementation...")
    subprocess.run(["python", "dqn_implementation.py"])
    csv_paths.append("./logs/dqn_data.monitor.csv")


if "DDPG" in args:
    print("Running DDPG implementation...")
    subprocess.run(["python", "ddpg_implementation.py"])
    csv_paths.append("./logs/ddpg_data.monitor.csv")

# save graphs to a folder called DQN_PLOTS
# output ending reward, loss, and episode len

for path in csv_paths:
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(path, skiprows=1)

    name = path.replace("./logs/", "")
    name = path.replace("_data.monitor.csv", " summary").upper()
    print("\n", name)
    print("Starting Reward", df["r"].iloc[0])
    print("Ending Reward", df["r"].iloc[-1])
    print("Starting Episode Length", df["l"].iloc[0])
    print("Ending Episode Length", df["l"].iloc[-1])

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

    df_loss = pd.read_csv(path.replace("_data.monitor.csv", "/progress.csv"))
    df_loss.columns = df_loss.columns.str.strip()
    if "train/actor_loss" in df_loss.columns and "train/critic_loss" in df_loss.columns:
        print("Starting Actor Loss", df_loss["train/actor_loss"].iloc[0])
        print("Ending Actor Loss", df_loss["train/actor_loss"].iloc[-1])

        df_loss.plot(x="time/total_timesteps", y="train/actor_loss", kind="line")
        plt.title("Actor Loss over Timesteps")
        plt.xlabel("Total Timesteps")
        plt.ylabel("Actor Loss")
        plt.savefig(path.replace("_data.monitor.csv", "_actor_loss.png"))
        plt.clf()

        print("Starting Critic Loss", df_loss["train/critic_loss"].iloc[0])
        print("Ending Critic Loss", df_loss["train/critic_loss"].iloc[-1])

        df_loss.plot(x="time/total_timesteps", y="train/critic_loss", kind="line")
        plt.title("Critic Loss over Timesteps")
        plt.xlabel("Total Timesteps")
        plt.ylabel("Critic Loss")
        plt.savefig(path.replace("_data.monitor.csv", "_critic_loss.png"))
    
    elif "train/loss" in df_loss.columns:
        print("Starting Loss", df_loss["train/loss"].iloc[0])
        print("Ending Loss", df_loss["train/loss"].iloc[-1])

        df_loss.plot(x="time/total_timesteps", y="train/loss", kind="line")
        plt.title("Loss over Timesteps")
        plt.xlabel("Total Timesteps")
        plt.ylabel("Loss")
        plt.savefig(path.replace("_data.monitor.csv", "_loss.png"))


# if command line arguments include ddpg, run DDPG script
# save graphs to a folder called DDPG_PLOTS
# output ending reward, loss, and episode len