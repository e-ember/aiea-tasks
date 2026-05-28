# Read in command line arguments
import sys, subprocess
#if command line arguments include DQN, run DQN script
args = sys.argv
csv_paths = []
if "DQN" in args:
    import dqn_implementation
    subprocess.run(["python", "dqn_implementation.py"])
    csv_paths.append("./logs/dqn_data.monitor.csv")


if "DDPG" in args:
    import ddpg_implementation
    subprocess.run(["python", "ddpg_implementation.py"])
    csv_paths.append("./logs/ddpg_data.monitor.csv")

# save graphs to a folder called DQN_PLOTS
# output ending reward, loss, and episode len

for path in csv_paths:
    import pandas as pd
    import matplotlib.pyplot as plt

    print(f"Processing {path}...")

# if command line arguments include ddpg, run DDPG script
# save graphs to a folder called DDPG_PLOTS
# output ending reward, loss, and episode len