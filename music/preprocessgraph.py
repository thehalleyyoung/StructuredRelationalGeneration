
import pickle
import random
import numpy as np
import symmetries
from inspect import getmembers, isfunction

# Load the Z3 solver library
from z3 import *

# Set the number of measures
n_meas = 16

# Load the list of songs from a pickle file
songs = pickle.load(open("pickles/meas" + str(n_meas) + ".pcl", "rb"))

# Initialize lists to store various types of rewards
all_rewards_tot = []  # List to store total rewards
all_rewards_vec = []  # List to store rewards as vectors
all_rewards_mat = []  # List to store rewards as matrices

# Initialize a dictionary to store the reward values for each symmetry function
rewards = {}
best_rewards = []  # List to store the best rewards

# Extract the list of functions from the symmetries module
functions_list = [o for o in getmembers(symmetries) if isfunction(o[1])]
functions_name_list = [o[0] for o in functions_list]
print(functions_name_list, len(functions_name_list))

# Define a list of important symmetry functions to be weighted differently
functions_name_list_important = [
    'hasSameRhythm',
    'hasAddOnePitchSymmetry',
    'hasIntervalSymmetry',
    'hasSameContour',
    'hasSamePitches',
    'hasAddOneRhythmSymmetry'
]

# Assign weights to each symmetry function based on their importance
for (name, func) in functions_list:
    if name in functions_name_list_important:
        rewards[name] = 2
    else:
        rewards[name] = 1

# Evaluate the reward for each song
for z in range(len(songs)):
    song = songs[z]

    # Skip songs with fewer measures than n_meas
    if len(song) < n_meas:
        continue
    else:
        song = song[:n_meas]

    # Initialize matrices to store rewards
    tot_rewards = [[0 for _ in range(len(song))] for _ in range(len(song))]
    rewards_names = [[[] for _ in range(len(song))] for _ in range(len(song))]
    rewards_mat = np.zeros((len(song), len(song), len(rewards)))

    # Calculate rewards for each pair of measures in the song
    for i in range(len(song)):
        for j in range(len(song)):
            if i == j:
                # Assign total possible rewards to diagonal elements (self-comparison)
                tot_rewards[i][j] += sum(rewards.values())
                rewards_names[i][j] = list(rewards.keys())
                rewards_mat[i][j] = np.ones(len(rewards))
            else:
                meas_i = song[i]
                meas_j = song[j]
                # Check each pair of measures for all types of symmetries and accumulate rewards
                for (name, func) in functions_list:
                    if name != "mod12Same":  # Exclude "mod12Same" from the evaluation
                        if func(meas_i, meas_j):
                            tot_rewards[i][j] += rewards[name]
                            rewards_names[i][j].append(name)
                            rewards_mat[i, j, functions_name_list.index(name)] = 1
        # Determine the best reward for each measure compared to others
        best_reward_j = max(tot_rewards[i][k] for k in range(len(song)) if k != i)
        best_rewards.append(best_reward_j)

    # Store the computed rewards for the current song
    all_rewards_vec.append(rewards_names)
    all_rewards_tot.append(tot_rewards)
    all_rewards_mat.append(rewards_mat)

# Save the computed rewards to pickle files for later use
pickle.dump(all_rewards_vec, open("pickles/all_rewards_vec" + str(n_meas) + ".pcl", "wb"))
pickle.dump(all_rewards_tot, open("pickles/all_rewards_tot" + str(n_meas) + ".pcl", "wb"))
pickle.dump(all_rewards_mat, open("pickles/all_rewards_mat" + str(n_meas) + ".pcl", "wb"))
