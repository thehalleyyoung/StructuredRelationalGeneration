
import pickle
import numpy as np
import symmetries
from inspect import getmembers, isfunction
import itertools
import torch

# Define the folder where pickle files are stored
folder = "pickles"

# Number of measurements
n_meas = 16

# Size of the magenta vector
magenta_size = 256

# Function to get prototypes from adjacency matrix
def getPrototypes(adj_mat):
    # Sum the adjacency matrix along the third dimension
    adj_mat = np.sum(adj_mat, axis=2)
    xs = []
    # Loop over each measurement
    for i in range(n_meas):
        # Find the prototype for each measurement
        prot = (np.argmax(adj_mat[i, 16:]))
        prot_in_meas = np.argmax(adj_mat[:, prot])
        xs.append(prot_in_meas)
    y = np.array(xs)
    return y

# Load data from pickle files
meas = pickle.load(open(folder + "/meas16.pcl", "rb"))
simple_mats = pickle.load(open(folder + "/simple_mats" + str(n_meas) + ".pcl", "rb"))
inds = pickle.load(open(folder + "/inds" + str(n_meas) + ".pcl", "rb"))
meas = [meas[i] for i in inds]
adj_mats = pickle.load(open(folder + "/adj_mats" + str(n_meas) + ".pcl", "rb"))

# Ensure that the lengths of measurements and simple matrices are equal
assert(len(meas) == len(simple_mats))

# Get prototypes from adjacency matrices
prototypes = list(map(getPrototypes, adj_mats))

# Load all rewards vectors
all_rewards_vec = pickle.load(open(folder + "/all_rewards_vec" + str(n_meas) + ".pcl", "rb"))

# Load magenta vectors
magents = pickle.load(open(folder + "/analyzedmagents.pcl", "rb"))

# Process magenta vectors
new_magents = []
for graph_mag in magents:
    new_magents.append([])
    for bar in graph_mag:
        try:
            new_magents[-1].append(bar[0, :])
        except:
            new_magents[-1].append(bar)
magents = new_magents

# Initialize lists to store graph elements
all_nodes = []
all_edge_inds = []
all_edge_attrs = []

# Retrieve list of functions from the symmetries module
functions_list = [o for o in getmembers(symmetries) if isfunction(o[1])]
functions_name_list = [i[0] for i in functions_list if i[0] != "mod12Same"]

# Calculate the number of symmetries and total edge attributes
num_symmetries = len(functions_name_list)
num_total_edge_attrs = 2 * num_symmetries + 1 + 1 + 1

# Construct graph elements for each magenta vector
for z in range(len(magents)):
    edge_attrs = []
    edge_inds = []
    rewards_vec = all_rewards_vec[z]
    sorted_prototypes = sorted(list(set(prototypes[z])))

    # Initialize node features matrix
    x = np.zeros((len(set(prototypes[z])) + len(magents[z]), magenta_size))

    # Assign node features for prototypes and magenta vectors
    for h in range(len(set(prototypes[z]))):
        try:
            x[h] = magents[z][prototypes[z][h]]
        except:
            print("error")
            x[h] = magents[z][prototypes[z][0]]
        x[len(set(prototypes[z])) + h] = magents[z][h]

    # Create edges between consecutive magenta vectors
    for h in range(1, len(magents[z])):
        edge_inds.append([len(set(prototypes[z])) + h - 1, len(set(prototypes[z])) + h])
        edge_attr = np.zeros(num_total_edge_attrs)
        edge_attr[2 * num_symmetries + 1] = 1
        edge_attrs.append(edge_attr)

    # Create edges based on symmetries and rewards
    for h in range(1, 16):
        edge_attr = np.zeros(num_total_edge_attrs)
        try:
            for (k_ind, k) in enumerate(functions_name_list):
                if k in rewards_vec[h][prototypes[z][h]]:
                    edge_attr[k_ind] = 1
        except:
            print("error")
        edge_attrs.append(edge_attr)
        edge_inds.append([prototypes[z][h], len(set(prototypes[z])) + h])

    # Rewards vector and edges for neighbors
    rewards_vec = all_rewards_vec[z]
    for h in range(len(magents[z])):
        neighbors = [k for k in range(h - 3, h) if k >= 0 and k < len(magents[z])]
        for neighbor in neighbors:
            edge_inds.append([h + len(set(prototypes[z])), neighbor + len(set(prototypes[z]))])
            edge_attr = np.zeros(num_total_edge_attrs)
            try:
                for (k_ind, k) in enumerate(functions_name_list):
                    if k in rewards_vec[h][neighbor]:
                        edge_attr[num_symmetries + k_ind] = 1
            except:
                print("error")
            edge_attr[2 * num_symmetries + 2] = 1
            edge_attrs.append(edge_attr)

    # Add node features, edge indices, and edge attributes to the lists
    all_nodes.append(x)
    all_edge_inds.append(edge_inds)
    all_edge_attrs.append(edge_attrs)

# Save the graph elements to a pickle file
pickle.dump((all_nodes, all_edge_inds, all_edge_attrs), open("pickles/graphelements.pcl", "wb"))
