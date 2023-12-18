
import pickle
import random
import numpy as np
import symmetries
from inspect import getmembers, isfunction
import itertools
from z3 import *

# Set up a signal handler for timeouts (not shown in the original code snippet)

rewards = {}  # Initialize an empty dictionary for rewards
vec_mean = []  # List to store the mean of vectors
all_vecs = []  # List to store all vectors
folder = "pickles"  # Directory where pickled files are stored

# Retrieve all functions from the symmetries module
functions_list = [o for o in getmembers(symmetries) if isfunction(o[1])]
functions_name_list = [o[0] for o in functions_list]  # List of function names
num_symmetries = len(functions_name_list)  # Number of symmetry functions

n_meas = 16  # Number of measures

# Load precomputed data from pickled files
all_rewards_vec = pickle.load(open(folder + "/all_rewards_vec" + str(n_meas) + ".pcl", "rb"))
songs = pickle.load(open(folder + "/meas" + str(n_meas) + ".pcl", "rb"))
all_tot_rewards = pickle.load(open(folder + "/all_rewards_tot" + str(n_meas) + ".pcl", "rb"))

# Initialize rewards for each function to 1
rewards = {i: 1 for i in functions_name_list}

# Set Z3 parameters
set_param(timeout=60*1000)  # Set timeout for Z3 solver
set_param("parallel.enable", True)  # Enable parallel mode

# Initialize lists for storing various information
adj_mats = []  # Adjacency matrices
simple_adjs = []  # Simple adjacency matrices (yes/no edge)
prototypes = []  # List of prototypes
prototype_ids = []  # List of prototype IDs
inds = []  # List of indices

# Process each song
for z in range(len(songs)):
    song = songs[z][:n_meas]  # Select measures for the current song
    tot_rewards = all_tot_rewards[z]  # Total rewards for the current song
    rewards_vec = all_rewards_vec[z]  # Rewards vector for the current song

    # Setup Z3 optimization
    tot_cost = Int("tot_cost")
    o = Optimize()  # Create an optimizer instance for maximum weighted set cover
    refs = [BitVec("ref" + str(i), 5) for i in range(len(song))]  # BitVecs for reference measures
    refs_used = [Bool("refs_used" + str(i)) for i in range(len(song))]  # Booleans indicating if a reference is used
    cost_refs = [Int("costref" + str(i)) for i in range(len(song))]  # Costs associated with references

    # Add constraints to the optimizer
    o.add(Sum([If(refs_used[i], 1, 0) for i in range(len(song))]) >= 3)
    o.add(Sum([If(refs_used[i], 1, 0) for i in range(len(song))]) <= min(6, len(song) // 2))
    ijs = itertools.product(range(len(song)), range(len(song)))
    for i in range(len(song)):
        o.add(BV2Int(refs[i]) < len(song))
        o.add(BV2Int(refs[i]) >= 0)
        for j in range(len(song)):
            o.add(Implies(BV2Int(refs[i]) == j, refs_used[j]))
            o.add(Implies(BV2Int(refs[i]) == j, (cost_refs[i]) == tot_rewards[i][j]))

    # Perform optimization
    try:
        h = o.maximize(Sum(cost_refs) - Sum([If(And(refs_used[i], refs_used[j]), tot_rewards[i][j], 0) for (i, j) in ijs if i != j]) + 5 * Sum([If(refs[i] == refs[j], 2 * (tot_rewards[i][j] - 8), -1 * tot_rewards[i][j] + 8) for (i, j) in ijs if i != j]))
        o.check()
        o.upper(h)
        a = o.model()
    except:
        print("timeout " + str(z))
        continue

    # Retrieve values from the model
    refs_used = [i for i in range(len(refs_used)) if a[refs_used[i]]]
    refs = [int(str(a[i])) for i in refs]

    prototypes_ = refs
    # Initialize adjacency matrices with features and simple yes/no edges
    mat = np.zeros((len(song) + len(refs_used), len(refs_used) + len(song), len(rewards) + 1), dtype=np.bool)
    simple_mat = np.zeros((len(song), len(refs_used)), dtype=np.bool)

    for i in range(1, len(song)):
        mat[i, i - 1, len(rewards)] = 1

    for i in range(len(song)):
        j = refs[i]  # Reference measure that i is similar to
        j_ind = refs_used.index(refs[i])  # Index of the reference measure
        simple_mat[i, j_ind] = 1
        vec = [functions_name_list.index(k) for k in rewards_vec[i][j]]
        vec_mean.append(len(vec))
        for val in vec:
            mat[i, j_ind + len(song), val] = 1

    # Store the results
    prototype_ids.append((refs_used))
    prototypes.append([song[q] for q in refs_used])
    adj_mats.append(mat)
    simple_adjs.append(simple_mat)
    inds.append(z)

    # Pickle intermediate results
    pickle.dump(inds, open(folder + "/inds" + str(n_meas) + ".pcl", "wb"))
    pickle.dump(prototypes, open(folder + "/prototypes" + str(n_meas) + ".pcl", "wb"))
    pickle.dump(prototype_ids, open(folder + "/prototype-ids" + str(n_meas) + ".pcl", "wb"))
    pickle.dump(adj_mats, open(folder + "/adj_mats" + str(n_meas) + ".pcl", "wb"))
    pickle.dump(simple_adjs, open(folder + "/simple_mats" + str(n_meas) + ".pcl", "wb"))

    # Process vectors for transformer model
    vecs = []
    sorted_prototypes = sorted(list(set(refs)))
    rewards_vec = all_rewards_vec[z]

    for h in range(3, 16):
        ref = prototypes_[h]
        shares_ref_prev = [i for i in range(h) if prototypes_[i] == ref and abs(i - h) <= 8]
        if len(shares_ref_prev) == 0:
            ref_ahead = 0
        else:
            ref_ahead = h - shares_ref_prev[-1]
        ref_ahead_np = np.zeros(9)
        ref_ahead_np[ref_ahead] = 1
        prev_syms = [np.zeros(num_symmetries) for i in range(3)]
        for i in range(3):
            neighbor = h - i - 1
            for (k_ind, k) in enumerate(functions_name_list):
                if k in rewards_vec[h][neighbor]:
                    prev_syms[i][k_ind] = 1
        try:
            a = prototypes_[h]
            ref_syms = adj_mats[-1][h, 16 + sorted_prototypes.index(a)]
        except Exception as e:
            print("error", e)
            continue
        vecs.append(np.concatenate([ref_syms, prev_syms[0], prev_syms[1], prev_syms[2], ref_ahead_np]))

    all_vecs.append(vecs)
    print(all_vecs)
    # Pickle the final vector representations for transformer model
    pickle.dump(all_vecs, open("pickles/transformervecs.pcl", "wb"))
