
import pickle
import random
import numpy as np
from inspect import getmembers, isfunction
import itertools
from z3 import *  # Z3 is a high-performance theorem prover
import pdfkit
import poetryrelations

# Set the number of lines in the poem
n_lines = 10

# Directory where pickled data will be stored
output_dir = "pickles"

# Retrieve all functions from poetryrelations module, excluding the 'word' function
functions_list = [o for o in getmembers(poetryrelations) if isfunction(o[1]) if o[0] != "word"]
# Extract the names of the functions for later use
functions_name_list = [o[0] for o in functions_list if o[0] != "word"]

# Define a function to format text with HTML color tags
def color_format(color):
    return """
      <p style="color: """ + color + """; font-family: 'Liberation Sans',sans-serif">
        {}
      </p>
    """

# Load poem sections and vectors from pickled files
poem_secs, poem_vecs = (
    list(pickle.load(open("pickles/poem_sec_vecs.pcl", "rb")).keys()), 
    list(pickle.load(open("pickles/poem_sec_vecs.pcl", "rb")).values())
)
# Load poem scores from pickled file
poem_scores = list(pickle.load(open("pickles/poem_sec_scores.pcl", "rb")).values())
# Sort poems by their scores in descending order
order = sorted(range(len(poem_scores)), key=lambda i: poem_scores[i], reverse=True)

# Initialize lists to store various outputs
prototype_ids = []
prototypes = []
adj_mats = []
simple_adjs = []
inds = []

# Set a timeout parameter for the Z3 optimizer (1 minute)
set_param(timeout=60*1000)

# Process the top 2500 poems based on their scores
for (z_ind, z) in enumerate(order[:2500]):
    
    # Define Z3 optimization variables and constraints
    tot_cost = Int("tot_cost")
    o = Optimize()

    poem_vec = poem_vecs[z]
    # Set diagonal values of poem vector matrix based on function names
    for i in range(n_lines):
        for (function_ind, function) in enumerate(functions_name_list):
            poem_vec[i,i,function_ind] = 1 if function != "endrhyme" else 5
    poem = poem_secs[z]

    # Define BitVecs for references and other related Z3 variables
    refs = [BitVec("ref" + str(i), 5) for i in range(n_lines)]
    refs_used = [Bool("refs_used" + str(i)) for i in range(n_lines)]
    cost_refs = [Int("costref" + str(i)) for i in range(n_lines)]
    
    # Add constraints to the optimizer
    for i in range(n_lines):
        o.add(Implies(refs_used[i], refs[i] == i))
    o.add(Sum([If(refs_used[i],1, 0) for i in range(n_lines)]) >= 1)
    o.add(Sum([If(refs_used[i],1, 0) for i in range(n_lines)]) <= 6)
    ijs = list(itertools.product(list(range(n_lines)), list(range(n_lines))))

    # More constraints
    for i in range(n_lines):
        o.add(BV2Int(refs[i]) < n_lines)
        o.add(BV2Int(refs[i]) >= 0)

        for j in range(n_lines):
            o.add(Implies(BV2Int(refs[i]) == j, refs_used[j]))
            o.add(Implies(BV2Int(refs[i]) == j, (cost_refs[i]) == np.sum(poem_vec[i][j])))
    
    # Objective function components
    b = Sum([If(And(refs_used[i], refs_used[j]), 10 + -3*np.sum(poem_vec[i][j]), 0)  for (i,j) in list(ijs) if i != j])
    c = Sum(cost_refs)
    q = Sum([If(refs[i] == refs[j], np.sum(poem_vec[i][j]) - 4, 0) for (i,j) in list(ijs) if i != j])
    
    try:
        # Maximize the objective function and check for solution
        h = o.maximize(q + c + b)
        o.check()
        o.upper(h)
        a = o.model()
        refs_used = [i for i in range(len(refs_used)) if a[refs_used[i]]]
    except:
        print("error")
        continue

    # Extract the reference indices from the model
    refs = [int(str(a[i])) for i in refs]

    # Initialize adjacency matrices for each feature and for simple yes/no edges
    mat = np.zeros((n_lines + len(refs_used), len(refs_used) + n_lines, poem_vec.shape[2] + 2), dtype=np.bool)
    simple_mat = np.zeros((n_lines, len(refs_used)), dtype=np.bool)
    
    # Fill in the adjacency matrices
    for i in range(3, n_lines):
        for j in range(1, min(i,4)):
            if j == 1:
                mat[i,i - j, :] = np.array(list(poem_vec[i, i - j, :]) + [1,0])
            else:
                mat[i,i - j, :] = np.array(list(poem_vec[i, i - j, :]) + [0,0])

    for i in range(len(poem)):
        j = refs[i]  # j is ref measure that i is similar to
        j_ind = refs_used.index(refs[i])  # index of reference measure
        simple_mat[i,j_ind] = 1
        vec = poem_vec[i][j]
        # Set the corresponding features in the adjacency matrix
        for k in range(len(vec)):
            if vec[k]:
                mat[i, j_ind + len(poem), k] = 1

    # Append the results to the respective lists
    prototype_ids.append((refs_used))
    prototypes.append([poem[q] for q in refs_used])
    adj_mats.append(mat)
    simple_adjs.append(simple_mat)
    inds.append(z)

    # Dump the results into pickled files
    pickle.dump(inds, open(output_dir + "/inds" + str(n_lines) +".pcl", "wb"))
    pickle.dump(prototypes, open(output_dir + "/prototypes" + str(n_lines) + ".pcl", "wb"))
    pickle.dump(prototype_ids, open(output_dir + "/prototype-ids" + str(n_lines) + ".pcl", "wb"))
    pickle.dump(adj_mats, open(output_dir + "/adj_mats" + str(n_lines) + ".pcl", "wb"))
    pickle.dump(simple_adjs, open(output_dir + "/simple_mats" + str(n_lines) + ".pcl", "wb"))

    # Define colors for formatting the poem lines
    colors = ["red", "green", "blue", "brown", "pink", "purple"]
    string = ""
    # Format each line of the poem with the corresponding color
    for (q, line) in enumerate(poem):
        string += color_format(colors[refs_used.index(refs[q])]).format(line)
