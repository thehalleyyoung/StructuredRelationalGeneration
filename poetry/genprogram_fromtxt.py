
import pickle
import random
import numpy as np
from inspect import getmembers, isfunction
import itertools
from z3 import *
import pdfkit
import poetryrelations

# Number of lines in each poem
n_lines = 10

# Folder where pickled data will be stored
folder = "pickles"

# Retrieve all functions from the poetryrelations module except 'word'
functions_list = [o for o in getmembers(poetryrelations) if isfunction(o[1]) if o[0] != "word"]
# Extract the names of the functions for later use
functions_name_list = [o[0] for o in functions_list if o[0] != "word"]

# Helper function to format poem lines with color in HTML
def color_format(color):
    return """
      <p style="color: """ + color + """; font-family: 'Liberation Sans',sans-serif">
        {}
      </p>
    """

# Load vectors representing poems from a pickle file
poem_secs, poem_vecs = (list(pickle.load(open(folder + "/poem_vecs.pcl", "rb")).keys()), list(pickle.load(open(folder + "/poem_vecs.pcl", "rb")).values()))
print(len(poem_secs))
# Total number of poems
num_poems = len(poem_vecs)

# Initialize lists to store various poem attributes
prototype_ids = []
prototypes = []
adj_mats = []
simple_adjs = []
inds = []

# Set a timeout parameter for the Z3 solver in milliseconds
set_param(timeout=60*1000)

# Main loop to process each poem
for z in range(num_poems):
    print(str(z) + " of " + str(num_poems - 1))
    
    # Define the total cost as an integer variable
    tot_cost = Int("tot_cost")
    # Initialize the Z3 optimizer
    o = Optimize()

    # Retrieve the vector for the current poem
    poem_vec = poem_vecs[z]
    print(poem_vec)
    # Set the diagonal of the poem vector based on function names
    for i in range(n_lines):
        for (function_ind, function) in enumerate(functions_name_list):
            poem_vec[i,i,function_ind] = 1 if function != "endrhyme" else 5
    # Retrieve the current poem's lines
    poem = poem_secs[z]

    # Define bit vectors and other variables for modeling the poem
    refs = [BitVec("ref" + str(i), 5) for i in range(n_lines)]
    refs_used = [Bool("refs_used" + str(i)) for i in range(n_lines)]
    cost_refs = [Int("costref" + str(i)) for i in range(n_lines)]
    
    # Add constraints for the references used in the poem
    for i in range(n_lines):
        o.add(Implies(refs_used[i], refs[i] == i))
    o.add(Sum([If(refs_used[i],1, 0) for i in range(n_lines)]) >= 1)
    o.add(Sum([If(refs_used[i],1, 0) for i in range(n_lines)]) <= 6)
    ijs = list(itertools.product(list(range(n_lines)), list(range(n_lines))))

    # Add constraints for the references and their costs
    for i in range(n_lines):
        o.add(BV2Int(refs[i]) < n_lines)
        o.add(BV2Int(refs[i]) >= 0)

        for j in range(n_lines):
            o.add(Implies(BV2Int(refs[i]) == j, refs_used[j]))
            o.add(Implies(BV2Int(refs[i]) == j, (cost_refs[i]) == np.sum(poem_vec[i][j])))
    b = Sum([If(And(refs_used[i], refs_used[j]), 10 + -3*np.sum(poem_vec[i][j]), 0)  for (i,j) in list(ijs) if i != j])
    c = Sum(cost_refs)
    q = Sum([If(refs[i] == refs[j], np.sum(poem_vec[i][j]) - 4, 0) for (i,j) in list(ijs) if i != j])
    h = o.maximize(q + c + b)
    o.check()
    o.upper(h)
    a = o.model()
    print(a.eval(q))
    print(a.eval(c))
    print(a.eval(b))

    # Extract the used references from the model
    refs_used = [i for i in range(len(refs_used)) if a[refs_used[i]]]
    print(refs_used)
    
    # Convert the reference bit vectors to integers
    refs = [int(str(a[i])) for i in refs]
    print(refs)
    
    # Create adjacency matrices for the poem
    mat = np.zeros((n_lines + len(refs_used), len(refs_used) + n_lines, poem_vec.shape[2] + 2), dtype=np.bool)
    simple_mat = np.zeros((n_lines, len(refs_used)), dtype=np.bool)
    
    # Populate the adjacency matrices based on the poem's structure
    for i in range(3, n_lines):
        for j in range(1, min(i,4)):
            if j == 1:
                mat[i,i - j, :] = np.array(list(poem_vec[i, i - j, :]) + [1,0])
            else:
                mat[i,i - j, :] = np.array(list(poem_vec[i, i - j, :]) + [0,0])

    for i in range(len(poem)):
        j = refs[i]
        j_ind = refs_used.index(refs[i])
        simple_mat[i,j_ind] = 1
        vec = poem_vec[i][j]
        for k in range(len(vec)):
            if vec[k]:
                mat[i, j_ind + len(poem), k] = 1

    # Store the processed poem's attributes
    prototype_ids.append((refs_used))
    prototypes.append([poem[q] for q in refs_used])
    print(refs)
    adj_mats.append(mat)
    simple_adjs.append(simple_mat)
    inds.append(z)

    # Save the processed data into pickle files
    pickle.dump(inds, open(folder + "/inds" + str(n_lines) +".pcl", "wb"))
    pickle.dump(prototypes, open(folder + "/prototypes" + str(n_lines) + ".pcl", "wb"))
    pickle.dump(prototype_ids, open(folder + "/prototype-ids" + str(n_lines) + ".pcl", "wb"))
    pickle.dump(adj_mats, open(folder + "/adj_mats" + str(n_lines) + ".pcl", "wb"))
    pickle.dump(simple_adjs, open(folder + "/simple_mats" + str(n_lines) + ".pcl", "wb"))

    # Generate a string representation of the poem with color-coded references
    colors = ["red", "green", "blue", "brown", "pink", "purple"]
    string = ""
    for (q, line) in enumerate(poem):
        string += color_format(colors[refs_used.index(refs[q])]).format(line)

    # Uncomment the following lines to generate HTML and PDF versions of the poem
    # open(folder + "/poems/poem" + str(z) + ".html", "w+").write(string)
    # pdfkit.from_url(folder + "/poems/poem" + str(z) + ".html", folder + "/poems/poem" + str(z) + ".pdf")
