
from z3 import *  
import pickle  
import numpy as np  
from numba import jit, cuda  

# Load precomputed poem vectors from a pickle file
poem_vecs = pickle.load(open("pickles/poem_vecs.pcl", "rb"))

# Define the number of relations between lines in the poem
num_relations = 7

# Initialize dictionaries to store scores and vectors for poem sections
poem_sec_scores = {}
poem_sec_vecs = {}

# Set the timeout parameter for the Z3 optimizer (30 seconds)
set_param(timeout=30*1000)  # Increase for more accuracy, decrease for faster runtime

# Function to calculate the total number of symmetry relations in 10 lines after line i in a vector
def getSum(vec, i):
    return np.sum([np.sum(vec[i + a, :10 - a, :]) for a in range(10)])

# Function to adjust the vector for a specific section of the poem
def getAdj(vec, sec):
    new_vec = np.zeros((10, 10, 7))
    for i in range(10):
        for j in range(10 - i):
            new_vec[i, i + j, :] = vec[sec + i, j, :]
            new_vec[i + j, i, :] = new_vec[i, i + j, :]
    return new_vec

# Iterate over the poem vectors
for (ind, (poem, vec)) in enumerate(poem_vecs.items()):
    # Create boolean variables for each line in the poem
    lines = [Bool(str(i)) for i in range(len(vec))]
    # Calculate scores for each line in the poem
    scores = [getSum(vec, i) for i in range(len(vec) - 10)]

    # Initialize the Z3 optimizer
    o = Optimize()

    # Add constraints to the optimizer
    for i in range(len(vec)):
        c = Not(And(lines[i], Or(lines[i + 1:min(i + 10, len(vec)):])))
        o.add(c)
    try:
        # Attempt to maximize the sum of scores for selected lines
        h = o.maximize(Sum([If(lines[i], scores[i], 0) for i in range(len(vec) - 10)]))
        o.check()  # Check if the optimization problem is solvable
        o.upper(h)  # Get the upper bound of the maximum score
        a = o.model()  # Get the model (solution) from the optimizer
        # Determine which sections of the poem are selected
        secs = [k for k in range(len(vec) - 10) if a.eval(lines[k])]
        print("in x")
    except:
        # Print an error message if optimization fails
        print("in error")
        continue

    # Store vectors and scores for the selected sections of the poem
    for sec in secs:
        poem_sec_vecs[tuple(list(poem)[sec:sec + 10])] = getAdj(vec, sec)
        poem_sec_scores[tuple(list(poem)[sec:sec + 10])] = np.sum(getAdj(vec, sec))

    # Periodically save the computed scores and vectors to pickle files
    if ind % 10 == 9:
        pickle.dump(poem_sec_scores, open("pickles/poem_sec_scores.pcl", "wb"))
        pickle.dump(poem_sec_vecs, open("pickles/poem_sec_vecs.pcl", "wb"))
        print(len(poem_sec_vecs))
