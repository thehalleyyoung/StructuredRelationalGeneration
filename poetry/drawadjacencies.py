
# Importing necessary libraries
import random
import numpy as np
import inspect
import poetryrelations
import pickle
from inspect import getmembers, isfunction

# Extracting all functions from the poetryrelations module, excluding the 'word' function
functions_list = [o for o in getmembers(poetryrelations) if isfunction(o[1]) if o[0] != "word"]
# Creating a list of function names, excluding 'word'
functions_name_list = [o[0] for o in functions_list if o[0] != "word"]

# Loading a list of poems from a pickle file
poems = pickle.load(open("pickles/poems.pcl", "rb"))

# Initializing a dictionary to store poem vectors
poem_vals = {}

# Loop through each poem in the list of poems
for (poem_ind, poem) in enumerate(poems):
    # Calculate one third length of the poem
    one_third_poem = len(poem) // 3
    # Truncate the first and last third of the poem
    poem = poem[one_third_poem:-1 * one_third_poem]
    # Initialize a 3D numpy array to hold vectors for each poem
    vec = np.zeros((len(poem), 10, len(functions_list)))

    # Loop through the poem to populate the vector
    for j in range(len(poem) - 10):
        for k in range(j + 1, j + 9):
            # Loop through each function in the functions list
            for (ind, (name, f)) in enumerate(functions_list):
                # Call the function with the poem and two positions
                if f(poem, j, k):
                    # If the function is 'endrhyme', assign a value of 5
                    if name == "endrhyme":
                        vec[j, k - j, ind] = 5
                    # Otherwise, assign a value of 1
                    else:
                        vec[j, k - j, ind] = 1
    # Store the vector in the poem_vals dictionary with the poem as key
    poem_vals[tuple(poem)] = vec

    # Every 20 poems, dump the poem_vals dictionary to a pickle file
    if poem_ind % 20 == 19:
        pickle.dump(poem_vals, open("pickles/poem_vecs.pcl", "wb"))
