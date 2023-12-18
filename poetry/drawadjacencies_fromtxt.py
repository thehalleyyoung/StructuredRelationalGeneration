
# Import necessary libraries
import random
import numpy as np
import inspect
import poetryrelations
import pickle
from inspect import getmembers, isfunction
import sys

# Retrieve all functions from the poetryrelations module, excluding the 'word' function
functions_list = [o for o in getmembers(poetryrelations) if isfunction(o[1]) if o[0] != "word"]
# Extract the names of these functions for later use
functions_name_list = [o[0] for o in functions_list if o[0] != "word"]

# Define the folder where the poems data will be stored
poetry_folder = "poems-data"

# Load the poems from a pickle file
poems = pickle.load(open(poetry_folder + "/poems.pcl", "rb"))
# Print the number of poems loaded
print(len(poems))
# Store the number of poems for later use
num_poems = len(poems)

# Initialize dictionaries to store values and vectors for poems
poem_vals = {}
poem_vecs = {}

# Iterate over each poem in the dataset
for (poem_ind, poem) in enumerate(poems):
    # Print the current index and the total number of poems
    print("i: " + str(poem_ind) + " of " + str(num_poems - 1))
    # Print the current poem
    print(poem)
    # Initialize a 3D numpy array to store function outputs for each pair of lines in the poem
    vec = np.zeros((len(poem), len(poem), len(functions_list)))
    # Iterate over each line in the poem
    for j in range(len(poem)):
        print("j: " + str(j))
        # Iterate over each line in the poem again to form pairs
        for k in range(len(poem)):
            # Ensure we're not comparing the same line with itself
            if j != k:
                # Iterate over each function in the list
                for (ind, (name, f)) in enumerate(functions_list):
                    try:
                        # If the function returns True for the line pair, update the vector
                        if f(poem, j, k):
                            # Special case for 'endrhyme' function
                            if name == "endrhyme":
                                print("in")
                                vec[j, k, ind] = 5
                            else:
                                vec[j, k, ind] = 1
                    except:
                        # Print an error message if the function call fails
                        print("error")
    # Store the vector for the current poem in the dictionary using the poem as a key
    poem_vals[tuple(poem)] = vec
    # Periodically save the poem vectors to a pickle file
    if poem_ind % 20 == 19:
        pickle.dump(poem_vals, open(poetry_folder + "/poem_vecs.pcl", "wb"))
