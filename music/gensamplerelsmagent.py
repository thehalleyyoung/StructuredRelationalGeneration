
# Import necessary libraries
import pickle
import numpy as np
import random
from inspect import getmembers, isfunction
import symmetries  # This module is assumed to contain various symmetry functions

# Extract a list of functions from the symmetries module, excluding 'mod12Same'
functions_list = [o for o in getmembers(symmetries) if isfunction(o[1]) and o[0] != "mod12Same"]
functions_name_list = [o[0] for o in getmembers(symmetries) if isfunction(o[1]) and o[0] != "mod12Same"]

# Define the folder where pickle files are stored
folder = "pickles"

# Define a function to concatenate a list of lists
def concat(xss):
    new = []
    for xs in xss:
        new.extend(xs)
    return new

# Load data from pickle files and concatenate if necessary
data_z = concat(pickle.load(open(folder + "/meas16.pcl", "rb")))
print(len(data_z))
print(data_z[0])
data_x = pickle.load(open(folder + "/analyzedmagents2.pcl", "rb"))
print(len(data_x))

# Ensure that the lengths of data_z and data_x are equal
assert(len(data_z) == len(data_x))

# Print the lengths of the datasets
print(len(data_z))
print(len(data_x))

# Generate pairs of indices for creating relationships between data points
pairs_0 = [random.choice(range(3, len(data_x) - 4)) for i in range(80000)]
pairs_1 = [i + random.randint(-3, 3) for i in pairs_0]

# Initialize lists to hold the combined data and the relationships
all_magents = []
all_rels = []

# Iterate over the generated index pairs and build the relationships
for z in range(len(pairs_0)):
    try:
        # Concatenate the corresponding data points and reshape
        all_magents.append(np.reshape(np.concatenate([data_x[pairs_0[z]][0, :], data_x[pairs_1[z]][0, :]]), (512)))
        human = (data_z[pairs_0[z]], data_z[pairs_1[z]])
        rels = np.zeros(len(functions_name_list))

        # Check for each symmetry function if it applies to the current pair
        for (ind, (func_name, func)) in enumerate(functions_list):
            if func(human[0], human[1]):
                rels[ind] = 1
        all_rels.append(rels)
        print(z)
    except:
        print("fail")

# Shuffle the indices of the combined data to randomize the order
inds = list(range(len(all_rels)))
random.shuffle(inds)
all_magents = [all_magents[ind] for ind in inds]
all_rels = [all_rels[ind] for ind in inds]

# Dump the combined data and relationships into a pickle file
pickle.dump((all_magents, all_rels), open(folder + "/reldata.pcl", "wb"))
