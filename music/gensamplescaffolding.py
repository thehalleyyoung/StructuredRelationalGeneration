
import torch
import torch.nn as nn
import pickle
import random
import numpy as np
from torch.utils.data import DataLoader
from predictnext3 import Predictor, num_references, num_features, max_spacing
from music21 import *
import pitchprominence
from inspect import getmembers, isfunction

# Load the trained model
model = Predictor()
model.load_state_dict(torch.load("graphnn/predictprogram.pth"))

# Load feature names used in the model
feat_names = pickle.load(open("../symmfuncs.pcl", "rb"))

# Set the total number of iterations for data generation
n_tot = 1000000

# Function to transpose a sequence of notes to the key of C
def transposeToC(xss):
    pcs = range(-6,7)
    pcs_ordering = {0:7, 7:5, 5:4, 4:2, 2:2, 11:2, 9:2, 3:0, 8:0, 10:-1, 1:-5, 6:-5}
    transposed = [[[(x[0] + pc, x[1]) for x in xs] for xs in xss] for pc in pcs]
    transposed_best = max(transposed, key=lambda i: sum([sum([pcs_ordering[k[0] % 12] * k[1] for k in j]) for j in i]))
    return transposed_best

# Retrieve the list of functions from the pitchprominence module
functions_list = [o for o in getmembers(pitchprominence) if isfunction(o[1])]

# Function to get a pitch profile for a measure
def getProfile(measure):
    measure_profile = np.zeros(12)
    for pc in range(12):
        try:
            for (name, func) in functions_list:
                (has_feat, weight) = func(measure, pc)
                if has_feat:
                    measure_profile[pc] += weight
        except:
            print(measure)
    return measure_profile

# Initialize lists to store generated data
all_spacings = []
all_refs = []
all_feats = []
ind = 0
ref_mean = []
aheads = []

# Main loop to generate data
while ind < n_tot:
    # Initialize the previous references, features, and spacings
    prev_refs = [random.randint(0, 4) for i in range(3)]
    prev_feats = [{k: True for k in feat_names} for j in range(3)]
    prev_spacings = [0, 0, 0]

    # Initialize the input vectors for the model
    prev_vec1 = np.zeros((6, num_references))
    prev_vec2 = np.zeros((6, max_spacing + 1))
    prev_vec3 = np.ones(2 * num_features)
    prev_vec = np.concatenate([prev_vec1, prev_vec2], axis=1)
    prev_vec = np.concatenate([prev_vec, prev_vec3], axis=0)

    # Generate sequences until we have a 16-measure piece
    while len(prev_refs) < 16:
        spacing, refs, ahead = model(torch.from_numpy(prev_vec).float())
        # Add randomness to the model's predictions
        ahead = ahead + 0.4 * torch.randn(ahead.shape)
        spacing = spacing + 0.01 * torch.randn(spacing.shape)
        spacing = int(torch.argmax(spacing))
        ahead = int(torch.argmax(ahead))
        aheads.append(ahead)
        refs = refs + 0.1 * torch.randn(refs.shape)
        if random.uniform(0, 1) < 0.3:
            refs = np.ones(refs[0].shape)
        else:
            refs = (refs > 0.73)[0].numpy()
        ref_mean.append(sum(refs))

        # Update the input vectors with the new predictions
        new_vec1 = np.zeros((6, num_references))
        new_vec2 = np.zeros((6, max_spacing + 1))
        # ... (The rest of the vector update logic goes here)

        # Check if the generated sequence meets certain criteria
        if len(set(prev_refs)) == 5 and ...:
            all_spacings.append(prev_spacings[:16])
            all_feats.append(prev_feats[:16])
            all_refs.append(prev_refs[:16])
            ind += 1
            print((ind, np.mean(ref_mean)))

# Retrieve actual measures from the generated reference vectors
all_notes = []
all_ref_measures = []
all_prev_profiles = []

# Process each generated sequence
for z in range(n_tot):
    ref_measures = []
    # ... (Code to parse MIDI files and extract measures)

# Save the generated data to pickle files
pickle.dump(all_ref_measures, open("pickle/allrefmeasures.pcl", "wb"))
pickle.dump(all_feats, open("pickle/reference_features.pcl", "wb"))
pickle.dump(all_refs, open("pickle/allrefs.pcl", "wb"))
