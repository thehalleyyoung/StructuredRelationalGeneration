
import os
import sys
import warnings

# Suppress specific user warnings during execution
warnings.simplefilter(action='ignore', category=UserWarning)

# Define a list of folder names that will be created
folders = "models, generate-poems, pickles".split(", ")
for folder in folders:
    # Check if the folder exists and create it if it does not
    if not os.path.exists(folder):
        os.mkdir(folder)

# Define the Python version to use for running subsequent scripts
_version = "3.6"

# Run a sequence of Python scripts using the specified Python version
# Each script is responsible for a step in processing or analysis

# Create pickles from JSON data
os.system(_version + " createpickle_fromjson.py")

# Draw dense adjacency matrices
os.system(_version + " drawadjacencies.py")

# Get 10-line fragments with the most internal rhyme/meter
os.system(_version + " getbestdivide.py")

# Generate the programmatic structure of the data
os.system(_version + " genprogram.py")

# Learn the structure of the poems using a variational autoencoder
os.system(_version + " poemstrucvae.py")

# Write poems using a pretrained BERT model conditioned on the learned structure
os.system(_version + " writepoem.py")
