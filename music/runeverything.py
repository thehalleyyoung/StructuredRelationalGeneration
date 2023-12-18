
import sys
import os
from inspect import getmembers, isfunction

# Define the folders to be created if they do not exist
folders = "graphnn, pickles, referenceMids, generatedMids, tmpmids".split(", ")
for folder in folders:
    # Create the folder if it does not exist
    if not os.path.exists(folder):
        os.mkdir(folder)

# Get the method from command line arguments or set default to "z3"
method = sys.argv[1] if len(sys.argv) > 1 else "z3"

# Define the  version to use for running the scripts
_version = "3.6"

# Function to perform program synthesis
def performProgramSynthesis():
    # Preprocess the graph to get dense adjacency tensors
    os.system(_version + " preprocessgraph.py")
    # Synthesize reference measures
    os.system(_version + " genprogram.py")

# Filter out non-4/4 or non-16 bar pieces from the corpus
os.system(_version + " preprocesscorpus.py")
# Perform program synthesis
performProgramSynthesis()
# Train the program predictor
os.system(_version + " trainProgramPredictor.py")

# Function to generate music using Z3 method
def genZ3():
    # Generate reference measures sampled from MusicVAE
    os.system(_version + " music_vae_generate_random.py")
    # Generate sample programs
    os.system(_version + " gensamplescaffolding.py")
    # Generate music conditioned on sampled reference measures and programs
    os.system(_version + " genstrucz3.py")

# Function to generate music using VAE method
def genVAE():
    # Learn the symmetry classifier for the semantic loss
    os.system(_version + " gensamplerelsmagent.py")
    # Predict relationships
    os.system(_version + " predictrel.py")

    # Turn data into torch_geometric graph format
    os.system(_version + " genrealgraphs.py")
    # Train the Graph VAE
    os.system(_version + " graphvaetrain.py")
    # Generate Magenta Embeddings by sampling from Graph VAE
    os.system(_version + " vae2gen.py")
    # Turn embeddings into MIDI
    os.system(_version + " music_vae_generate_reals.py")

# Decide which generation method to use based on the method argument
if method.lower() == "z3":
    # Generate music using Z3 method
    genZ3()
elif method.lower() == "vae":
    # Generate music using VAE method
    genVAE()
else:
    # Print error message if an invalid method is provided
    print("invalid method - use 'z3' or 'vae' as the first argument")
