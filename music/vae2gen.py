
# Import necessary modules
from vaemodel import VAE  # Import the Variational Autoencoder model
import pickle  
import numpy as np  
from trainprogrampredictor import Predictor, create_from_noise  # Import the Predictor module and noise creation function
import torch  

# Initialize a Predictor instance for structure prediction
struc_predictor = Predictor()
# Load the trained weights into the structure predictor model
struc_predictor.load_state_dict(torch.load("graphnn/predictprogram.pth"))

# Initialize a VAE instance for the conditional VAE model
cond_vae = VAE()
# Load the trained weights into the conditional VAE model
cond_vae.load_state_dict(torch.load("graphnn/graphvae.pth"))

# Set the size for the magenta portion of the output
magenta_size = 256
# Define the number of symmetries (not used in this script)
num_symmetries = 22

# Initialize lists to store the magenta vectors and random magenta vectors
magents = []
random_magents = []

# Generate 2000 magenta vectors using the structure predictor and the conditional VAE
for i in range(2000):
    # Generate a random graph structure using the structure predictor
    x, edge_index, edge_attr = create_from_noise(struc_predictor)
    # Decode the generated graph structure using the conditional VAE decoder
    recon = cond_vae.decoder(x, edge_index.T, edge_attr)

    # Extract the last 20 magenta vectors from the reconstructed output and convert to numpy array
    recon = list(recon[-20:, :magenta_size].detach().numpy())
    # Append the reconstructed magenta vectors to the list
    magents.append(recon)

# Serialize the list of magenta vectors and save to a file
pickle.dump(magents, open("pickles/recons.pcl", "wb"))
