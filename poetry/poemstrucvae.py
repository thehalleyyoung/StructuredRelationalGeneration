
import torch as t
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle
import copy
import numpy as np
import os
import random

# Define the directory where pickled data is stored
folder = "pickles"

# Set the size of the latent space
latent_size = 50

# Define the Encoder neural network
class Encoder(nn.Module):
    def __init__(self, in_size, latent_size):
        super(Encoder, self).__init__()

        # LSTM layer for sequence processing
        self.rnn = nn.LSTM(input_size=in_size - 4,
                           hidden_size=latent_size//2,
                           num_layers=3,
                           batch_first=True,
                           bidirectional=True)
        # Output size of the encoder
        self.out_size = latent_size
        # Embedding layer for categorical input
        self.embed = nn.Embedding(9, 5)

    def forward(self, input, batch_size=8):
        # Forward pass of the encoder
        input = torch.cat((self.embed(torch.argmax(input[:, :, :9], dim=2)), input[:, :, 9:]), axis=2)
        _, (_, final_state) = self.rnn(input)

        # Process final states to form a single latent vector
        final_state = final_state.view(3, 2, batch_size, self.out_size//2)
        final_state = final_state[-1]
        h_1, h_2 = final_state[0], final_state[1]
        final_state = t.cat([h_1, h_2], 1)

        return final_state

# Define the Decoder neural network
class Decoder(nn.Module):
    def __init__(self, latent_size, out_spacing_size, out_rels_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.out_spacing_size = out_spacing_size
        self.out_rels_size = out_rels_size

        # Mapping from latent space to output space
        self.latent_to_out_spacing = nn.Sequential(nn.Linear(self.latent_size, self.out_spacing_size*5), nn.ReLU(), nn.Linear(self.out_spacing_size*5, self.out_spacing_size*10), nn.Sigmoid())
        self.latent_to_rels = nn.Sequential(nn.Linear(self.latent_size, self.out_rels_size * 5), nn.ReLU(), nn.Linear(self.out_rels_size * 5, self.out_rels_size * 10), nn.Sigmoid())

    def forward(self, z):
        # Forward pass of the decoder
        return (self.latent_to_out_spacing(z), self.latent_to_rels(z))


# Define the Predictor neural network that encapsulates encoder and decoder
class Predictor(nn.Module):
    def __init__(self, in_size, latent_size, out_spacing_size, out_rels_size):
        super(Predictor, self).__init__()
        self.in_size = in_size
        self.encoder_mu = nn.Linear(latent_size, latent_size)
        self.encoder_logvar = nn.Linear(latent_size, latent_size)
        self.encoder = Encoder(in_size, latent_size)
        self.decoder = Decoder(latent_size, out_spacing_size, out_rels_size)

    def reparameterize(self, mu, logvar):
        # Reparameterization trick to sample from a Gaussian
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Forward pass through the entire model
        x = self.encoder(x)
        mu, logvar = self.encoder_mu(x), self.encoder_logvar(x)
        z = self.reparameterize(mu, logvar)
        (spacing_out, rels_out) = self.decoder(z)
        return spacing_out, rels_out, mu, logvar

# Main execution of the script
if __name__ == "__main__":
    # Load datasets from pickles
    prototypes = pickle.load(open("pickles/prototypes10.pcl", "rb"))
    adj_mats = pickle.load(open("pickles/adj_mats10.pcl", "rb"))
    simple_mats = pickle.load(open("pickles/simple_mats10.pcl", "rb"))
    songs = [range(10) for i in range(len(simple_mats))]

    # Create dataset
    pairs = []
    max_spacing = 9
    for z in range(len(prototypes)):
        refs = [np.argmax(simple_mats[z][i, :]) for i in range(len(simple_mats[z]))]
        range_refs = sorted(range(len(set(refs))), key=lambda i: refs.index(i))
        if refs[0] == refs[1] and refs[2] == refs[3] and refs[4] == refs[5] and refs[6] == refs[7]:
            range_refs = sorted(range(len(set(refs))), key=lambda i: refs.index(i))
            refs2 = [range_refs.index(i) for i in refs]

        refs = [range_refs.index(i) for i in refs]

        vecs = []
        spacings = []
        for k in range(10):
            prev_k_indices = [1 if refs[q] == refs[k] else 0 for q in range(k)]
            spacing = (k - max([q for q in range(k) if prev_k_indices[q]])) if sum(prev_k_indices[max(k - 8, 0):k]) != 0 else (0)
            spacings.append(spacing)
            spacing_tens = torch.zeros(10)
            spacing_tens[spacing] = 1
            symmetries = torch.from_numpy(adj_mats[z][k, refs[k] + 10, :])
            vec = torch.cat((spacing_tens, symmetries))
            vecs.append(vec)
            prev_vec = copy.copy(vecs)
            if k == 9:
                X = torch.stack(prev_vec)
                pairs.append((X, torch.argmax(X[:, :10], dim=1), X[:, 10:]))

    # Set the batch size for the DataLoader
    batch_size = 8
    loader = DataLoader(pairs[:-200], batch_size=batch_size, shuffle=True)

    # Shuffle the pairs to randomize the training process
    random.shuffle(pairs)

    # Initialize the model
    model = Predictor(in_size=symmetries.shape[0] + 10, latent_size=latent_size, out_spacing_size=10, out_rels_size=symmetries.shape[0])

    # Set learning rate and optimizer
    lr = 1e-0
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.9)

    # Loss functions for different parts of the output
    criterion1 = nn.CrossEntropyLoss(weight=torch.Tensor([0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1]))  # for spacing
    criterion2 = nn.BCELoss()  # for symmetries

    # Training loop
    for epoch in range(200):
        print(epoch)
        model.train()
        total_loss = 0.
        log_interval = 100
        for batch, i in enumerate(loader):
            if i[0].shape[0] != batch_size:
                continue
            spacing, rels, mu, logvar = model(i[0])
            optimizer.zero_grad()
            loss1 = 100 * criterion1(spacing.view(batch_size, 10, 10), i[1].view(batch_size, 10))
            loss2 = criterion2(rels.view(batch_size, 10, -1), i[2].view(batch_size, 10, -1))
            loss3 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = loss1 + loss2 + loss3
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if batch % log_interval == 0 and batch > 0:
                total_loss = 0

            total_loss += loss.item()
        scheduler.step()
        # Save the model state after training
        torch.save(model.state_dict(), "models/predict_program.pth")
