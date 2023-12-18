
import torch
import torch.nn as nn
import pickle
import random
import numpy as np
from torch.utils.data import DataLoader
import os
import itertools

# Define constants and paths
folder = "pickles"
num_features = 23
num_symmetries = num_features
max_spacing = 7 + 1
spacing_n = 6
num_references = 6
prev_references = (num_references)
batch_size = 64
num_in = 182
prop_size = 21
magenta_size = 256

# Predictor class definition, inherits from nn.Module
class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        # Define the architecture of the neural network
        self.lins6 = nn.Sequential(
            nn.Linear(num_in, 6*(max_spacing + num_references)//2),
            nn.ReLU(),
            nn.Linear(6*(max_spacing + num_references)//2, 6*(max_spacing + num_references)//4),
            nn.ReLU(),
            nn.Linear(6*(max_spacing + num_references)//4, max_spacing + 1),
            nn.Softmax()
        )
        self.relations = nn.Sequential(
            nn.Linear(num_in, num_features),
            nn.ReLU(),
            nn.Linear(num_features, 4*num_features),
            nn.Sigmoid()
        )
        self.total_in_a_row = nn.Sequential(
            nn.Linear(num_in, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.Softmax()
        )

    def forward(self, x, batch_size=batch_size):
        # Forward pass through the network
        return (
            self.lins6(x.view(batch_size, -1)),
            self.relations(x.view(batch_size, -1)),
            self.total_in_a_row(x.view(batch_size, -1))
        )

# Main training loop
if __name__ == "__main__":
    # Load data from pickle files
    prototypes = pickle.load(open(folder + "/prototypes16.pcl", "rb"))
    simple_mats = pickle.load(open(folder + "/simple_mats16.pcl", "rb"))
    symmetries = pickle.load(open(folder + "/transformervecs.pcl", "rb"))

    all_vecs = []
    pairs = []
    # Preprocess the data
    for z in range(len(prototypes)):
        print(z)
        all_vecs.append([])
        refs = [np.argmax(simple_mats[z][i, :]) for i in range(len(simple_mats[z]))]
        prev_refs = [refs[max(k - max_spacing, 0):k] for k in range(len(refs))]
        spacings = []
        for k in range(16):
            prev_k_indices = [1 if refs[q] == refs[k] else 0 for q in range(k)]
            if sum(prev_k_indices[max(k - 6, 0):k]) == 0:
                spacings.append(0)
            else:
                spacings.append(k - max([q for q in range(k) if prev_k_indices[q]]))
        for q in range(16 - 10):
            for k in range(2, 7):
                cur_space = spacings[q + k + 1]
                n_ahead = 0
                for i in range(2, min(6, len(spacings) - q - k)):
                    if spacings[q + k + i] == cur_space:
                        n_ahead += 1
                    else:
                        break
                vec1 = np.zeros((6, num_references))
                vec2 = np.zeros((6, max_spacing + 1))
                for ind in range(k):
                    vec1[ind + (6 - k), refs[ind]] = 1
                    vec2[ind + (6 - k), spacings[ind]] = 1
                try:
                    vec3 = symmetries[z][q + k][:4*num_symmetries]
                    c = np.concatenate(np.concatenate([vec1, vec2], axis=1), axis=0)
                    print(c.shape)
                    c = np.concatenate([c, vec3], axis=0)
                    all_vecs[-1] = c
                    vec4 = np.array(symmetries[z][q + k][:4*num_symmetries])
                    num_in = all_vecs[-1].shape[0]
                    pairs.append((torch.from_numpy(c).float(), torch.from_numpy(np.array([cur_space])).long(), torch.from_numpy(np.array([n_ahead])).long(), torch.from_numpy(vec4).float()))
                except:
                    print("error")

    # Prepare DataLoader for training and testing
    loader = DataLoader(pairs[:-100], batch_size=batch_size, shuffle=True)
    testloader = DataLoader(pairs[-100:], batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer, and scheduler
    model = Predictor()
    lr = 1e-1
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3.0, gamma=0.95)

    # Loss functions for different parts of the output
    criterion1 = nn.CrossEntropyLoss()  # for spacing
    criterion2 = nn.BCELoss()  # for symmetries

    # Training loop
    for epoch in range(100):
        print(epoch)
        model.train()
        total_loss = 0.
        ntokens = 17*6
        tot_acc = 0
        tot = 0
        log_interval = 100
        for batch, i in enumerate(loader):
            if i[0].shape[0] != batch_size:
                continue
            tot += batch_size
            data_x, spacing_in, ahead_in, rels_in = i
            optimizer.zero_grad()
            spacing, rels, in_a_row = model(data_x)
            loss1 = criterion1(spacing, spacing_in.view(batch_size))
            loss2 = criterion1(in_a_row, ahead_in.view(batch_size))
            loss3 = criterion2(rels, rels_in.view(batch_size, -1))
            loss = loss1 + loss2 + loss3
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            if batch % log_interval == 0 and batch > 0:
                print(("batch " + str(batch), "loss " + str(total_loss)))
                total_loss = 0
                tot = 0

        # Adjust learning rate
        scheduler.step()

        # Evaluation loop
        model.eval()
        for batch, i in enumerate(testloader):
            total_loss2 = 0
            if i[0].shape[0] != batch_size:
                continue
            tot += batch_size
            data_x, spacing_in, ahead_in, rels_in = i
            spacing, rels, in_a_row = model(data_x)
            loss1 = criterion1(spacing, spacing_in.view(batch_size))
            loss2 = criterion1(in_a_row, ahead_in.view(batch_size))
            loss3 = criterion2(rels, rels_in.view(batch_size, -1))
            loss = loss1 + loss2 + loss3
            total_loss += loss.item()
            log_interval = 50
            if batch % log_interval == 0 and batch > 0:
                print("test loss " + str(loss))
                total_loss2 = 0
                tot = 0

        # Save and load the model state
        torch.save(model.state_dict(), "graphnn/predictprogram.pth")
        model.load_state_dict(torch.load("graphnn/predictprogram.pth"))

# Function to create graph representation from noise
def create_from_noise(model, graph=True):
    # Initialization and noise generation
    max_guess = 0
    for z in range(16):
        # Initialize variables
        aheads = []
        ref_mean = []
        prev_refs = [random.randint(0, 6) for i in range(3)]
        prev_feats = [True for i in range(num_symmetries) for j in range(3)]
        prev_spacings = [0, 0, 0]
        prev_vec1 = np.zeros((6, num_references))
        prev_vec2 = np.zeros((6, max_spacing + 1))
        prev_vec1[4, random.randint(0, 4)] = 1
        prev_vec1[5, random.randint(0, 4)] = 1
        prev_vec1[3, random.randint(0, 4)] = 1
        prev_vec2[3, random.randint(0, 1)] = 1
        prev_vec2[4, random.randint(0, 1)] = 1
        prev_vec2[5, random.randint(0, 1)] = 1
        prev_vec3 = np.ones(4*num_features)
        prev_vec = np.concatenate(np.concatenate([prev_vec1, prev_vec2], axis=1))
        prev_syms = []
        prev_vec = np.concatenate([prev_vec, prev_vec3], axis=0)
        # Generate a sequence based on noise
        while len(prev_refs) < 16 + 3:
            # Model predictions
            spacing, refs, ahead = model(torch.from_numpy(prev_vec).float(), 1)
            ahead = ahead + 0.05*torch.randn(ahead.shape)
            spacing = spacing + 0.2*torch.randn(spacing.shape)
            spacing = int(torch.argmax(spacing))
            ahead = int(torch.argmax(ahead))
            aheads.append(ahead)
            refs = refs + 0.2*torch.randn(refs.shape)
            refs = (refs)[0].view(4, num_symmetries).detach().numpy()
            ref_mean.append(sum(refs))
            # Update vectors with predictions
            new_vec1 = np.zeros((6, num_references))
            new_vec2 = np.zeros((6, max_spacing + 1))
            for q_ in range(ahead + 1):
                for i in range(5):
                    new_vec1[i, :] = prev_vec1[i - 1, :]
                    new_vec2[i, :] = prev_vec2[i - 1, :]
                ref_ = prev_refs[-1*spacing] if (spacing != 0 and spacing < len(prev_refs) and (len(prev_refs) < 9 or (len(prev_refs[:-9]) >= 3 and len(prev_refs[:-4]) > 1))) else max([k for k in range(6) if all([k_ in prev_refs for k_ in range(k)])]) if (len(set(prev_refs)) <= 5 and random.uniform(0,1) < 0.5) else random.randint(0,5)
                poss_new = random.uniform(0,1)
                if poss_new < len([i for i in prev_refs[-4:] if i == ref_]):
                    ref_ = random.randint(0,4)
                    spacing = 0 if ref_ not in prev_refs[-7:] else len(prev_refs) - [k for k in range(len(prev_refs)) if prev_refs[k] == ref_][-1]
                new_vec2[-1, spacing] = 1

                if ref_ > 5:
                    ref_ = random.randint(0, 5)
                new_vec1[-1, ref_] = 1
                prev_refs.append(ref_)
                for i in range(4):
                    for j in range(num_symmetries):
                        if i == 0:
                            refs[i][j] = min(1.0, max(0.0, refs[i][j] + .4*np.random.normal()))
                        else:
                            refs[i][j] = min(1.0, max(0.0, refs[i][j] + .4*np.random.normal() - 0.2))
prev_syms.append(np.concatenate(refs))
                new_vec3 = np.zeros(4*num_features)
                new_vec3 = np.concatenate(refs)
                prev_vec = np.concatenate(np.concatenate([new_vec1, new_vec2], axis=1))
                prev_vec = np.concatenate([prev_vec, new_vec3], axis=0)
        refs = prev_refs
        spacings = prev_spacings
        # Convert to graph representation
        x, edge_inds, edge_attrs = interpretAsGraph(prev_refs[3:], prev_syms[:4*len(prev_refs)], graph)
        dis_guess = 1 # Placeholder for a distance function
        if dis_guess > max_guess:
            cur_best = (x, edge_inds, edge_attrs)
            max_guess = dis_guess
        return cur_best

# Function to interpret the sequence as a graph
def interpretAsGraph(ref_vecs, sym_vecs, graph):
    # Initialize variables for graph construction
    prot_vecs = len(set(ref_vecs))
    num_total_edge_attrs = 49
    prot_vecs = 0
    seq_vecs = {}
    edge_inds = []
    edge_attrs = []
    for i in range(len(ref_vecs)):
        ref_syms = sym_vecs[i][:num_symmetries] > 0.25
        prev_syms = [(sym_vecs[i][num_symmetries*j:num_symmetries*(j + 1)]) > 0.1 for j in range(1, 4)]
        seq_vecs[i] = ref_vecs[i]
        if graph:
            # Construct graph edges and attributes
            edge_inds.append([seq_vecs[i], prot_vecs + i])
            edge_attr = np.zeros(num_total_edge_attrs)
            for k in range(num_symmetries):
                edge_attr[k] = float(ref_syms[k])
            edge_attrs.append(edge_attr)
            # Add sequential and previous edges
            if i < len(ref_vecs) - 1:
                edge_inds.append([prot_vecs + i, prot_vecs + i + 1])
                edge_attr = np.zeros(num_total_edge_attrs)
                edge_attr[2*num_symmetries + 1] = 1
                edge_attrs.append(edge_attr)
            for j in range(0, min(i, 3)):
                edge_inds.append([prot_vecs + i, prot_vecs + i - j])
                edge_attr = np.zeros(num_total_edge_attrs)
                for k in range(num_symmetries):
                    edge_attr[k + num_symmetries] = float(prev_syms[2 - j][k])
                edge_attrs.append(edge_attr)
        else:
            syms.append(ref_syms)
    if not graph:
        return (ref_vecs, syms)
    # Construct tensor representations of the graph
    nodes = torch.normal(0, 0.2, (len(ref_vecs) + len(set(ref_vecs)), 200))
    return (nodes.float(), torch.from_numpy(np.array(edge_inds)).long(), torch.from_numpy(np.array(edge_attrs)).float())
