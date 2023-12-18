
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import pickle
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Data, DataLoader
import numpy as np
from torch import optim
import random
from vaemodel import VAE

# Function to compute the covariance matrix
def cov(m, y=None):
    if y is not None:
        m = torch.cat((m, y), dim=0)
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov

# Define constants
magenta_size = 256
num_features = 256
(all_nodes, all_edge_inds, all_edge_attrs) = pickle.load(open("pickles/graphelements.pcl", "rb"))
num_edge_features = 47  # Placeholder value, will be updated with actual data
num_symmetries = 23

# Loss function definition
def loss_function(recon_x, x, mu, logvar, z, attr_predict, edge_attr):
    std_loss = nn.L1Loss(reduction="mean")(torch.std(recon_x, dim=0), torch.std(x, dim=0))
    recon_loss1 = 10 * nn.MSELoss(reduction="sum")(recon_x.view(-1), x.view(-1))
    cosine_sim = 10 * nn.CosineEmbeddingLoss(reduction="sum")(recon_x.view(-1, num_features), x.view(-1, num_features), torch.ones(recon_x.shape[0]))
    try:
        recon_loss2 = 20 * nn.BCELoss(reduction="mean")(attr_predict, edge_attr)
    except:
        recon_loss2 = 0
    mean_loss = 0.5 * torch.sum(recon_x * recon_x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    trace_covar = 0.000001 * torch.trace(cov(z))
    return (KLD + recon_loss1 + recon_loss2, {"cosine_sum": cosine_sim, "std_loss": std_loss, "sum": mean_loss, "KLD": KLD, "TC": trace_covar, "r1": recon_loss1, "r2": recon_loss2})

if __name__ == "__main__":
    # Load data from pickle files
    (all_nodes, all_edge_inds, all_edge_attrs) = pickle.load(open("pickles/graphelements.pcl", "rb"))
    num_edge_features = all_edge_attrs[0][0].shape[0]
    
    # Create a list of PyTorch Geometric Data objects
    dataset = []
    for i in range(len(all_nodes)):
        dataset.append(Data(x=torch.from_numpy(np.array(all_nodes[i])).float(), edge_index=torch.from_numpy(np.array(all_edge_inds[i])).long().T, edge_attr=torch.from_numpy(np.array(all_edge_attrs[i])).float()))

    # Shuffle the dataset
    random.shuffle(dataset)
    
    # Initialize the VAE model
    model = VAE()

    # Set up the optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.2)

    # Create a DataLoader for batch processing
    loader = DataLoader(dataset, shuffle=True)
    
    # Initialize variables for tracking progress
    prev_loss_parts = {}
    recons = []
    xs = []
    scheduler_steps = 0
    prev_epoch_loss = 0
    prev_step = 0
    
    # Training loop
    for epoch in range(10000):
        loss_parts_sum = {}
        print(epoch)
        print("in train")

        # Initialize loss accumulators
        train_loss = [0, 0, 0, 0, 0]
        epoch_loss = 0
        
        # Begin processing batches
        for batch_ind, batch in enumerate(dataset):
            print(batch.x.shape)
            
            # Perform a forward pass through the model
            recon, z, mu, logvar, actual_attr = model(batch.x, batch.edge_index, batch.edge_attr)
            
            # Compute the loss
            loss, loss_parts = loss_function(recon, batch.x, mu, logvar, z, actual_attr, batch.edge_attr[:, :num_symmetries] + batch.edge_attr[:, num_symmetries:num_symmetries * 2])
            
            # Accumulate the loss components
            for k, v in loss_parts.items():
                loss_parts_sum[k] = loss_parts_sum.get(k, 0) + v
            
            # Store reconstructions for later analysis
            recons.append(list(recon[-16:, :magenta_size].detach().numpy()))
            xs.append(batch.x[-16:, :magenta_size].numpy())

            # Check for reasonable loss values before backpropagation
            if loss < 1e11:
                loss.backward()
                
                # Clip gradients to prevent exploding gradients
                nn.utils.clip_grad_value_(model.parameters(), 1)

                # Update model parameters
                optimizer.step()
                nn.utils.clip_grad_value_(model.parameters(), 1)
                
                # Update loss accumulators
                train_loss[0] += loss_parts["r1"].item()
                train_loss[1] += loss_parts["r2"].item()
                train_loss[2] += loss_parts["KLD"].item()
                train_loss[3] += loss_parts["TC"].item()
                train_loss[4] += loss.item()
                
                # Reset gradients for the next batch
                optimizer.zero_grad()
            else:
                print((epoch, scheduler_steps))
                exit()
                scheduler.step()
                print(loss_parts, prev_loss_parts)
                print("error")
                train_loss[0] += 1e11
                train_loss[1] += 1e11

            prev_loss_parts = loss_parts
            
            # Print training loss every 1000 batches
            if batch_ind % 1000 == 999:
                print("trainloss is " + str(train_loss))
                epoch_loss += train_loss[4]
                print(loss_parts_sum)
                loss_parts_sum = {}
                recons = []
                xs = []
                train_loss = [0, 0, 0, 0, 0]
        
        # Adjust learning rate if necessary
        if prev_epoch_loss != 0:
            print(epoch_loss / prev_epoch_loss)
        if (prev_epoch_loss != 0 and (epoch_loss / prev_epoch_loss) > 1 and prev_step > 2 and scheduler_steps < 12):
            print("in step")
            scheduler.step()
            scheduler_steps += 1
            prev_step = 0
        else:
            prev_step += 1

        prev_epoch_loss = epoch_loss

        # Save the model every 10 epochs
        if epoch % 10 == 9:
            torch.save(model.state_dict(), "graphnn/graphvae.pth")
        
        # Exit condition for the learning rate scheduler
        if scheduler_steps == 10:
            exit()
            optimizer = optim.Adam(model.parameters(), lr=1e-2)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.2)
            scheduler_steps = 0
        
        # Shuffle the dataset for the next epoch
        random.shuffle(dataset)
