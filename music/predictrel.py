
import torch
import torch.utils.data
from torch.nn import functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import pickle
from fastprogress import master_bar, progress_bar
import numpy as np
import random
import os
from scipy.stats import norm

# Define the folder where the pickle files are stored
folder = "pickles"

# Define the number of relationships to predict
num_rels = 23

# Define the neural network architecture for relationship prediction
class RelPredictor(nn.Module):
    def __init__(self):
        super(RelPredictor, self).__init__()
        # Sequential model with linear layers and ReLU activations
        self.nn = nn.Sequential(
            nn.Linear(512, 200),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, num_rels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Forward pass through the network
        return self.nn(x)


# Main execution of the script
if __name__ == "__main__":
    # Load data from pickle file
    data_x, data_y = pickle.load(open(folder + "/reldata.pcl", "rb"))
    print(data_x[0].shape)
    num_rels = len(data_y[0])
    print(num_rels)

    # Define a custom loss function
    def loss_function(x, y):
        return torch.sum(5 * nn.BCELoss()(x, y) * (y > 0.5).float() + nn.BCELoss()(x, y) * (y < 0.5).float())

    # Create a dataset from the loaded data
    dataset = [(torch.from_numpy(data_x[i]).float(), torch.from_numpy(data_y[i]).float()) for i in range(len(data_x))]
    batch_size = 8

    # Initialize the model and optimizer
    model = RelPredictor()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=1.0)

    # Shuffle and split the data into training and testing sets
    data = dataset
    print(len(data))
    random.shuffle(data)
    trainloader = torch.utils.data.DataLoader(data[:-2000], shuffle=True, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(data[-2000:], shuffle=True, batch_size=batch_size)

    # Initialize variables to track the minimum test loss
    min_test_loss = 1000000000
    n_epochs = 100

    # Create a directory to save the model if it doesn't exist
    if not os.path.exists(folder + "/relpredict"):
        os.makedirs(folder + "/relpredict")

    # Training loop
    for epoch in range(n_epochs):
        print("in train")
        model.train()
        train_loss = 0
        tot_loss = 0
        tot_acc = 0
        tot_false = 0

        # Iterate over batches of data
        for batch_idx, (data_x, data_y) in enumerate(trainloader):
            optimizer.zero_grad()
            trans = model(data_x)
            loss = loss_function(trans, data_y)
            for j in range(num_rels):
                tot_acc += len([k for k in range(batch_size) if bool(trans[k, j] > 0.5) == bool(data_y[k, j])])
                tot_false += len([k for k in range(batch_size) if bool(trans[k, j] < 0.5)])
            tot_loss += loss.item()

            # Log accuracy and loss every 200 batches
            if batch_idx % 200 == 199 and len(data_x) == batch_size:
                print("acc " + str(tot_acc / (batch_size * 200 * num_rels)))
                print("tot false " + str(tot_false / (batch_size * 200 * num_rels)))
                tot_acc = 0
                tot_loss = 0
                tot_false = 0
                torch.save(model.state_dict(), "graphnn/relpredictor.pth")
                scheduler.step()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
