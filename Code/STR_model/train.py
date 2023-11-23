from model import STR_Model
import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

def train(model, train_loader, loss_function, optimizer, device):
    # Setting the model to training mode
    model.train() 
    length = len(train_loader)
    # Looping over each batch from the training set 
    for batch_idx, (data, target) in enumerate(train_loader):  
        # Setting the data and target to device
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  
        output = model(data)  
        loss = loss_function(output, target) 
        loss.backward()
        # Updating the model parameters
        optimizer.step() 

        if batch_idx % max(1, int(length/10)) == 0:
            print(f'Train Epoch: {batch_idx} \t Loss: {loss.item():.6f}') 


def model_evaluation(model, data_loader, loss_function, device):
    # Setting the model to evaluation mode
    model.eval()
    loss = 0
    length = len(data_loader.dataset)
    with torch.no_grad():
        for data, target in data_loader:
            # Setting the data and target to device
            data, target = data.to(device), target.to(device)
            output = model(data)
            # calculating loss
            loss += loss_function(output, target).item()

    avg_loss = loss/length
    return avg_loss,


def model_fit(model, train_loader, test_loader, loss_function, optimizer, num_epochs, device, checkpoint):
    train_loss = []
    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        train(model, train_loader, loss_function, optimizer, device)
        loss = model_evaluation(model, train_loader, loss_function, device)
        train_loss.append(loss)
        
        if epoch % checkpoint == 0:
            model_file = str(epoch) + "_strmodel_" + str(loss) + ".pth"
            torch.save(model.state_dict(), model_file) 


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STR_Model().to(device)

    # loss function = 
 
    # optimizer = 

    num_epochs = 100 # can change this
    checkpoint = 20 # can change this
    batch_size = 32 # can change this
 
    # dataset = 
    # train_loader = 
    # test_loader =

    # Fitting the model
    # model_fit(model, train_loader, test_loader, loss_function, optimizer, num_epochs, device, checkpoint)

if __name__ == "__main__":
    main()