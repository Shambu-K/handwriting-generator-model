import torch
from tqdm import tqdm
import os
from model import STR_Model
from dataset.iam_dataloader import HandwritingDataset
from loss.stroke_loss import STR_Loss
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from util.stroke_plotting import get_strokes, plot_word_strokes, plot_str_word_strokes, animate_word
import matplotlib.pyplot as plt
import numpy as np
from fastdtw import fastdtw
from loss.dtw_alignment import plot_dtw_path

def train(model, train_loader, loss_function, optimizer, device, epoch=0):
    # Setting the model to training mode
    model.train() 
    length = len(train_loader)
    # Looping over each batch from the training set 
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), desc=f'Epoch {epoch}', total=length):  
        optimizer.zero_grad()  
        output = model(data)  
        loss = loss_function(output, target) 
        loss.backward()
        # Updating the model parameters
        optimizer.step() 

        if batch_idx % 100 == 0:
            print(f'   Batch: {batch_idx} | Loss: {loss.item()}')
            
    return loss.item()


def model_evaluation(model, data_loader, loss_function, device):
    # Setting the model to evaluation mode
    model.eval()
    loss = 0
    length = len(data_loader.dataset)
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            # calculating loss
            loss += loss_function(output, target).item()

    avg_loss = loss/length
    return avg_loss


def model_fit(model, train_loader, loss_function, optimizer, scheduler, num_epochs, device, checkpoint):
    train_losses = []
    for epoch in range(num_epochs):
        print('=====================================================================\n')
        loss = train(model, train_loader, loss_function, optimizer, device, epoch+1)
        train_losses.append(loss)
        scheduler.step()
        visualize_progress(model, train_loader, device)
        
        if epoch % checkpoint == 0:
            model_file = f'./checkpoints/STR_model_{epoch}_{int(loss)}.pth'
            torch.save(model.state_dict(), model_file) 
    
    return train_losses

# Util functions

def predict(model: STR_Model, input: torch.Tensor, device: torch.device) -> torch.Tensor:
    ''' Predicts the output sequence for the input sequence.'''
    model.eval()
    with torch.no_grad():
        output = model(input.to(device))
    return output

def get_strokes_from_model_output(pred) -> list:
    ''' Get strokes from the model output.'''
    print(pred.shape)
    pred = pred.cpu().detach().numpy()
    pred[:, 2] = np.round(pred[:, 2])
    pred[:, 3] = np.round(pred[:, 3])
    
    return pred

def display_img(img):
    plt.imshow(img.squeeze(0).permute(1, 2, 0), cmap='gray')
    plt.show()

def visualize_progress(model, train_loader, device):
    '''Display the input image and the predicted strokes.'''
    img, stroke = next(iter(train_loader))
    idx = np.random.randint(img.shape[0])
    img = img[idx, :, :, :]
    stroke = stroke[idx, :, :]
    
    # Predict the output sequence
    pred = predict(model, img.unsqueeze(0), device)
    pred = get_strokes_from_model_output(pred.squeeze(1))
    
    display_img(img)
    plot_word_strokes(pred, split_strokes=False)
    plot_str_word_strokes(pred, split_strokes=False)
    animate_word(pred, speed=1, save_path='./predict.gif', split_strokes=False)
    
    distance, path = fastdtw(pred[:,:2], stroke[:,:2], dist=2)
    print(f'Loss: {distance}')
    plot_dtw_path(pred, stroke, path)
    

def plot_losses(losses):
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.show()
    
def set_best_model(model, checkpoint_dir):
    ''' Set the model with least loss as the best model. '''
    best_loss = 100000
    best_model = None
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pth'):
            loss = int(file.split('_')[-1].split('.')[0])
            if loss < best_loss:
                best_loss = loss
                best_model = file
    if best_model is not None:
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, best_model)))
        print(f'Best model: {best_model}')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    # Training parameters
    num_epochs = 100
    bath_size = 16 # Archibald it is 32
    checkpoint_interval = 1
    learning_rate = 0.0001
    lr_decay = 0.99
    
    # Load data
    root_dir = '../../DataSet/IAM-Online/Resized_Dataset/Train'
    dataset = HandwritingDataset(root_dir, bath_size, device)
    dataloader = DataLoader(dataset, batch_size=bath_size, shuffle=False, drop_last=True)
    
    # Model
    model = STR_Model().to(device)
    set_best_model(model, './checkpoints/')
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    
    # Loss function
    loss_function = STR_Loss(sos_weight=20)

    # Fitting the model
    losses = model_fit(model, dataloader, loss_function, optimizer, scheduler, num_epochs, device, checkpoint_interval)
    
    # Plot losses
    plot_losses(losses)

if __name__ == "__main__":
    main()