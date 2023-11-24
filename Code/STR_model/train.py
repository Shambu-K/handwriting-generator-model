import torch
from tqdm import tqdm

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
        print('===================================\n')
        loss = train(model, train_loader, loss_function, optimizer, device, epoch+1)
        train_losses.append(loss)
        scheduler.step()
        
        if epoch % checkpoint == 0:
            model_file = f'./checkpoints/STR_model_{epoch}_{int(loss)}.pth'
            torch.save(model.state_dict(), model_file) 
    
    return train_losses

def plot_losses(losses):
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.show()

def main():
    from model import STR_Model
    from dataset.iam_dataloader import HandwritingDataset
    from loss.stroke_loss import STR_Loss
    from torch.optim import Adam, lr_scheduler
    from torch.utils.data import DataLoader
    
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
    # model.load_state_dict(torch.load('./checkpoints/STR_model_0_0.pth'))
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    
    # Loss function
    loss_function = STR_Loss(sos_weight=5)

    # Fitting the model
    losses = model_fit(model, dataloader, loss_function, optimizer, scheduler, num_epochs, device, checkpoint_interval)
    
    # Plot losses
    plot_losses(losses)

if __name__ == "__main__":
    main()