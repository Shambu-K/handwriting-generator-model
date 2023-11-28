# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor
from fastdtw import fastdtw
import sys
sys.path.append('../')
from util.stroke_plotting import get_strokes, plot_word_strokes, plot_str_word_strokes, animate_word
from loss.dtw_alignment import plot_dtw_path

def predict(model, input: torch.Tensor, device: torch.device) -> torch.Tensor:
    ''' Predicts the output sequence for the input sequence.'''
    model.eval()
    with torch.no_grad():
        output = model(input.to(device))
    return output

def get_strokes_from_model_output(pred) -> list:
    ''' Get strokes from the model output.'''
    pred = pred.cpu().detach().numpy()
    pred[:, -2] = np.round(pred[:, -2])
    pred[:, -1] = np.round(pred[:, -1])
    
    return pred

def display_img(img, title):
    plt.figure(figsize=(2, 2))
    plt.title(title)
    plt.imshow(img[0], cmap='gray')
    plt.show()

def visualize_progress(model, device, dataloader, epoch=0):
    '''Display the input image and the predicted strokes.'''
    batch_id = np.random.randint(0, len(dataloader))
    for i, (img, stroke) in enumerate(dataloader):
        if i == batch_id:
            break
    
    # Predict the output sequence
    pred = predict(model, img, device)
    pred = get_strokes_from_model_output(pred.squeeze(0))
    
    img = img[0].cpu().detach().numpy()
    stroke = stroke[0].cpu().detach().numpy()
    distance, path = fastdtw(pred[:,:2], stroke[:,:2], dist=2)
    
    display_img(img, title='Sample image')
    # plot_word_strokes(stroke, title='Ground truth strokes', split_strokes=True)
    # plot_word_strokes(pred, title='Predicted strokes', split_strokes=False)
    plot_str_word_strokes(pred, title='Predicted strokes with directions', split_strokes=True)
    animate_word(pred, speed=1, save_path=f'./train_outputs/predict_{epoch}.gif', title='Animated predicted strokes', split_strokes=True)
    plot_dtw_path(pred, stroke, path, title=f'DTW Warping Path -> L2-distance = {distance}')
    
    
def plot_losses(losses):
    ''' Plot the losses (total, dtw, sos, eos)'''
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Training Losses', fontsize=16, fontweight='bold', y=0.93)
    axs[0, 0].plot(losses[0])
    axs[0, 0].set_title('Total Loss')
    axs[0, 0].legend([f'Final Total Loss = {losses[0][-1]:.4f}'])

    axs[0, 1].plot(losses[1])
    axs[0, 1].set_title('DTW Loss')
    axs[0, 1].legend([f'Final DTW Loss = {losses[1][-1]:.4f}'])
    axs[1, 0].plot(losses[2])
    axs[1, 0].set_title('SOS Loss')
    axs[1, 0].legend([f'Final SOS Loss = {losses[2][-1]:.4f}'])
    axs[1, 1].plot(losses[3])
    axs[1, 1].set_title('EOS Loss')
    axs[1, 1].legend([f'Final EOS Loss = {losses[3][-1]:.4f}'])
    plt.show()

    

def main():
    import sys
    sys.path.append('../')
    from model import STR_Model
    from dataset.iam_dataloader import HandwritingDataset
    from torch.utils.data import DataLoader
    
    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STR_Model().to(device)
    # model_path = '../checkpoints/STR_model_0_8465.pth'
    # model.load_state_dict(torch.load(model_path))
    
    data_path = '../../../DataSet/IAM-Online/Resized_Dataset/Train'
    dataset = HandwritingDataset(data_path, 1, device, max_allowed_width=400)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)
    visualize_progress(model, device, dataloader)

if __name__ == '__main__':
    main()