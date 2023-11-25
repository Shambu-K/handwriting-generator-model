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

def visualize_progress(model, device, path):
    '''Display the input image and the predicted strokes.'''
    num_images = 40000
    img_id = np.random.randint(0, num_images)
    print(f'Image id: {img_id}')
    img_path = f'{path}/Images/image_{img_id}.png'
    stroke_path = f'{path}/Strokes/stroke_{img_id}.npy'
    stroke = np.load(stroke_path)
    stroke = np.delete(stroke, 2, 1)

    # Load the image
    img = Image.open(img_path).convert('L')
    img = ToTensor()(img)
    img.unsqueeze_(0)
    
    # Predict the output sequence
    pred = predict(model, img, device)
    pred = get_strokes_from_model_output(pred.squeeze(1))
    distance, path = fastdtw(pred[:,:2], stroke[:,:2], dist=2)
    
    display_img(img.cpu().detach().numpy().squeeze(0), title='Sample image')
    # plot_word_strokes(stroke, title='Ground truth strokes', split_strokes=True)
    # plot_word_strokes(pred, title='Predicted strokes', split_strokes=False)
    plot_str_word_strokes(pred, title='Predicted strokes with directions', split_strokes=False)
    animate_word(pred, speed=1, save_path=f'./predict_{img_id}.gif', title='Animated predicted strokes', split_strokes=False)
    plot_dtw_path(pred, stroke, path, title='DTW Warping Path')
    
    
def plot_losses(losses):
    plt.plot(losses)
    plt.show()
    

def main():
    import sys
    sys.path.append('../')
    from model import STR_Model
    
    # Load the model
    model_path = '../checkpoints/STR_model_0_8465.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STR_Model().to(device)
    model.load_state_dict(torch.load(model_path))
    
    data_path = '../../../DataSet/IAM-Online/Resized_Dataset/Train/'
    visualize_progress(model, device, data_path)
    
if __name__ == '__main__':
    main()
# %%
