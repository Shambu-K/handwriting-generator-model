# %%
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import sys
# sys.path.append('')
from .gt_resampling import resample_strokes
from tqdm import tqdm

class HandwritingDataset(Dataset):
    ''' Dataset class for the handwriting dataset. Loads the data into memory and preprocesses them based on expected batch size.'''
    def __init__(self, root_dir, batch_size=5, transform=ToTensor()):
        self.transform = transform
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'Images')
        self.stroke_dir = os.path.join(root_dir, 'Strokes')
        self.image_filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        self.stroke_filenames = sorted([f for f in os.listdir(self.stroke_dir) if f.endswith('.npy')])
        
        self.load_data()
        self.preprocess_data()
    
    def load_data(self):
        ''' Load the images and strokes from the disk and store them in memory. '''
        self.images = []
        self.strokes = []
        assert len(self.image_filenames) == len(self.stroke_filenames), 'Number of images and strokes do not match'
        for image_name, stroke_name in tqdm(zip(self.image_filenames, self.stroke_filenames), desc='Loading data'):
            image = Image.open(os.path.join(self.image_dir, image_name)).convert('L')
            image = self.transform(image)
            stroke_data = np.load(os.path.join(self.stroke_dir, stroke_name))
            stroke_data = np.delete(stroke_data, 2, axis=1) # Remove the third column (time)
            self.images.append(image)
            self.strokes.append(stroke_data)
        
    def preprocess_data(self):
        ''' Make the data more batch friendly. (Pads the images and resamples the strokes to be proportional to the image width)'''
        # Sort the images and strokes by width
        self.images, self.strokes = zip(*sorted(zip(self.images, self.strokes), key=lambda x: x[0].shape[2]))
        self.images = list(self.images)
        self.strokes = list(self.strokes)
        
        # iterate over the batches
        for i in tqdm(range(0, len(self), self.batch_size), desc='Preprocessing data'):
            images = self.images[i:i+self.batch_size]
            strokes = self.strokes[i:i+self.batch_size]
            
            max_width = max([image.shape[2] for image in images])
            max_width += (max_width % 2) # Makes width even since model requires even width
            for j, image in enumerate(images):
                assert isinstance(image, torch.Tensor), 'Image is not a tensor'
                assert image.shape[0] == 1 and image.shape[1] == 60, f'Image shape is wrong: {image.shape}'
                self.images[i+j] = torch.nn.functional.pad(image, (0, max_width-image.shape[2]), mode='constant', value=0)
            for j, stroke in enumerate(strokes):
                assert stroke.shape[1] == 4, 'Stroke shape is wrong'
                self.strokes[i+j] = resample_strokes(stroke, max_width, num_EoS_extra=5)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        stroke_data = self.strokes[idx]
        return image, stroke_data

# %%
def test_dataset():
    root_dir = '../../../DataSet/IAM-Online/Resized_Dataset/Train'
    batch_size = 5
    dataset = HandwritingDataset(root_dir, batch_size)
    print(len(dataset))
    
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False) # Shuffle false since images are sorted by width
    for image, stroke in train_loader:
        print(image.shape, stroke.shape)
    
if __name__ == '__main__':
    test_dataset()