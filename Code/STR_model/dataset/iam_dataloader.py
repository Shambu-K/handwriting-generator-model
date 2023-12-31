# %%
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import sys
sys.path.append('../')
from dataset.gt_resampling import resample_strokes
from tqdm import tqdm

class HandwritingDataset(Dataset):
    ''' Dataset class for the handwriting dataset. Loads the data into memory and preprocesses them based on expected batch size.'''
    def __init__(self, root_dir, batch_size=5, transform=ToTensor(), max_allowed_width=300):
        self.transform = transform
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.max_allowed_width = max_allowed_width
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
        for image_name, stroke_name in tqdm(zip(self.image_filenames, self.stroke_filenames), desc='Loading data', total=len(self.image_filenames)):
            image = Image.open(os.path.join(self.image_dir, image_name)).convert('L')
            image = self.transform(image)
            stroke_data = np.load(os.path.join(self.stroke_dir, stroke_name))
            stroke_data = np.delete(stroke_data, 2, axis=1) # Remove the third column (time)
            if image.shape[2] > self.max_allowed_width: continue
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
                self.strokes[i+j] = torch.tensor(resample_strokes(stroke, max_width, num_EoS_extra=5))
                
        
        # Group the images and strokes into batches
        self.image_batches = []
        self.stroke_batches = []
        for i in range(0, len(self), self.batch_size):
            self.image_batches.append(torch.stack(self.images[i:i+self.batch_size]))
            self.stroke_batches.append(torch.stack(self.strokes[i:i+self.batch_size]))
        self.images = self.image_batches
        self.strokes = self.stroke_batches
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_batch = self.images[idx]
        strokes_batch = self.strokes[idx]
        return image_batch, strokes_batch

# %%
def test_dataset():
    root_dir = '../../../DataSet/IAM-Online/Resized_Dataset/Train'
    batch_size = 16
    dataset = HandwritingDataset(root_dir, batch_size)
    print(len(dataset))
    
    train_loader = DataLoader(dataset=dataset, shuffle=True)
    for batch, (image, stroke_data) in enumerate(train_loader):
        # if batch % 100 == 0:
            image, stroke_data = image.squeeze(0), stroke_data.squeeze(0)
            print(f'Batch {batch:<4}: Shape {image.shape}, {stroke_data.shape}')
    
if __name__ == '__main__':
    test_dataset()