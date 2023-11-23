# %%
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

root_dir = '/content/'
image_dir = os.path.join(root_dir, 'Images')
stroke_dir = os.path.join(root_dir, 'Strokes')
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
stroke_files = [f for f in os.listdir(stroke_dir) if f.endswith('.npy')]
image_files = sorted(image_files)
stroke_files = sorted(stroke_files)

images = []
strokes = []
for i in image_files:
    image_idx = int(i[6:][:-4])
    img_name = os.path.join(image_dir, image_files[image_idx-1])
    stroke_name = os.path.join(stroke_dir, stroke_files[image_idx-1])
    image = Image.open(img_name).convert('L')
    stroke_data = np.load(stroke_name)
    stroke_data = np.delete(stroke_data, 2, axis=1)
    images.append(image)
    strokes.append(stroke_data)

class HandwritingDataset(Dataset):
    def __init__(self, image_files, stroke_files, transform=None):
        self.transform = transform
        self.image_files = image_files
        self.stroke_files = stroke_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = self.image_files[idx]
        stroke_data = self.stroke_files[idx]

        if self.transform:
            image = self.transform(image)

        return (image, stroke_data, image.shape[2])

# Define the transformation for your images (resizing, normalization, etc.)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create the dataset
data_path = '/content/' # The path should point to the Train directory (the directly available subfolders should be Images/ and Strokes/)
dataset = HandwritingDataset(image_files=images, stroke_files=strokes, transform=transform)

# Calculate the split indices for train and test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

def resample(stroke, max_width):
    return torch.zeros((max_width,4))

def collate_fn(data):
    images, strokes, widths = zip(*data) # (image, stroke, width)
    max_width = max(widths)
    max_width += (max_width % 2) # Makes width even
    height = data[0][0].shape[1]
    img_features = torch.zeros((len(data), height, max_width))

    height_stroke = data[0][1].shape[1]
    stroke_features_new = []
    for stroke_idx in range(len(data)):
        stroke = resample(data[stroke_idx][1], max_width)
        stroke_features_new.append(stroke)
        print(stroke.shape)
    stroke_features = stroke_features_new

    widths = torch.tensor(widths)

    return img_features.float(), stroke_features, widths.long()

# Create dataloaders
train_loader = DataLoader(dataset=train_dataset, collate_fn=collate_fn, batch_size=4, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False)

# Check the number of batches in the train and test loaders
print(len(train_loader), len(test_loader))

for (img, stroke, len) in train_loader:
    print(len)