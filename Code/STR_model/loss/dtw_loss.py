import torch
import torch.nn as nn
import random
import os
import numpy as np

dir_path = '../../../DataSet/IAM-Online/Resized_Dataset/Train/Images/'
num_files = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
print(num_files)

def loss_fn(input_seq, target_seq):
    '''Loss Function for DTW Loss'''
    ''' Input sequence and target sequence are of the form n x 5, where n is the number of points in the sequence and 5 is the dimension of each point representing x, y, time, start_of_stroke (binary), end_of_stroke (binary)'''

    n = input_seq.shape[0]
    m = target_seq.shape[0]

    # Create a matrix to store the accumulated distances
    dtw_matrix = torch.zeros((n + 1, m + 1))

    # Initialize the first row and column of the matrix
    dtw_matrix[0, 1:] = float('inf')
    dtw_matrix[1:, 0] = float('inf')

    # Create a matrix to store the optimal warping path
    path_matrix = torch.zeros((n + 1, m + 1), dtype=torch.int)

    # Calculate the accumulated distances and optimal warping path
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = torch.dist(input_seq[i-1, :2], target_seq[j-1, :2])
            min_cost = min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
            dtw_matrix[i, j] = cost + min_cost

            # Update the path matrix based on the minimum cost
            if min_cost == dtw_matrix[i-1, j]:
                path_matrix[i, j] = 1  # Vertical movement
            elif min_cost == dtw_matrix[i, j-1]:
                path_matrix[i, j] = 2  # Horizontal movement
            else:
                path_matrix[i, j] = 3  # Diagonal movement

    # Calculate the DTW loss as the last element in the matrix
    dtw_loss = dtw_matrix[-1, -1]

    # Compute the optimal warping path
    i, j = n, m
    warping_path = [(i, j)]
    while i > 1 or j > 1:
        if path_matrix[i, j] == 1:
            i -= 1  # Vertical movement
        elif path_matrix[i, j] == 2:
            j -= 1  # Horizontal movement
        else:
            i -= 1  # Diagonal movement
            j -= 1
        warping_path.append((i, j))

    warping_path.reverse()

    # Perform backward propagation to compute gradients
    # dtw_loss.backward()

    # Retrieve the gradients
    gradients = input_seq.grad

    return dtw_loss, warping_path, gradients

# Test the loss function
# function to get random images from the dataset
img_num1 = random.randint(1, num_files + 1)
stroke_path = '../../../DataSet/IAM-Online/Resized_Dataset/Train/Strokes/' + f'stroke_{img_num1}.npy'
stroke = np.load(stroke_path)
input_seq = torch.from_numpy(stroke).float()
#ADD an offset of constant value to x and y in input_seq
inp_seq = input_seq 
input_seq[:, 0] += 10
input_seq[:, 1] += 10

img_num2 = random.randint(1, num_files + 1)
stroke_path = '../../../DataSet/IAM-Online/Resized_Dataset/Train/Strokes/' + f'stroke_{img_num2}.npy'
stroke = np.load(stroke_path)
target_seq = torch.from_numpy(stroke).float()
print(input_seq.shape)
print(target_seq.shape)


# Compute DTW loss, optimal warping path, and gradients
loss, path, gradients = loss_fn(input_seq, inp_seq)
#plot input_seq(only x and y coordinates)
import matplotlib.pyplot as plt
plt.plot(input_seq[:, 0], input_seq[:, 1])
plt.show()

#plot inp_seq(only x and y coordinates)
plt.plot(inp_seq[:, 0], inp_seq[:, 1])
plt.show()
print('DTW loss:', loss)
print('Optimal Warping Path:', path)
print('Gradients:', gradients)