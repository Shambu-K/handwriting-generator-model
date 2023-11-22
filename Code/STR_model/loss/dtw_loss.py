import torch
import torch.nn as nn

import torch

def loss_fn(input_seq, target_seq):
    '''Loss Function for DTW Loss'''
    # Input sequence is of the form n x 5, where n is the number of points in the sequence and 5 is the dimension of each point representing x, y, time, start_of_stroke (binary), end_of_stroke (binary)
    # Target sequence is of the form m x 5, where m is the number of points in the sequence and 5 is the dimension of each point representing x, y, time, start_of_stroke (binary), end_of_stroke (binary)
    # n and m can be different

    n = input_seq.shape[0]
    m = target_seq.shape[0]

    # Create a matrix to store the accumulated distances
    dtw_matrix = torch.zeros((n + 1, m + 1))

    # Initialize the first row and column of the matrix
    dtw_matrix[0, 1:] = float('inf')
    dtw_matrix[1:, 0] = float('inf')

    # Calculate the accumulated distances
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = torch.dist(input_seq[i-1, :2], target_seq[j-1, :2])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])

    # Calculate the DTW loss as the last element in the matrix
    dtw_loss = dtw_matrix[-1, -1]

    return dtw_loss

# Test the loss function
input_seq = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
target_seq = torch.tensor([[2, 3, 5], [6, 7, 8]], dtype=torch.float32)
loss = loss_fn(input_seq, target_seq)
print('Input sequence: ', input_seq)
print('Target sequence: ', target_seq)
print('DTW loss: ', loss)