import torch.nn as nn

import torch
import torch.nn as nn

def loss_fn(input_seq, target_seq):
    # Get the dimensions of the input and target sequences
    a, b = input_seq.size()
    c, d = target_seq.size()
    
    # Create a matrix to store the accumulated distances
    dtw_matrix = torch.zeros((b + 1, d + 1))
    
    # Initialize the first row and column of the matrix
    dtw_matrix[0, 1:] = float('inf')
    dtw_matrix[1:, 0] = float('inf')
    
    # Calculate the accumulated distances
    for i in range(1, b + 1):
        for j in range(1, d + 1):
            cost = torch.dist(input_seq[:, i-1], target_seq[:, j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
    
    # Calculate the DTW loss as the last element in the matrix
    dtw_loss = dtw_matrix[-1, -1]
    
    return dtw_loss