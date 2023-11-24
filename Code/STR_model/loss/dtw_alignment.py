'''
Contains our implementation of the DTW loss using dynamic programming to find the dtw_cost_matrix and the warping_path_matrix. 
Time complexity of this implementation is O(n*m), where n and m are the lengths of the input and target sequences respectively.
Since we are using python loops, this implementation is very slow and doesnt take full advantage of possible vectorizations in the implementation.
Hence we use fastdtw (https://cs.fit.edu/~pkc/papers/tdm04.pdf) which works in O(n) time and is implemented in the fastdtw package.
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
import time

def dtw_path(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ''' Computes the DTW cost between the (unbatched) input (n x 2) and target sequences (m x 2) and 
        returns the optimal warping path between the sequences (the mapping between the sequences) '''
    n = pred.shape[0]
    m = target.shape[0]
    
    # Initialize the cumulative cost matrix and the warping path matrix
    dtw_cost_matrix = torch.zeros((n+1, m+1))
    warping_path_matrix = torch.zeros((n+1, m+1), dtype=torch.int32)
    
    # Initialize the first row and column of the cumulative cost matrix
    dtw_cost_matrix[0, 1:] = torch.tensor(np.inf)
    dtw_cost_matrix[1:, 0] = torch.tensor(np.inf)
    
    # Compute the cumulative cost matrix and the warping path matrix
    cost = torch.cdist(pred, target, p=2) # Pairwise l2 distance between the input and target sequences
    for i in range(1, n+1):
        for j in range(1, m+1):
            min_cost = torch.min(torch.stack([dtw_cost_matrix[i-1, j], dtw_cost_matrix[i, j-1], dtw_cost_matrix[i-1, j-1]]))
            dtw_cost_matrix[i, j] = cost[i-1, j-1] + min_cost
            if min_cost == dtw_cost_matrix[i-1, j]: warping_path_matrix[i, j] = 1
            elif min_cost == dtw_cost_matrix[i, j-1]: warping_path_matrix[i, j] = 2
            else: warping_path_matrix[i, j] = 3
    
    # The loss is the last element of the cumulative cost matrix (non differentiable)
    loss = dtw_cost_matrix[-1, -1]
    
    # Compute the warping path
    warping_path = []
    i, j = n, m
    while i > 0 or j > 0:
        warping_path.append((i-1, j-1))
        if warping_path_matrix[i, j] == 1: i -= 1 # Vertical move
        elif warping_path_matrix[i, j] == 2: j -= 1 # Horizontal move
        else: i -= 1; j -= 1 # Diaognal move
    warping_path.reverse()
    
    return warping_path, loss

def batched_fastdtw_paths(pred_batch: torch.Tensor, target_batch: torch.Tensor) -> list:
    ''' Computes the DTW alignment between the input (batch_size x n x 2) and target sequences (batch_size x m x 2) and 
        returns the optimal warping path between the sequences '''
    warping_paths = []
    for (pred, target) in zip(pred_batch, target_batch):
        warping_paths.append(fastdtw(pred.cpu().detach(), target.cpu().detach(), dist=2)[1])
    return warping_paths

def plot_dtw_path(pred: torch.Tensor, target: torch.Tensor, warping_path: torch.Tensor):
    # Plot the input and target sequences
    plt.scatter(pred[:, 0], pred[:, 1], c='r', label='Pred')
    plt.scatter(target[:, 0], target[:, 1], c='b', label='Target')
    plt.legend()
    
    # Plot the warping path
    for i in range(len(warping_path)):
        plt.plot([pred[warping_path[i][0], 0], target[warping_path[i][1], 0]], [pred[warping_path[i][0], 1], target[warping_path[i][1], 1]], c='k')
        
    plt.show()
    
def animate_dtw_path(pred: torch.Tensor, target: torch.Tensor, warping_path: torch.Tensor, save_path: str = None):
    import matplotlib.animation as animation
    from matplotlib import style
    style.use('fivethirtyeight')
    
    # Plot the input and target sequences
    plt.scatter(pred[:, 0], pred[:, 1], c='r', label='Pred')
    plt.scatter(target[:, 0], target[:, 1], c='b', label='Target')
    
    def animate(i):
        plt.plot([pred[warping_path[i][0], 0], target[warping_path[i][1], 0]], [pred[warping_path[i][0], 1], target[warping_path[i][1], 1]], c='k')
        
    ani = animation.FuncAnimation(plt.gcf(), animate, interval=100, frames=len(warping_path))
    if save_path is not None: ani.save(save_path, writer='pillow')
    plt.show()


def test_dtw_warping_path():
    start = time.time()
    target_file = '../../../DataSet/IAM-Online/Resized_Dataset/Train/Strokes/stroke_42.npy'
    target = np.load(target_file)
    pred = target + 2
    
    target = torch.tensor(target)
    pred = torch.tensor(pred)
    
    print(f'(DTW)\nInput Shape: {pred.shape}')
    
    # Compute the warping path and the loss
    warping_path, loss = dtw_path(pred[:,:2], target[:,:2])
    print(f'Loss: {loss}')
    print(f'Num of mappings: {len(warping_path)}')
    print(f'Time taken (DTW): {time.time() - start}\n')
    
    plot_dtw_path(pred, target, warping_path)
    # animate_dtw_path(pred, target, warping_path, save_path='../images/dtw_warping_path.gif')
    
def test_fastdtw():
    start = time.time()
    target_file = '../../../DataSet/IAM-Online/Resized_Dataset/Train/Strokes/stroke_42.npy'
    target = np.load(target_file)
    pred = target + 10
    
    target = torch.tensor(target)
    pred = torch.tensor(pred)
    
    print(f'(FastDTW):\nInput Shape: {pred.shape}')
    
    # Compute the warping path and the loss
    loss, warping_path = fastdtw(pred[:,:2], target[:,:2], dist=2, radius=1) # radius = len(x) will give exact dtw loss as calculated using dynamic programming
    print(f'Loss: {loss}')
    print(f'Num of mappings: {len(warping_path)}')
    print(f'Time taken: {time.time() - start}')

    plot_dtw_path(pred, target, warping_path)
    # animate_dtw_path(pred, target, warping_path, save_path='../images/fastdtw_warping_path.gif')

def test_resampled_strokes():
    import sys
    sys.path.append('../')
    from dataset.gt_resampling import resample_strokes
    
    target_file = '../../../DataSet/IAM-Online/Resized_Dataset/Train/Strokes/stroke_2.npy'
    target = np.load(target_file)
    target = np.delete(target, 2, axis=1) # Remove the third column (time)
    pred = resample_strokes(target, len(target)*2)
    pred[:, 0] += 4
    pred[:, 1] += 2
    
    target = torch.tensor(target)
    pred = torch.tensor(pred)
    
    print(f'(FastDTW on Resampled strokes):\nInput Shape: {pred.shape}')
    print(f'Target Shape: {target.shape}')
    
    # Compute the warping path and the loss
    loss, warping_path = fastdtw(pred[:,:2], target[:,:2], dist=2, radius=1) # radius = len(x) will give exact dtw loss as calculated using dynamic programming
    print(f'Loss: {loss}')
    print(f'Num of mappings: {len(warping_path)}')
    
    plot_dtw_path(pred, target, warping_path)
    # animate_dtw_path(pred, target, warping_path, save_path='../images/fastdtw_warping_path.gif')

if __name__ == '__main__':
    test_dtw_warping_path()
    test_fastdtw()
    test_resampled_strokes()