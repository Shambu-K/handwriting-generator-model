''' We use the dtw alignment and L1 norm to compute the similarity between the input and target sequences. '''
import torch
import torch.nn as nn

class DTW_Loss(nn.Module):
    '''Given mapping f: {1...n} -> {1...m}, computes loss as L_dtw = \sum_{i=1}^{n} ||x_i - y_{f(i)}||_1 '''
    def forward(self, preds: torch.Tensor, targets: torch.Tensor, paths: list):
        ''' preds: (batch_size, seq_len, 2)
            target: (batch_size, seq_len, 2)'''
        loss = 0
        for batch, (pred, target, warping_path) in enumerate(zip(preds, targets, paths)):
            # Compute the L1 norm loss using the warping path (mapping)    
            for i, j in warping_path:
                loss += torch.norm(pred[i] - target[j], p=1)
        
        return loss / (preds.shape[0] * preds.shape[1])