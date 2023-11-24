''' We use the dtw alignment and L1 norm to compute the similarity between the input and target sequences. '''
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastdtw import fastdtw

class DTW_Loss(nn.Module):
    '''Given mapping f: {1...n} -> {1...m}, computes loss as L_dtw = \sum_{i=1}^{n} ||x_i - y_{f(i)}||_1 '''
    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        ''' pred: (batch_size, seq_len, 2)
            target: (batch_size, seq_len, 2)'''
        loss = torch.tensor(0.0, requires_grad=True)
        for batch, (pred_seq, target_seq) in enumerate(zip(pred, target)):
            non_diff_loss, warping_path = fastdtw(pred_seq.cpu().detach().numpy(), target_seq.cpu().detach().numpy(), dist=1)
            
            # Compute the L1 norm loss using the warping path (mapping)    
            for i, j in warping_path:
                loss += torch.norm(pred_seq[i] - target_seq[j], p=1)
        
        return loss / pred.shape[0]