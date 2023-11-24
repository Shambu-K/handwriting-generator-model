''' Main loss function for the STR model. We first get the DWT alignment between the input and target sequences and then get the losses for the coordinated and SoS and EoS tokens using this alignment. '''
import torch
import torch.nn as nn
from .dtw_alignment import batched_fastdtw_paths
from .cross_entropy_loss import SoS_Loss, EoS_Loss
from .dtw_loss import DTW_Loss

class STR_Loss(nn.Module):
    def __init__(self, sos_weight: float = 5.0):
        super().__init__()
        self.sos_weight = sos_weight
        self.dtw_loss = DTW_Loss()
        self.sos_loss = SoS_Loss(weight=self.sos_weight)
        self.eos_loss = EoS_Loss()
        
    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        ''' pred: (batch_size, seq_len, 4)
            target: (batch_size, seq_len, 4)'''
        # Get the DTW alignment between the input and target sequences
        paths = batched_fastdtw_paths(preds[:, :, :2], targets[:, :, :2])
        
        # Compute the losses for the different output features
        dtw_loss = self.dtw_loss(preds[:, :, :2], targets[:, :, :2], paths)
        sos_loss = self.sos_loss(preds[:, :, 2], targets[:, :, 2], paths)
        eos_loss = self.eos_loss(preds[:, :, 3], targets[:, :, 3], paths)
        
        return dtw_loss + sos_loss + eos_loss