''' Given the DTW alignment, we compute the loss for the SoS and EoS tokens using this alignment and cross entropy loss. '''
import torch
import torch.nn as nn

class SoS_Loss(nn.Module):
    ''' Gives weighted cross entropy loss for the SoS token '''
    def __init__(self, weight: float = 5.0):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight = torch.tensor(weight, device=self.device)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=self.weight)
        # self.loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, self.weight], device=self.device))
        
    def forward(self, preds: torch.Tensor, targets: torch.Tensor, paths: list):
        ''' pred: (batch_size, seq_len)
            target: (batch_size, seq_len) '''
        loss = 0
        for batch, (pred, target, warping_path) in enumerate(zip(preds, targets, paths)):
            ground_truth = self.get_gt_for_pred(pred, target, warping_path)
            loss += self.loss(pred, ground_truth)
        
        return loss / preds.shape[0]
    
    def get_gt_for_pred(self, pred: torch.Tensor, target: torch.Tensor, warping_path: torch.Tensor):
        ''' For each SoS target token, we find the first predicted token that is aligned to it and use that as the 
            ground truth for the prediction '''
        gt = torch.zeros(pred.shape[0], device=self.device)
        target_SoS_done = set()
        for i, j in warping_path:
            if target[j] == 1 and j not in target_SoS_done:
                gt[i] = 1
                target_SoS_done.add(j)
        return gt
    

class EoS_Loss(nn.Module):
    ''' Gives cross entropy loss for the EoS token. The EoS tokens in the target have been duplicated to mitigate the 
        class imbalance issue (which is handled using weights for SoS) '''
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss = nn.BCEWithLogitsLoss()
        # self.loss = nn.CrossEntropyLoss()
        
    def forward(self, preds: torch.Tensor, targets: torch.Tensor, paths: list):
        ''' pred: (batch_size, seq_len)
            target: (batch_size, seq_len) '''
        loss = 0
        for batch, (pred, target, warping_path) in enumerate(zip(preds, targets, paths)):
            ground_truth = self.get_gt_for_pred(pred, target, warping_path)
            loss += self.loss(pred, ground_truth)
            
        return loss / preds.shape[0]
    
    def get_gt_for_pred(self, pred: torch.Tensor, target: torch.Tensor, warping_path: torch.Tensor):
        ''' All predictions that map to the EoS token in the target are considered as positive examples '''
        gt = torch.zeros(pred.shape[0], device=self.device)
        for i, j in warping_path:
            if target[j] == 1:
                gt[i] = 1
        return gt