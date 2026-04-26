import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss

import torch.nn.functional as F

class CustomFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights normalized relative to mean
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        # ce_loss includes label smoothing and alpha weights
        ce_loss = F.cross_entropy(
            logits, targets, weight=self.alpha, 
            reduction='none', label_smoothing=self.label_smoothing
        )
        # Compute p_t without weights/smoothing for the gamma scaling (focal scaling)
        p_t = torch.exp(-F.cross_entropy(logits, targets, reduction='none'))
        loss = ((1 - p_t) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()

class HybridLoss(nn.Module):
    """
    Hybrid Loss = 0.5 * DiceLoss + 0.5 * FocalLoss
    Uses normalized inverse-frequency class weights and label smoothing.
    """
    def __init__(self, class_weights=None, focal_weight=0.5, dice_weight=0.5):
        super(HybridLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
        self.focal_loss = CustomFocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.05)
        self.dice_loss = DiceLoss(mode='multiclass')
        
    def forward(self, logits, targets):
        focal = self.focal_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.focal_weight * focal + self.dice_weight * dice

def compute_iou(preds, targets, num_classes):
    """
    Compute Intersection over Union (IoU) per class and mean IoU.
    preds: (B, H, W) integer targets
    targets: (B, H, W) integer targets
    """
    ious = []
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    # Ignore index 255 if using ignore_index, but we don't have it defined right now.
    
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = targets == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in mean IoU
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            
    # Calculate valid mean IoU discarding NaNs
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    
    return ious, mean_iou

import numpy as np
