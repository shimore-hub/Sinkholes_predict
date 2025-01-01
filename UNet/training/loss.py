
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class old_DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(old_DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5):

        # Assuming raw logits from the model, apply sigmoid if necessary
        inputs = torch.sigmoid(inputs)

        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Ensure numerical stability
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        # BCE loss
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        # Combined loss
        Dice_BCE = BCE + dice_loss

        # Debugging checks
        assert torch.min(targets) >= 0 and torch.max(targets) <= 1, f"Targets must be in [0, 1]. Got Min: {torch.min(targets).item()}, Max: {torch.max(targets).item()}"


        return Dice_BCE


import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5):

        # Use binary_cross_entropy_with_logits to combine sigmoid and BCE loss
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')

        # Flatten tensors for dice calculation
        inputs = torch.sigmoid(inputs)  # Apply sigmoid to logits for dice calculation
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Ensure numerical stability for dice coefficient calculation
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        # Combined loss
        Dice_BCE = BCE + dice_loss

        # Debugging checks
        assert torch.min(targets) >= 0 and torch.max(targets) <= 1, f"Targets must be in [0, 1]. Got Min: {torch.min(targets).item()}, Max: {torch.max(targets).item()}"

        return Dice_BCE


