"""Object for the binary and multiclass Dice Similarity Coefficient (DSC) training loss."""

import torch
import torch.nn as nn

class DSCLoss(nn.Module):
    def __init__(self, multiclass=False, num_classes=None):
        super(DSCLoss, self).__init__()
        """ Class for the binary and multiclass Dice Similarity Coefficient (DSC) training loss. 
        
        Args:
            multiclass (str): whether the segmentation is binary of multiclass
            num_classes (int): number of the segmentaion labels 
        """
        self.multiclass = multiclass
        self.num_classes = num_classes

    def forward(self, inputs, targets, smooth=1.):
        """Returns an int (1 - dice) corresponding to the DSC loss."""

        if self.multiclass: 
            dice = 0.0

            # DSC computation for each class 
            for class_id in range(1, self.num_classes):
                y_pred_class = (inputs == class_id).float()
                y_true_class = (targets == class_id).float()

                intersection = torch.sum(y_pred_class * y_true_class)
                union = torch.sum(y_pred_class) + torch.sum(y_true_class)
                class_dice = (2.0 * intersection + smooth) / (union + smooth)

                # DSC weighting with respect to the class imbalance
                class_tot = torch.sum(targets == 1) + torch.sum(targets == 2)
                class_cur = torch.sum(targets == class_id)
                weight = class_tot / (class_cur * (self.num_classes-1))
                dice += weight * class_dice

            dice = (dice / self.num_classes)
        else: 
            # Flatten label and prediction tensors
            inputs = inputs.contiguous().view(-1)
            targets = targets.contiguous().view(-1)
            # DSC computation
            intersection = (inputs * targets).sum()  
                                      
            dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice