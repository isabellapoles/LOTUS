"""Objects to create the UNet CNN architecture on which student and teacher model are built."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- Utilities for the UNet object. ----------------------------- #

def conv_bn_leru(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    """Utility function to make in&out matrix same size when filter kernel= 3x3, padding=1 

        Args:
            in_channels (int): number of input feature map channels
            out_channels (int): number of output feature map channels
            kernel_size (int): size of the kernel
            padding (int): amount of padding
        
        Returns:
            module: Conv2d+BatchNorm+ReLU sequential
    """
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
    )


def down_pooling():
    """Utility function to perform the down scaling operation 
        
        Returns:
            module: MaxPool2d 
    """
    return nn.MaxPool2d(2)


def up_pooling(in_channels, out_channels, kernel_size=2, stride=2):
    """Utility function to perform the up scaling operation 

        Args:
            in_channels (int): number of input feature map channels
            out_channels (int): number of output feature map channels
            kernel_size (int): size of the kernel
            stride (int): amount of stride
        
        Returns:
            module: Conv2dTranspose+BatchNorm+ReLU sequential
    """
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    """Object to create the UNet CNN architecture on which student and teacher model are built."""

    def __init__(self, input_channels, nclasses):
        super().__init__()
        """Class for the UNet CNN.

        Args:
            in_channels (int): number of input feature map channels
            nclasses (int): number of classes to output
        """

        self.nclasses = nclasses
        # Encoding layers
        self.conv1 = conv_bn_leru(input_channels,16)
        self.conv2 = conv_bn_leru(16, 32)
        self.conv3 = conv_bn_leru(32, 64)
        self.conv4 = conv_bn_leru(64, 128)
        self.conv5 = conv_bn_leru(128, 256)
        self.down_pooling = nn.MaxPool2d(2)

        # Decoding layers
        self.up_pool6 = up_pooling(256, 128)
        self.conv6 = conv_bn_leru(256, 128)
        self.up_pool7 = up_pooling(128, 64)
        self.conv7 = conv_bn_leru(128, 64)
        self.up_pool8 = up_pooling(64, 32)
        self.conv8 = conv_bn_leru(64, 32)
        self.up_pool9 = up_pooling(32, 16)
        self.conv9 = conv_bn_leru(32, 16)

        self.conv10 = nn.Conv2d(16, self.nclasses, 1)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        """ Returns a tuple of the probaility map and the lasts 2 features maps (output, x10, x9)"""

        # Encode
        x1 = self.conv1(x)
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)
        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)
        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)
        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)

        # Decode
        p6 = self.up_pool6(x5)
        x6 = torch.cat([p6, x4], dim=1)
        x6 = self.conv6(x6)

        p7 = self.up_pool7(x6)
        x7 = torch.cat([p7, x3], dim=1)
        x7 = self.conv7(x7)

        p8 = self.up_pool8(x7)
        x8 = torch.cat([p8, x2], dim=1)
        x8 = self.conv8(x8)

        p9 = self.up_pool9(x8)
        x9 = torch.cat([p9, x1], dim=1)
        x9 = self.conv9(x9)

        x10 = self.conv10(x9)

        if self.nclasses == 1: 
            output = torch.sigmoid(x10)
        else: 
            output = self.softmax(x10)

        return output, x10, x9


class MLP(nn.Module):
    """Object to create the Multi-Layer Perceptron architecture to aligned feature embeddings from different modalities."""
    
    def __init__(self, input_dim=16, feat_dim=16):
        super().__init__()
        """Class for the MLP.

        Args:
            input_dim (int): number of input feature map channels
            feat_dim (int): number of output feature map channels

        """

        layer = []
        layer.append(nn.Linear(input_dim, 2*feat_dim))
        layer.append(nn.ReLU(inplace=True))
        layer.append(nn.Linear(2*feat_dim, feat_dim))
        self.layer = nn.Sequential(*layer)
    
    def forward(self,x):
        """ Returns the projected tensor feature maps."""

        x = self.layer(x)
        x = F.normalize(x, p=2.0, dim=1)

        return x