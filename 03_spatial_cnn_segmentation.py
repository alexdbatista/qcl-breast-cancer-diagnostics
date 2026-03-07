"""
03_spatial_cnn_segmentation.py
==============================
PyTorch U-Net Architecture for Label-Free Hyperspectral Pathology Segmentation.

This module defines a Convolutional Neural Network (CNN) specifically tailored
for spatial segmentation of Breast Cancer Tissue Microarrays (TMAs) using 
Mid-IR QCL hyperspectral cubes.

Objective:
Map a 3D hyperspectral tensor (H, W, Spectral_Bands) to a 2D segmentation 
mask (H, W) where each pixel is classified into histological categories 
(e.g., Malignant Stroma, Benign Epithelium).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperspectralUNet(nn.Module):
    """
    A modified U-Net architecture adapted for highly dimensional 
    hyperspectral chemical imaging.
    """
    
    def __init__(self, in_channels, num_classes):
        """
        Parameters
        ----------
        in_channels : int
            Number of spectral bands (or PCA components) per pixel.
        num_classes : int
            Number of histological target classes (e.g., 4 for breast tissue).
        """
        super(HyperspectralUNet, self).__init__()
        
        # --- Encoder (Downsampling) ---
        # We use 1x1 convolutions initially to act as a learned dimensionality 
        # reduction layer across the spectral bands before spatial pooling.
        self.enc1 = self._conv_block(in_channels, 64, kernel_size=1, padding=0)
        self.enc2 = self._conv_block(64, 128, kernel_size=3, padding=1)
        self.enc3 = self._conv_block(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- Bottleneck ---
        self.bottleneck = self._conv_block(256, 512, kernel_size=3, padding=1)
        
        # --- Decoder (Upsampling) ---
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256, kernel_size=3, padding=1)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128, kernel_size=3, padding=1)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64, kernel_size=3, padding=1)
        
        # Final classification layer (1x1 conv to map to classes)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def _conv_block(self, in_c, out_c, kernel_size, padding):
        """Helper to create a standard Conv -> BatchNorm -> ReLU block."""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Forward pass through the U-Net.
        
        Parameters
        ----------
        x : torch.Tensor
            Input hyperspectral cube of shape (Batch, Bands, Height, Width).
            
        Returns
        -------
        logits : torch.Tensor
            Unnormalized class probabilities of shape (Batch, Classes, Height, Width).
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
        
        # Decoder (with skip connections)
        d3 = self.upconv3(b)
        
        # Ensure spatial dimensions match before concatenation (optional padding logic can go here)
        diffY = e3.size()[2] - d3.size()[2]
        diffX = e3.size()[3] - d3.size()[3]
        d3 = F.pad(d3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        
        diffY_2 = e2.size()[2] - d2.size()[2]
        diffX_2 = e2.size()[3] - d2.size()[3]
        d2 = F.pad(d2, [diffX_2 // 2, diffX_2 - diffX_2 // 2, diffY_2 // 2, diffY_2 - diffY_2 // 2])
        
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        
        diffY_1 = e1.size()[2] - d1.size()[2]
        diffX_1 = e1.size()[3] - d1.size()[3]
        d1 = F.pad(d1, [diffX_1 // 2, diffX_1 - diffX_1 // 2, diffY_1 // 2, diffY_1 - diffY_1 // 2])
        
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        
        logits = self.final_conv(d1)
        return logits

# ---------------------------------------------------------------------------
# Sanity Check & Architecture Summary
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: 120 PCA-reduced spectral bands, predicting 4 tissue classes
    BANDS = 120
    CLASSES = 4
    HEIGHT, WIDTH = 256, 256
    
    print("[INFO] Initializing Hyperspectral U-Net...")
    model = HyperspectralUNet(in_channels=BANDS, num_classes=CLASSES)
    
    # Create a dummy batch (1 sample, 120 bands, 256x256 pixels)
    # Using float32 tensors which is the default
    dummy_input = torch.randn((1, BANDS, HEIGHT, WIDTH))
    
    print(f"[INFO] Passing dummy hyperspectral cube: {dummy_input.shape}")
    output = model(dummy_input)
    
    print(f"[SUCCESS] Output spatial map shape: {output.shape} -> (Batch, Classes, Height, Width)")
