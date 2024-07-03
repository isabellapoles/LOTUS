# Modified from https://github.com/hunto/DiffKD 

"""Object for the forward and backward denoising diffusion processes."""

import torch
from torch import nn
import torch.nn.functional as F

from .diff_modules import DiffusionModel, NoiseAdapter, DDIMPipeline
from .scheduling_ddim import DDIMScheduler

class DiffKD(nn.Module):
    """Object to train the diffusion model. """
    def __init__(self, student_channels, teacher_channels, kernel_size=3, inference_steps=5, num_train_timesteps=1000):
        super().__init__()
        """ Class to train the diffusion model. 
        
        Args:
            student_channels (int): number of channels of the student input features
            teacher_channels (int): number of channels of the teacher input features
            kernel_size (int): dimension of the convolution kernel to use during noising/denoising
            inference_steps (int): number of step required for denoising 
            num_train_timesteps (int): number of the train time step 
        """

        self.diffusion_inference_steps = inference_steps
        
        # Transform student feature to the same dimension as teacher
        self.trans = nn.Conv2d(student_channels, teacher_channels, 1)
        # Diffusion model - predict noise
        self.model = DiffusionModel(channels_in=teacher_channels, kernel_size=kernel_size)
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False, beta_schedule="linear")
        self.noise_adapter = NoiseAdapter(teacher_channels, kernel_size)
        # Pipeline for denoising student feature
        self.pipeline = DDIMPipeline(self.model, self.scheduler, self.noise_adapter)#
        self.proj = nn.Sequential(nn.Conv2d(teacher_channels, teacher_channels, 1), nn.BatchNorm2d(teacher_channels))

    def forward(self, student_feat, teacher_feat):
        """ Returns: 
                refined_feat (tensor): the denoised student feature map
                teacher_feat (tensor): the noise free teacher feature map
                ddim_loss (tensor): the diffusion loss
                alpha (tensor): the amount of residual noise
        """
        # Project student feature to the same dimension as teacher feature
        student_feat = self.trans(student_feat)

        # Denoise student feature
        refined_feat, alpha = self.pipeline(
            batch_size=student_feat.shape[0],
            device=student_feat.device,
            dtype=student_feat.dtype,
            shape=student_feat.shape[1:],
            feat=student_feat,
            num_inference_steps=self.diffusion_inference_steps,
            proj=self.proj
        )
        refined_feat = self.proj(refined_feat)
        
        # Train diffusion model
        ddim_loss = self.ddim_loss(teacher_feat)
        return refined_feat, teacher_feat, ddim_loss, alpha

    def ddim_loss(self, gt_feat):
        # Sample noise to add to the images
        noise = torch.randn(gt_feat.shape, device=gt_feat.device) #.to(gt_feat.device)
        bs = gt_feat.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt_feat.device).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.scheduler.add_noise(gt_feat, noise, timesteps)
        noise_pred = self.model(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        return loss