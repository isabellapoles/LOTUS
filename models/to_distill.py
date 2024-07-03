"""Objects for knowledge distillation."""

import torch
import torch.nn as nn

from torch.nn import functional as F

from models.unet import MLP
from .diff.diff_main import DiffKD


class CriterionIFV(nn.Module):
    """Intra-class feature variation - knowledge distillation loss."""

    def __init__(self, klass):
        super(CriterionIFV, self).__init__()
        """ Class representing the intra-class feature variation distillation strategy. 
        
        Args:
            klass (arr): class knowledge variation to ditill
        """

        self.klass = klass

    def forward(self, preds_S, preds_T, target_S, target_T):
        """ Returns tensor of the discrepancy between the teacher and student intra-class feature variations."""

        # Feature maps dimension matching
        feat_S = preds_S
        feat_T = preds_T
        feat_T.detach()
        size_f = (feat_S.shape[2], feat_S.shape[3])
        tar_feat_S = nn.Upsample(size_f, mode='nearest')(target_S.float()).expand(feat_S.size())
        tar_feat_T = nn.Upsample(size_f, mode='nearest')(target_T.float()).expand(feat_T.size())

        # Feature maps centers as prototypes for each class
        center_feat_S = feat_S.clone()
        center_feat_T = feat_T.clone()
        for k in self.klass: 
            mask_feat_S = (tar_feat_S == k).float()
            mask_feat_T = (tar_feat_T == k).float()
            center_feat_S = (1 - mask_feat_S) * center_feat_S + mask_feat_S * ((mask_feat_S * feat_S).sum(-1).sum(-1) / (mask_feat_S.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)
            center_feat_T = (1 - mask_feat_T) * center_feat_T + mask_feat_T * ((mask_feat_T * feat_T).sum(-1).sum(-1) / (mask_feat_T.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)

        # Intra-class feature varaition as cosine similarity between each feature and feature prototype center for each class
        cos = nn.CosineSimilarity(dim=1)
        pcsim_feat_S = cos(feat_S, center_feat_S)
        pcsim_feat_T = cos(feat_T, center_feat_T)
        
        # MSE as distillation loss between student and teacher intra-class feature variations
        mse = nn.MSELoss()
        loss = mse(pcsim_feat_S, pcsim_feat_T)

        return loss


class CriterionKD(nn.Module):
    """ Object for the Kullback-Leibler divergence - knowledge distillation loss."""
    def __init__(self, temperature = 10):
        """ Class for the Kullback-Leibler divergence. 
        
        Args:
            temperature (int): temperature parameter
        """
        super(CriterionKD, self).__init__()
        self.temperature = temperature
        self.kl = nn.KLDivLoss(size_average=False, reduce=False).cuda() 

    def forward(self, pred, soft):
        """ Returns Kullback-Leibler divergence tensor."""

        loss = self.kl(F.log_softmax(pred / self.temperature, dim = 1), F.softmax(soft / self.temperature, dim = 1)).sum(-1).mean() 

        return loss


KD_MODULES = {
    'student': dict(modules=['conv9'], channels=[16]), 
    'teacher': dict(modules=['conv9'], channels=[16])
}

class DiffDenCriterion(nn.Module):
    """ Object for the diffusion denoising - knowledge distillation loss."""

    def __init__(self, student, device): 
        super (DiffDenCriterion, self).__init__()
        """ Class for the diffusion denoising knowledge distillation process. 
        
        Args:
            student (model): student model
            device (str): device to use
        """

        # Knowledge distillation loss initialization
        kernel_sizes = [3]  
        student_modules = KD_MODULES['student']['modules']
        student_channels = KD_MODULES['student']['channels']
        teacher_modules = KD_MODULES['teacher']['modules']
        teacher_channels = KD_MODULES['teacher']['channels']

        self.diff = nn.ModuleDict()
        self.kd_loss = nn.ModuleDict()

        # Student denoising
        for tm, tc, sc, ks in zip(teacher_modules, teacher_channels, student_channels, kernel_sizes):
            self.diff[tm] = DiffKD(sc, tc, kernel_size=ks)
            self.kd_loss[tm] = CriterionKD()    
        
        self.diff.cuda()
        self.student_modules = student_modules
        self.teacher_modules = teacher_modules

        # Add diff module to the student model for optimization
        self.student = student
        self.student._diff = self.diff
        self._iter = 0
        self.align_layer = MLP(512, 512).to(device)
        self.student._align_layer = self.align_layer

    def forward(self, student_feat, teacher_feat):
        """ Return knowledge distillation loss between the denoised student and the teacher features and the residual noise."""
        
        kd_loss = 0
        student_feats, teacher_feats, anm = {}, {}, {}
        # Student features alignment
        for tm, sm, sf, tf in zip(self.teacher_modules, self.student_modules, student_feat, teacher_feat):
            student_feats[sm] = self.align_layer(sf)
            teacher_feats[tm] = tf

        for tm, sm in zip(self.teacher_modules, self.student_modules):
            # Denoise student feature
            refined_feat, teacher_feat, diff_loss, alpha_anm = self.diff[tm](student_feats[sm], teacher_feats[tm])

            # Compute knowledge distillation loss on denoised student 
            kd_loss_= self.kd_loss[tm](refined_feat, teacher_feat)

            kd_loss += diff_loss
            kd_loss += kd_loss_

            # Save the amount of residual noise 
            anm[tm] = alpha_anm
        
        self._iter += 1
        return kd_loss, anm



