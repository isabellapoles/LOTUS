"""Utility objects for processing - training - logging."""
import os
import wandb
import importlib

import torch 

import numpy as np
import nibabel as nib

from skimage.morphology import disk, binary_dilation, binary_closing

from models.unet import UNet

def get_dir_and_files_bonesai(dataset_path): 
    """Define how and which images belong to the same sample, to allow correct data stratification. 

        Arguments: 
            dataset_path (str): dataset path
        Returns:
            res (list): list cointaining images belonging to the same sample
    """
    
    res = []
    # List of samples
    for name in os.listdir(dataset_path):
        dir_path = os.path.join(dataset_path, name)
        # List of subsamples
        for sub_name in os.listdir(dir_path):
            sub_dir_path = os.path.join(dir_path, sub_name)
            # Finding images 
            if 'tissue images' in sub_dir_path:
                for sub_file in os.listdir(sub_dir_path):
                    sub_file_path = os.path.join(sub_dir_path, sub_file)
                # Map each image file to the same sample
                res.append(len(list(map(os.path.isfile, os.listdir(sub_dir_path)))))
    return res


def seg_processing (img_path): 
    """SR-microCT lacunae segmentation pre-processing to align segmentation to the images and fill mislabeled masks. 

        Arguments: 
            img_path (str): image segmentation path
        Returns:
            seg_rot_u8 (arr): pre-processed segmentation 
    """

    # Load segmentation 
    seg = nib.load (img_path)
    seg_arr = seg.get_fdata()
    seg_arr_u8 = np.copy (seg_arr)
    seg_arr_u8[seg_arr>0] = 255
    
    # Morphological processing to close mislabeled lacunae 
    seg_dil = binary_dilation(seg_arr_u8[:, :, 0], disk(1)).astype('uint8')
    seg_close = binary_closing(seg_arr_u8[:, :, 0], disk(3)).astype('uint8')
    seg_close [seg_close > 0 ] = 255
    
    # Image segmentation alignement
    seg_rot = np.rot90 (np.flip(seg_close, axis = 1), k = 1) 
    seg_rot_u8 = np.copy (seg_rot)
    seg_rot_u8[seg_rot>0] = 255
    
    return seg_rot_u8


def load_config(config_filename):
    """Load the config files for filling with parameter train/val/test functions.   

        Arguments: 
            config_filename (str): name of the config file 
        Returns:
            module.Config() (module): config module
    """
    config_path = "configs.{}".format(config_filename.split('.')[0])
    module = importlib.import_module(config_path)
    return module.Config()


def log_val_predictions(images, label, predicted, val_table, glob_idx):
    """Convenience funtion to log predictions on wandb for a batch of val/test images.
    
        Arguments: 
            images (tensor): image to predict
            label (tensor): ground truth segmentation
            predicted (tensor): predicted segmentation
            val_table: wandb table to store logs 
            glob_idx (int): index of the batch images to log
    """
  
    # Bring images/segmentations/predictions to the cpu
    log_images = images.cpu().numpy()[np.newaxis, ...]
    log_labels = label.cpu().numpy()[np.newaxis, ...]
    log_preds = predicted.cpu().numpy()[np.newaxis, ...]

    # Adding ids based on the order of the images
    _id = 0
    for patch in range (log_images.shape[1]):
    
        for i, l, p in zip(log_images, log_labels, log_preds): 
        # Add required info to data table:
        # id, image pixels, model's guess, true label, scores for all classes
            # transpose to obtain the right order (row, cols, chnl)
            i = i[patch].transpose(1, 2, 0)
            img_id = str(glob_idx) + "_" + str(_id)
            val_table.add_data(img_id, wandb.Image(i), wandb.Image(p[patch]), wandb.Image(l[patch]))

            _id += 1

def load_separate_model(configs, device, checkpoint_s, checkpoint_t):
    """Convenience funtion to load teacher and student models.
    
        Arguments: 
            configs (module): parameters
            device (str): cpu or gpu to use
            checkpoint_s (weights): student pre-trained weights
            checkpoint_t (weights): teacher pre-trained weights
    """
    # Set the device
    use_gpu = "cpu" != device.type

    # Initializing models 
    if (checkpoint_t is None):
        model_t = UNet(input_channels = configs.model_params["input_ch"], nclasses = configs.model_params["nr_classes"]) 
        model_s = UNet(input_channels = configs.model_params["input_ch"], nclasses = configs.model_params["nr_classes"]) 
    else:
        model_t = UNet(input_channels = configs.model_params["input_ch"], nclasses = configs.model_params["nr_classes"]) 
        model_s = UNet(input_channels = configs.model_params["input_ch"], nclasses = configs.model_params["nr_classes"]) 

        # Loading pre-trained weights
        if use_gpu:
            model_t = model_t.to(device)
            model_t.load_state_dict(torch.load(checkpoint_t, map_location="cuda"))

            model_s = model_s.to(device)
            # We train the student from scratch
            # model_s.load_state_dict(torch.load(checkpoint_s, map_location="cuda"))
        else:
            model_t.load_state_dict(torch.load(checkpoint_t, map_location={"cpu"}))
            # We train the student from scratch
            # model_s.load_state_dict(torch.load(checkpoint_s, map_location={"cpu"}))

    if use_gpu:
        model_t = model_t.to(device)
        model_s = model_s.to(device)

    return model_t, model_s