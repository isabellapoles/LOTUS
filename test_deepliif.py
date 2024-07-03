"""Testing UNet student model on DAPI."""

import os
import re
import random
import logging
import argparse
import numpy as np
import skimage.morphology
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)

import torch

from tqdm import tqdm
from glob import glob
from skimage.morphology import remove_small_objects

from torch.utils.data import DataLoader

from utils import *
from models.unet import UNet
from models.metrics import *
from loaders.datasets import *
from models.to_distill import *

c_wd = os.getcwd()

# Parse input arguments
parser = argparse.ArgumentParser(description='LOTUS - Testing',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--checkpoint-path', default='./checkpoint/DeepLIIF',
                    help='path name of the checkpoint file')
parser.add_argument('--log-freq', type = int, default = 1, 
                    help='how many batches to wait before logging training status')
parser.add_argument('--dataset-path', default='./data',
                    help='path name of the training dataset to process')
parser.add_argument("--model_configs", type=str, default='config_s_deepliif.py',
                    help="filename of the model configuration file.")
parser.add_argument('--gpu-id', default = '0',
                    help='id of the gpu to use for training')

args = parser.parse_args()

configs = load_config(args.model_configs)

id = configs.test_params["run_id"]

os.environ['CUDA_VISIBLE_DEVICES']= args.gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Seeding
SEED = 19                                  
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

CHECKPOINT_PATH = os.path.join (args.checkpoint_path, id)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

phase = configs.test_params["phase"]

# Defining data path lists
imgs_path_list_dapi = glob('{}*{}'.format(os.path.join(args.dataset_path, configs.test_params["dataset_name"], configs.modality_s_1, 'test/tissue images/'), 'png'))
masks_path_list_dapi = glob('{}*{}'.format(os.path.join(args.dataset_path, configs.test_params["dataset_name"], configs.modality_s_1, 'test/masks/'), 'png'))
imgs_path_list_dapi.sort(key=lambda f: int(re.sub('\D', '', f)))
masks_path_list_dapi.sort(key=lambda f: int(re.sub('\D', '', f)))

imgs_path_list_pm = glob('{}*{}'.format(os.path.join(args.dataset_path, configs.test_params["dataset_name"], configs.modality_s_2, 'test/tissue images/'), 'png'))
imgs_path_list_pm.sort(key=lambda f: int(re.sub('\D', '', f)))
aug_params = configs.aug_params["augmentation"]

# Create a new instance of the model for each fold
model_ft = UNet(input_channels = configs.model_params["input_ch"], nclasses = configs.model_params["nr_classes"]) 
test_data = DeepLIIF(imgs_path_list_dapi, masks_path_list_dapi, imgs_path_list_pm, None, aug_params["crop_size_row"], aug_params["crop_size_row"], configs.model_params["nr_classes"], SEED)

# Load the model to be finetuned on BonesAI
dd_criterion = DiffDenCriterion(model_ft, device)
model_ft._align_layer = MLP(aug_params["crop_size_row"][0], aug_params["crop_size_row"][0]).to(device)
model_ft.load_state_dict(torch.load(os.path.join(args.checkpoint_path, configs.test_params["run_id"], configs.test_params["run_id"] + '.h5')))
model = model_ft.to(device)

# Create data loaders using the batch_size from the Config class
dataloaders = {
    phase: DataLoader(test_data, 
    batch_size = configs.test_params["batch_size"], 
    shuffle = False,                             
    worker_init_fn = np.random.seed(SEED))
}

model_ft.eval()

with torch.no_grad():
    patients_dsc, patients_aji, patients_pq = [], [], []
    
    for batch in tqdm(dataloaders[phase]):

        imgs = batch['img'].to(device)
        masks = batch['mask'].to(device)
        
        DSC, AJI, PQ = [], [], []

        # Forward pass
        masks_pred_prob, _, _ = model(imgs)
        if configs.model_params["nr_classes"] == 1:
            masks_pred = (masks_pred_prob > configs.model_params["probab_th"]).float()
        else: 
            masks_pred = torch.argmax(masks_pred_prob, dim=1)
            
            mask_pred_rgb = torch.zeros ((3, masks_pred.shape[1], masks_pred.shape[2]))
            mask_rgb = torch.zeros ((3, masks_pred.shape[1], masks_pred.shape[2]))
            mask_pred_rgb[0, masks_pred[0] == 1] = 255
            mask_pred_rgb[2, masks_pred[0] == 2] = 255
            mask_rgb[0, masks[0, 0, ...] == 1] = 255
            mask_rgb[2, masks[0, 0, ...] == 2] = 255

        # Copy and bring to cpu
        masks_set_pred = masks_pred.cpu().numpy()
        masks_set_gt = masks.cpu().numpy()

        if configs.model_params["nr_classes"] == 1:
            masks_set_pred[masks_set_pred>0] = 1
            masks_set_gt[masks_set_gt>0] = 1
        
        # Segmentation metrics computation
        for pred in range(len(masks_pred_prob)): 
            if configs.model_params["nr_classes"] == 1:
                masks_set_gt[pred] = skimage.morphology.label(masks_set_gt[pred])
                masks_set_gt[pred] = remap_label(masks_set_gt[pred]) 
                
                output_raw_0 = np.squeeze(masks_set_pred[pred].astype(np.uint8))
                output_raw = skimage.morphology.label(output_raw_0)
                output_raw = remove_small_objects(output_raw, min_size=50, connectivity=2)
                output_raw = remap_label(output_raw) 
                
                DSC.append(dice(np.squeeze(masks_set_gt[pred].astype(np.uint8)), output_raw, labels = [0, 1], include_zero=True))
                AJI.append(fast_aji(np.squeeze(masks_set_gt[pred].astype(np.uint8)), output_raw))
                PQ.append(fast_pq(np.squeeze(masks_set_gt[pred].astype(np.uint8)), output_raw)[0][2])
           
            else: 

                output_raw = remove_small_objects(masks_set_pred[pred], min_size=50, connectivity=2)
                
                DSC.append(dice(np.squeeze(masks_set_gt[pred].astype(np.uint8)), output_raw, labels = [0, 1, 2], include_zero=True))
                
                masks_set_gt[pred] = skimage.morphology.label(masks_set_gt[pred])
                masks_set_gt[pred] = remap_label(masks_set_gt[pred]) 
                
                masks_set_pred = masks_set_pred[pred]
                masks_set_pred[masks_set_pred>0] = 1
                masks_set_gt[masks_set_gt>0] = 1
                output_raw_0 = np.squeeze(masks_set_pred.astype(np.uint8))
                output_raw = skimage.morphology.label(output_raw_0)
                output_raw = remove_small_objects(output_raw, min_size=50, connectivity=2)
                output_raw = remap_label(output_raw) 

                AJI.append(fast_aji(np.squeeze(masks_set_gt.astype(np.uint8)), output_raw))
                PQ.append(fast_pq(np.squeeze(masks_set_gt.astype(np.uint8)), output_raw)[0][2])
            
        # Calculate and print the average validation accuracy across the test
        average_dsc = sum(DSC) / imgs.size(0)
        average_aji = sum(AJI) / imgs.size(0)
        average_pq = sum(PQ) / imgs.size(0)

        if configs.model_params["nr_classes"] == 1:
            print (f'--- {phase} DSC-BG: {average_dsc[0]:.4f} DSC-FG: {average_dsc[1]:.4f} AJI: {average_aji:.4f} PQ: {average_pq:.4f}')
        else:
            print (f'--- {phase} DSC-BG: {average_dsc[0]:.4f} DSC-POS: {average_dsc[1]:.4f} DSC-NEG: {average_dsc[2]:.4f} DSC-AVG: {np.mean(average_dsc[1:]):.4f} AJI: {average_aji:.4f} PQ: {average_pq:.4f}')

        # Store test results
        patients_dsc.append(average_dsc)
        patients_aji.append(average_aji)
        patients_pq.append(average_pq)
    
    print()

    # Calculate and print the average validation accuracy across the current fold
    average_dsc = sum(patients_dsc) / len(imgs_path_list_dapi)
    average_aji = sum(patients_aji) / len(imgs_path_list_dapi)
    average_pq = sum(patients_pq) / len(imgs_path_list_dapi)

    if configs.model_params["nr_classes"] == 1:
        print (f'--- ALL TEST PATIENTS: DSC-BG: {average_dsc[0]:.4f} ({np.std(patients_dsc, axis = 0)[0]:.4f}) DSC-FG: {average_dsc[1]:.4f} ({np.std(patients_dsc, axis = 0)[1]:.4f}) AJI: {average_aji:.4f} ({np.std(patients_aji):.4f}) PQ: {average_pq:.4f} ({np.std(patients_pq):.4f}))')
    else: 
        print (f'--- ALL TEST PATIENTS: DSC-BG: {average_dsc[0]:.4f} ({np.std(patients_dsc, axis = 0)[0]:.4f}) DSC-POS: {average_dsc[1]:.4f} ({np.std(patients_dsc, axis = 0)[1]:.4f}) DSC-NEG: {average_dsc[2]:.4f} ({np.std(patients_dsc, axis = 0)[2]:.4f}) DSC-AVG: {np.mean(average_dsc[1:]):.4f} ({np.mean([np.std(patients_dsc, axis = 0)[0], np.std(patients_dsc, axis = 0)[1], np.std(patients_dsc, axis = 0)[2]]):.4f}) AJI: {average_aji:.4f} ({np.std(patients_aji):.4f}) PQ: {average_pq:.4f} ({np.std(patients_pq):.4f})')



