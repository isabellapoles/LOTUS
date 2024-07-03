"""Testing UNet student model on SR-microCT."""

import os
import re
import time
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
from medpy import metric
from skimage.morphology import remove_small_objects

from torch.utils.data import DataLoader

from utils import *
from models.metrics import *
from models.unet import UNet
from models.to_distill import *
from loaders.datasets import *


c_wd = os.getcwd()

# Parse input arguments
parser = argparse.ArgumentParser(description='LOTUS - Testing',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--checkpoint-path', default='./checkpoint/bonesai',
                    help='path name of the checkpoint file')
parser.add_argument('--log-freq', type = int, default = 1, 
                    help='how many batches to wait before logging training status')
parser.add_argument('--dataset-path', default='./data',
                    help='path name of the training dataset to process')
parser.add_argument("--model_configs", type=str, default='config_s_bonesai.py', 
                    help="filename of the model configuration file.")
parser.add_argument('--save-path', default = './models',
                    help='path name of the model to be saved')
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
imgs_path_list = glob(os.path.join(args.dataset_path, configs.test_params["dataset_name"], '*/tissue images/*.tif'))
imgs_path_list.sort(key=lambda f: int(re.sub('\D', '', f)))
masks_path_list = glob(os.path.join(args.dataset_path, configs.test_params["dataset_name"], '*/mask binary/*.png'))
masks_path_list.sort(key=lambda f: int(re.sub('\D', '', f)))

# Create a new instance of the model
model_ft = UNet(configs.model_params["input_ch"], configs.model_params["nr_classes"])   
nr_patches = configs.test_params["nr_patches"]
data_handler = BonesAITest(imgs_path_list, masks_path_list, nr_patches, configs.test_params["img_dim_row"], configs.test_params["img_dim_col"], SEED, modality = configs.modality_s)
# Load the model
dd_criterion = DiffDenCriterion(model_ft, device)
model_ft._align_layer = MLP(configs.test_params["img_dim_row"], configs.test_params["img_dim_col"]).to(device)
model_ft.load_state_dict(torch.load(os.path.join(args.checkpoint_path, configs.test_params["run_id"], configs.test_params["checkpoint_fold"], configs.test_params["run_id"] + '.h5')))
model = model_ft.to(device)

# Create data loaders using the batch_size from the Config class
dataloaders = {
    phase: DataLoader(data_handler, 
    batch_size = configs.test_params["batch_size"], 
    shuffle = False,                             
    worker_init_fn = np.random.seed(SEED))
}

model_ft.eval()

with torch.no_grad():
    patients_dsc, patients_hd, patients_aji, patients_pq = [], [], [], []
    for batch in tqdm(dataloaders[phase]):
        imgs = batch['img'].to(device).squeeze(0)
        masks = batch['mask'].to(device).squeeze(0)

        DSC, HD95, AJI, PQ = [], [], [], []

        # Forward pass per each WSI 
        for patch in range(imgs.size(0)): 
            patch_batch = imgs[patch].unsqueeze(0)
            masks_pred_prob, _, _ = model(patch_batch) 
            masks_pred = (masks_pred_prob > configs.test_params["probab_th"]).float()
            
            # Copy and bring to cpu
            masks_set_pred = masks_pred.cpu().numpy()
            masks_set_pred[masks_set_pred>0] = 1
            masks_set_gt = masks[patch].cpu().numpy()
            masks_set_gt[masks_set_gt>0] = 1

            # Segmentation metrics computation
            if (sum(np.unique(np.squeeze(masks_set_pred.astype(np.uint8)))) + sum(np.unique(np.squeeze(masks_set_gt.astype(np.uint8)))) == 2): 
                HD95.append(metric.binary.hd95(np.squeeze(masks_set_gt.astype(np.uint8)), np.squeeze(masks_set_pred.astype(np.uint8))))
            else: 
                HD95.append(0.0)

            masks_set_gt = skimage.morphology.label(masks_set_gt)
            masks_set_gt = remap_label(masks_set_gt) #labels
            
            output_raw_0 = np.squeeze(masks_set_pred.astype(np.uint8))
            output_raw = skimage.morphology.label(output_raw_0)
            output_raw = remove_small_objects(output_raw, min_size=50, connectivity=2)
            output_raw = remap_label(output_raw) #labels

            DSC.append(dice(np.squeeze(masks_set_gt.astype(np.uint8)), output_raw, labels = [0,1], include_zero=True))
            AJI.append(fast_aji(np.squeeze(masks_set_gt.astype(np.uint8)), output_raw))
            PQ.append(fast_pq(np.squeeze(masks_set_gt.astype(np.uint8)), output_raw)[0][2])

        # Calculate and print the average validation accuracy across the test
        average_dsc = sum(DSC) / imgs.size(0)
        average_hd = sum(HD95) / imgs.size(0)
        average_aji = sum(AJI) / imgs.size(0)
        average_pq = sum(PQ) / imgs.size(0)

        print (f'--- {phase} DSC-BG: {average_dsc[0]:.4f} DSC-FG: {average_dsc[1]:.4f} HD95: {average_hd:.4f} AJI: {average_aji:.4f} PQ: {average_pq:.4f}')

        # Store test results
        patients_dsc.append(average_dsc)
        patients_hd.append(average_hd)
        patients_aji.append(average_aji)
        patients_pq.append(average_pq)
    
    print()

    # Calculate and print the average validation accuracy across the current fold
    average_dsc = sum(patients_dsc) / len(imgs_path_list)
    average_hd = sum(patients_hd) / len(imgs_path_list)
    average_aji = sum(patients_aji) / len(imgs_path_list)
    average_pq = sum(patients_pq) / len(imgs_path_list)

    print (f'--- ALL TEST PATIENTS: DSC-BG: {average_dsc[0]:.4f} ({np.std(patients_dsc, axis = 0)[0]:.4f}) DSC-FG: {average_dsc[1]:.4f} ({np.std(patients_dsc, axis = 0)[1]:.4f}) HD95: {average_hd:.4f} ({np.std(patients_hd, axis = 0):.4f}) AJI: {average_aji:.4f} ({np.std(patients_aji):.4f}) PQ: {average_pq:.4f} ({np.std(patients_pq):.4f})')



