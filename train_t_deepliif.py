"""Training UNet teacher model on IHC."""
import os
import re
import wandb
import random
import argparse
import numpy as np
import skimage.morphology

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from glob import glob
from skimage.morphology import remove_small_objects

from torch.utils.data import DataLoader

from utils import *
from models.unet import *
from models.metrics import *
from loaders.datasets import *
from models.loss import DSCLoss
from loaders.augmentation import albumentation_aug


c_wd = os.getcwd()

# Parse input arguments
parser = argparse.ArgumentParser(description='LOTUS - Training',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--checkpoint-path', default='./checkpoint/DeepLIIF',
                    help='path name of the checkpoint file')
parser.add_argument('--log-freq', type = int, default = 1, 
                    help='how many batches to wait before logging training status')
parser.add_argument('--dataset-path', default='./data',
                    help='path name of the training dataset to process')
parser.add_argument("--model_configs", type=str, default='config_t_deepliif.py',
                    help="filename of the model configuration file.")
parser.add_argument('--gpu-id', default = '0',
                    help='id of the gpu to use for training')
parser.add_argument('--run-id', default = 'xxxxxxxx',
                    help='previous wandb run id if the training must be restarted')

args = parser.parse_args()

login_wandb_cmd = 'wandb login'                         
os.system(login_wandb_cmd)

os.environ['CUDA_VISIBLE_DEVICES']= args.gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

configs = load_config(args.model_configs)

# Seeding
SEED = 19                                  
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

# Losses initilization 
dsc_criterion = DSCLoss(multiclass = False, num_classes = configs.model_params["nr_classes"]) 
if configs.model_params["nr_classes"] == 1:
    ce_criterion = nn.BCELoss()
else: 
    ce_criterion = nn.CrossEntropyLoss() 

# WandB
if configs.train_params["start_epoch"] == 0:
    id = wandb.util.generate_id() 
    wandb.init(project=configs.wandb_params["project_name"], entity=configs.wandb_params["entity"], id= id, resume = False, dir = './' ) 
else:
    wandb.init(project=configs.wandb_params["project_name"], entity=configs.wandb_params["entity"], id = args.run_id, resume = True)

wandb.run.name = wandb.run.id
wandb.run.save()

CHECKPOINT_PATH = os.path.join (args.checkpoint_path, wandb.run.id)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

wandb.watch_called = False                  # Re-run the model without restarting the runtime, unnecessary after our next release

# WandB config is a variable that holds and saves hyperparameters and inputs
config = wandb.config                       # Initialize config
config.no_cuda = False                      # Disables CUDA training

# Defining data path lists
imgs_path_list_train = glob('{}*{}'.format(os.path.join(args.dataset_path, configs.train_params["dataset_name"][0], configs.modality, 'train/tissue images/'), 'png'))
masks_path_list_train = glob('{}*{}'.format(os.path.join(args.dataset_path, configs.train_params["dataset_name"][0], configs.modality, 'train/masks/'), 'png'))
imgs_path_list_train.sort(key=lambda f: int(re.sub('\D', '', f)))
masks_path_list_train.sort(key=lambda f: int(re.sub('\D', '', f)))

imgs_path_list_val = glob('{}*{}'.format(os.path.join(args.dataset_path, configs.train_params["dataset_name"][0], configs.modality, 'val/tissue images/'), 'png'))
masks_path_list_val = glob('{}*{}'.format(os.path.join(args.dataset_path, configs.train_params["dataset_name"][0], configs.modality, 'val/masks/'), 'png'))
imgs_path_list_val.sort(key=lambda f: int(re.sub('\D', '', f)))
masks_path_list_val.sort(key=lambda f: int(re.sub('\D', '', f)))

# Augmentation 
aug_params = configs.aug_params["augmentation"]
aug = albumentation_aug(aug_params["probability"], aug_params["crop_size_row"][0], aug_params["crop_size_col"])

print ('--- Training start! ---')

with open(os.path.join(CHECKPOINT_PATH,  wandb.run.id + "_result.txt"), "a") as result_file:

    fold_dsc = []
    fold_aji = []
    fold_pq = []

    # Create a new instance of the model
    model_ft = UNet(input_channels = configs.model_params["input_ch"], nclasses = configs.model_params["nr_classes"]) 

    # Create data handler
    train_data = DeepLIIF(imgs_path_list_train, masks_path_list_train, None, aug, aug_params["crop_size_row"], aug_params["crop_size_col"], configs.model_params["nr_classes"], SEED)
    val_data = DeepLIIF(imgs_path_list_val, masks_path_list_val, None, None, aug_params["crop_size_row"], aug_params["crop_size_row"], configs.model_params["nr_classes"], SEED)
    model = model_ft.to(device)

    # Create data loaders using the batch_size from the Config class
    dataloaders = {
    'train': DataLoader(train_data, batch_size=configs.train_params["batch_size"][0], shuffle=True), 
    'val': DataLoader(val_data, batch_size=configs.train_params["batch_size"][0]),
    }

    # Define the optimizer 
    if configs.train_params["optimizer"] == "adam":
        optimizer_params = configs.train_params["adam"]
        optimizer = optim.Adam(model.parameters(), lr=optimizer_params["lr"][0])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=optimizer_params["step_size"], gamma=optimizer_params["gamma"])
    
    # Make directory per run 
    os.makedirs(os.path.join(CHECKPOINT_PATH), exist_ok=True)

    wandb.watch(model, log = "all")

    # Training loop 
    for epoch in range(configs.train_params["start_epoch"], configs.train_params["epoch_num"]):

        print(f'Epoch {epoch}/{configs.train_params["epoch_num"] - 1}')
        print('-' * 10)

        # Wandb utilities
        columns=["id", "image", "guess", "truth"]
        val_table = wandb.Table(columns = columns)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            total_loss = 0.0
            num_imgs, glob_idx = 0, 0
            DSC, AJI, PQ = [], [], []

            for batch in tqdm(dataloaders[phase]):
                # Retrieve features and labels from the current batch
                imgs = batch['img'].to(device)
                masks = batch['mask'].to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass
                    masks_pred_prob, _, _ = model(imgs)
                    if configs.model_params["nr_classes"] == 1:
                        masks_pred = (masks_pred_prob > configs.model_params["probab_th"]).float()
                    else: 
                        masks_pred = torch.argmax(masks_pred_prob, dim=1) 

                    # Copy and bring to cpu
                    masks_set_pred = masks_pred.cpu().numpy()
                    masks_set_gt = masks.cpu().numpy()

                    # Calculate the loss
                    if configs.model_params["nr_classes"] == 1:
                        loss = .5*ce_criterion (masks_pred_prob, masks) + dsc_criterion(masks_pred, masks)
                    else: 
                        loss = .5*ce_criterion (masks_pred_prob, masks.squeeze(1).long()) + dsc_criterion(masks_pred, masks)
                    
                    # Validation step and image prediction wandb logging
                    if (((phase == 'val') & (epoch%25 == 0)) | ((phase == 'val') & (epoch == (configs.train_params["epoch_num"]-1)))):
                        log_count = 0
                        if log_count < configs.train_params["batch_size"][0]:
                            if configs.model_params["nr_classes"] > 1:
                                # From indexes to RGB
                                mask_pred_rgb = torch.zeros ((masks_pred.shape[0], 3, masks_pred.shape[1], masks_pred.shape[2]))
                                mask_rgb = torch.zeros ((masks.shape[0], 3, masks.shape[2], masks.shape[3]))
                                
                                for i in range (masks_pred.shape[0]): 
                                    mask_pred_rgb[i, 0, masks_pred[i, :, :] == 1] = 255
                                    mask_pred_rgb[i, 2, masks_pred[i, :, :] == 2] = 255
                                    mask_rgb[i, 0, masks[i, 0, ...] == 1] = 255
                                    mask_rgb[i, 2, masks[i, 0, ...] == 2] = 255
                                
                                masks = mask_rgb.permute(0, 2, 3, 1)
                                masks_pred = mask_pred_rgb.permute (0, 2, 3, 1)

                            log_val_predictions(imgs, masks, masks_pred, val_table, glob_idx)
                            log_count += 1
                            glob_idx += 1
                    
                    # Segmentation metrics computation
                    for pred in range(len(masks_pred_prob)): 
                        masks_set_gt[pred] = remap_label(masks_set_gt[pred])
                        
                        output_raw_0 = np.squeeze(masks_set_pred[pred].astype(np.uint8))
                        output_raw = skimage.morphology.label(output_raw_0)
                        output_raw = remove_small_objects(output_raw_0, min_size=50, connectivity=2)
                        output_raww = remap_label(output_raw)

                        if configs.model_params["nr_classes"] == 1:
                            DSC.append(dice(np.squeeze(masks_set_gt[pred].astype(np.uint8)), output_raw, labels = [0, 1], include_zero=True))
                        else: 
                            DSC.append(dice(np.squeeze(masks_set_gt[pred].astype(np.uint8)), output_raw, labels = [0, 1, 2], include_zero=True))
                        AJI.append(fast_aji(np.squeeze(masks_set_gt[pred].astype(np.uint8)), output_raww))
                        PQ.append(fast_pq(np.squeeze(masks_set_gt[pred].astype(np.uint8)), output_raww)[0][2])
                        num_imgs += 1
            
                    # Backpropagation and optimization
                    if phase == 'train': 
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    total_loss += loss.item()
            
            # Calculate and print the average validation accuracy across the training
            average_loss = total_loss / num_imgs
            average_dsc = sum(DSC) / num_imgs
            average_aji = sum(AJI) / num_imgs
            average_pq = sum(PQ) / num_imgs

            # wandb metrics logging
            wandb.log({phase + " loss" : average_loss, 'epoch': epoch})
            wandb.log({phase + " BG dsc" : average_dsc[0], 'epoch': epoch})
            wandb.log({phase + " aji ": average_aji, 'epoch': epoch})
            wandb.log({phase + " pq ": average_pq, 'epoch': epoch})
            
            if configs.model_params["nr_classes"] == 1:
                wandb.log({phase + " FG dsc " : average_dsc[1], 'epoch': epoch}) 
                print (f'--- {phase} Loss: {average_loss:.4f} DSC-BG: {average_dsc[0]:.4f} DSC-FG: {average_dsc[1]:.4f} AJI: {average_aji:.4f} PQ: {average_pq:.4f}') 

            else: 
                wandb.log({phase + " POS dsc " : average_dsc[1], 'epoch': epoch}) 
                wandb.log({phase + " NEG dsc " : average_dsc[2], 'epoch': epoch})        
                wandb.log({phase + " avg dsc " : np.mean(average_dsc[1:]), 'epoch': epoch}) 
                print (f'--- {phase} Loss: {average_loss:.4f} DSC-BG: {average_dsc[0]:.4f} DSC-POS: {average_dsc[1]:.4f} DSC-NEG: {average_dsc[2]:.4f} DSC-AVG: {np.mean(average_dsc[1:]):.4f} AJI: {average_aji:.4f} PQ: {average_pq:.4f}') 


            if phase == 'val': 
                # Store validation results for the current fold
                fold_dsc.append(average_dsc)
                fold_aji.append(average_aji)
                fold_pq.append(average_pq)

                if ((epoch%25 == 0) | (epoch == (configs.train_params["epoch_num"]-1))):
                    wandb.log({"Validation predictions epoch: " + str (epoch) : val_table})
            
            # Saves checkpoint to disk
            if ((epoch%50 == 0) | (epoch == (configs.train_params["epoch_num"]-1))): 
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_PATH, str(epoch) + '_' + wandb.run.id + '.h5'))
                # Saves checkpoint to wandb
                wandb.save(CHECKPOINT_PATH, base_path = configs.wandb_params["base_path"]) 

    print()

    # Calculate and print the average validation accuracy across the training
    average_dsc = sum(fold_dsc) / int(configs.train_params["epoch_num"] - configs.train_params["start_epoch"])
    average_aji = sum(fold_aji) / int(configs.train_params["epoch_num"] - configs.train_params["start_epoch"])
    average_pq = sum(fold_pq) / int(configs.train_params["epoch_num"] - configs.train_params["start_epoch"])

    if configs.model_params["nr_classes"] == 1:
        result_file.writelines("\n%s\n" %(f'--- {phase} Loss: {average_loss:.4f} DSC-BG: {average_dsc[0]:.4f} DSC-FG: {average_dsc[1]:.4f} AJI: {average_aji:.4f} PQ: {average_pq:.4f}'))
        print (f'--- {phase} Loss: {average_loss:.4f} DSC-BG: {average_dsc[0]:.4f} DSC-FG: {average_dsc[1]:.4f} AJI: {average_aji:.4f} PQ: {average_pq:.4f}')
        print()
    else: 
        result_file.writelines("\n%s\n" %(f'--- {phase} Loss: {average_loss:.4f} DSC-BG: {average_dsc[0]:.4f} DSC-POS: {average_dsc[1]:.4f} DSC-NEG: {average_dsc[2]:.4f} DSC-AVG: {np.mean(average_dsc[1:]):.4f} AJI: {average_aji:.4f} PQ: {average_pq:.4f}'))
        print (f'--- {phase} Loss: {average_loss:.4f} DSC-BG: {average_dsc[0]:.4f} DSC-POS: {average_dsc[1]:.4f} DSC-NEG: {average_dsc[2]:.4f} DSC-AVG: {np.mean(average_dsc[1:]):.4f} AJI: {average_aji:.4f} PQ: {average_pq:.4f}')
        print()
