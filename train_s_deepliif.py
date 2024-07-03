"""Training UNet student model on DAPI and ditilling from IHC."""

import os
import re
import wandb
import random
import logging
import argparse
import numpy as np
import skimage.morphology
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)

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
from models.to_distill import *
from models.loss import *
from loaders.datasets import *
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
parser.add_argument("--model_configs", type=str, default='config_s_deepliif.py',
                    help="filename of the model configuration file.")
parser.add_argument('--gpu-id', default = '0',
                    help='id of the gpu to use for training')
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
ifv_criterion = CriterionIFV(klass = configs.train_params["klass_to_distill"][0]).to(device)
kd_criterion = CriterionKD().to(device)
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
imgs_path_list_t = glob('{}*{}'.format(os.path.join(args.dataset_path, configs.train_params["dataset_name_teacher"][0], configs.modality_t, 'val/tissue images/'), 'png'))
imgs_path_list_t.sort(key=lambda f: int(re.sub('\D', '', f)))
imgs_path_list_train_s_dapi = glob('{}*{}'.format(os.path.join(args.dataset_path, configs.train_params["dataset_name_student"][0], configs.modality_s_1, 'train/tissue images/'), 'png'))
imgs_path_list_train_s_dapi.sort(key=lambda f: int(re.sub('\D', '', f)))
imgs_path_list_val_s_dapi = glob('{}*{}'.format(os.path.join(args.dataset_path, configs.train_params["dataset_name_student"][0], configs.modality_s_1, 'val/tissue images/'), 'png'))
imgs_path_list_val_s_dapi.sort(key=lambda f: int(re.sub('\D', '', f)))
imgs_path_list_train_s_pm = glob('{}*{}'.format(os.path.join(args.dataset_path, configs.train_params["dataset_name_student"][0], configs.modality_s_2, 'train/tissue images/'), 'png'))
imgs_path_list_train_s_pm.sort(key=lambda f: int(re.sub('\D', '', f)))
imgs_path_list_val_s_pm = glob('{}*{}'.format(os.path.join(args.dataset_path, configs.train_params["dataset_name_student"][0], configs.modality_s_2, 'val/tissue images/'), 'png'))
imgs_path_list_val_s_pm.sort(key=lambda f: int(re.sub('\D', '', f)))

masks_path_list_t = glob('{}*{}'.format(os.path.join(args.dataset_path, configs.train_params["dataset_name_teacher"][0], configs.modality_t, 'val/masks/'), 'png'))
masks_path_list_t.sort(key=lambda f: int(re.sub('\D', '', f)))
masks_path_list_train_s = glob('{}*{}'.format(os.path.join(args.dataset_path, configs.train_params["dataset_name_student"][0], configs.modality_s_1, 'train/masks/'), 'png'))
masks_path_list_train_s.sort(key=lambda f: int(re.sub('\D', '', f)))
masks_path_list_val_s = glob('{}*{}'.format(os.path.join(args.dataset_path, configs.train_params["dataset_name_student"][0], configs.modality_s_1, 'val/masks/'), 'png'))
masks_path_list_val_s.sort(key=lambda f: int(re.sub('\D', '', f)))

# Augmentation 
aug_params = configs.aug_params["augmentation"]
aug_s = albumentation_aug(aug_params["probability"], aug_params["crop_size_row"][0], aug_params["crop_size_col"])
aug_t = albumentation_aug(aug_params["probability"], aug_params["crop_size_row"][0], aug_params["crop_size_col"])

print ('--- Training start! ---')

with open(os.path.join(CHECKPOINT_PATH,  wandb.run.id + "_result.txt"), "a") as result_file:
    
    fold_dsc = []
    fold_aji = []
    fold_pq = []

    # Create a new instance of the model
    model_ft = UNet(input_channels = configs.model_params["input_ch"], nclasses = configs.model_params["nr_classes"]) 
    align_layer = MLP()

    # Create data handler
    train_data = MultiDeepLIIF(imgs_ihc_path_list = imgs_path_list_t, 
                                        masks_ihc_path_list = masks_path_list_t, 
                                        imgs_dapi_path_list = imgs_path_list_train_s_dapi, 
                                        imgs_pm_path_list = imgs_path_list_train_s_pm, 
                                        masks_dapi_path_list = masks_path_list_train_s, 
                                        aug_ihc = None, 
                                        aug_dapi = aug_s, 
                                        img_dim_row = aug_params["crop_size_row"][0], 
                                        img_dim_col = aug_params["crop_size_col"], 
                                        classes = configs.train_params["klass_to_distill"][0], 
                                        seed = SEED)
    val_data = MultiDeepLIIF(imgs_ihc_path_list = imgs_path_list_t, 
                                        masks_ihc_path_list = masks_path_list_t, 
                                        imgs_dapi_path_list = imgs_path_list_val_s_dapi, 
                                        imgs_pm_path_list = imgs_path_list_val_s_pm, 
                                        masks_dapi_path_list = masks_path_list_val_s, 
                                        aug_ihc = None, 
                                        aug_dapi = aug_s, 
                                        img_dim_row = aug_params["crop_size_row"][0], 
                                        img_dim_col = aug_params["crop_size_col"], 
                                        classes = configs.train_params["klass_to_distill"][0],
                                        seed = SEED)
        
    # Load the model
    pre_trained_model_path_s = os.path.join(args.checkpoint_path, configs.train_params["run_id_student"][0], configs.train_params["run_id_student"][0] + '.h5')
    pre_trained_model_path_t = os.path.join(args.checkpoint_path, configs.train_params["run_id_teacher"][0],  configs.train_params["run_id_teacher"][0] + '.h5')
    model_t, model_s = load_separate_model(configs, device, pre_trained_model_path_s, pre_trained_model_path_t)
    dd_criterion = DiffDenCriterion(model_s, device)

    # Create data loaders using the batch_size from the Config class
    dataloaders = {
    'train': DataLoader(train_data, batch_size=configs.train_params["batch_size"][0], shuffle=True), 
    'val': DataLoader(val_data, batch_size=configs.train_params["batch_size"][0]),
    }

    # Define the optimizer
    if configs.train_params["optimizer"] == "adam":
        optimizer_params = configs.train_params["adam"]
        optimizer_s = optim.Adam([{'params': model_s.parameters()}, {'params': align_layer.parameters()}], lr=optimizer_params["lr"][0]) 
        scheduler_s = optim.lr_scheduler.StepLR(optimizer_s, step_size=optimizer_params["step_size"], gamma=optimizer_params["gamma"])
    
    # Define the diffusion model early stopping parameters
    dd_early_stopping_params = configs.train_params["dd_early_stopping"]
    dd_activation = dd_early_stopping_params["dd_activation"]
    alpha_anm_min = dd_early_stopping_params["alpha_anm_min"]
    patient = dd_early_stopping_params["patient"]
    trigger_times = 0

    # Make directory per run 
    os.makedirs(os.path.join(CHECKPOINT_PATH), exist_ok=True)

    wandb.watch(model_s, log = "all")

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
                model_s.train()  # Set student model to training mode
            else:
                model_s.eval()   # Set student model to evaluate mode
                model_t.eval()   # Set teacher model to evaluate mode

            total_loss, total_anm_conv9, total_anm_1_conv9 = 0.0, 0.0, 0.0
            num_imgs, glob_idx = 0, 0
            DSC, AJI, PQ = [], [], []

            for batch in tqdm(dataloaders[phase]):
                # Retrieve features and labels from the current batch
                imgs_s = batch['img'].to(device)
                masks_s = batch['mask'].to(device) 
                imgs_t = batch['img_ihc'].to(device)
                masks_t = batch['mask_ihc'].to(device) 

                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass
                    masks_pred_prob_s, feat_s, feat_9s = model_s(imgs_s)
                    if configs.model_params["nr_classes"] == 1:
                        masks_pred_s = (masks_pred_prob_s > configs.model_params["probab_th"]).float()
                    else: 
                        masks_pred_s = torch.argmax(masks_pred_prob_s, dim=1) 

                    # Copy and bring to cpu
                    masks_set_pred_s = masks_pred_s.cpu().numpy()
                    masks_set_gt_s = masks_s.cpu().numpy()

                    if epoch >= configs.train_params["distill_epoch"]:
                        with torch.no_grad():
                            masks_pred_prob_t, feat_t, feat_9t = model_t(imgs_t)
                            if configs.model_params["nr_classes"] == 1:
                                masks_pred_t = (masks_pred_prob_t > configs.model_params["probab_th"]).float()
                            else: 
                                masks_pred_t = torch.argmax(masks_pred_prob_t, dim=1) 

                    # Segmentation losses
                    if configs.model_params["nr_classes"] == 1:
                        loss_ce_s = ce_criterion (masks_pred_prob_s, masks_s) 
                    else: 
                        loss_ce_s = ce_criterion (masks_pred_prob_s, masks_s.squeeze(1).long()) 
                    loss_dsc_s = dsc_criterion(masks_pred_s, masks_s)   

                    # Knowledge distillation loss
                    loss_kd = kd_criterion (feat_9s, feat_9t)

                    # Diffusion Module Early stopping
                    if dd_activation == 1:
                        loss_diff, alpha_anm = dd_criterion([feat_9s], [feat_9t])
                        loss_s =  loss_dsc_s + .5*loss_ce_s + loss_diff 
                    else:
                        feat_s = align_layer(feat_s)
                        loss_ifv = ifv_criterion (feat_s, feat_t, masks_s, masks_t)
                        loss_s =  loss_dsc_s + .5*loss_ce_s + 2*loss_ifv + loss_kd
                    
                    # Validation step and image prediction wandb logging
                    if (((phase == 'val') & (epoch%25 == 0)) | ((phase == 'val') & (epoch == (configs.train_params["epoch_num"]-1)))):
                        log_count = 0
                        if log_count < configs.train_params["batch_size"][0]:
                            if configs.model_params["nr_classes"] > 1:
                                # From indexes to RGB
                                mask_pred_rgb_s = torch.zeros ((masks_pred_s.shape[0], 3, masks_pred_s.shape[1], masks_pred_s.shape[2]))
                                mask_rgb_s = torch.zeros ((masks_s.shape[0], 3, masks_s.shape[2], masks_s.shape[3]))
                                for i in range (masks_pred_s.shape[0]): 
                                    mask_pred_rgb_s[i, 0, masks_pred_s[i, :, :] == 1] = 255
                                    mask_pred_rgb_s[i, 2, masks_pred_s[i, :, :] == 2] = 255
                                    mask_rgb_s[i, 0, masks_s[i, 0, ...] == 1] = 255
                                    mask_rgb_s[i, 2, masks_s[i, 0, ...] == 2] = 255

                                masks_s = mask_rgb_s.permute(0, 2, 3, 1)
                                masks_pred_s = mask_pred_rgb_s.permute (0, 2, 3, 1)

                            log_val_predictions(imgs_s, masks_s, masks_pred_s, val_table, glob_idx)
                            log_count += 1
                            glob_idx += 1
                    
                    # Segmentation metrics computation
                    for pred in range(len(masks_pred_prob_s)):
                        masks_set_gt_s[pred] = remap_label(masks_set_gt_s[pred])
                        
                        output_raw_0 = np.squeeze(masks_set_pred_s[pred].astype(np.uint8))
                        output_raw = skimage.morphology.label(output_raw_0)
                        output_raw = remove_small_objects(output_raw_0, min_size=50, connectivity=2)
                        output_raww = remap_label(output_raw)

                        if configs.model_params["nr_classes"] == 1:
                            DSC.append(dice(np.squeeze(masks_set_gt_s[pred].astype(np.uint8)), output_raw, labels = [0, 1], include_zero=True))
                        else: 
                            DSC.append(dice(np.squeeze(masks_set_gt_s[pred].astype(np.uint8)), output_raw, labels = [0, 1, 2], include_zero=True))
                        AJI.append(fast_aji(np.squeeze(masks_set_gt_s[pred].astype(np.uint8)), output_raww))
                        PQ.append(fast_pq(np.squeeze(masks_set_gt_s[pred].astype(np.uint8)), output_raww)[0][2])

                        num_imgs += 1
                    
                    # Backpropagation and optimization
                    if phase == 'train': 
                        optimizer_s.zero_grad(set_to_none=True)
                        loss_s.backward()
                        optimizer_s.step()
                    # Loss tracking 
                    total_loss += loss_s
                    # Residual noise tracking 
                    total_anm_conv9 += alpha_anm['conv9']
                    total_anm_1_conv9 += (1-alpha_anm['conv9'])
            
            # Calculate and print the average validation accuracy across the training
            average_loss = total_loss / num_imgs
            average_dsc = sum(DSC) / num_imgs
            average_aji = sum(AJI) / num_imgs
            average_pq = sum(PQ) / num_imgs
            average_anm_conv9 = total_anm_conv9 / len(dataloaders[phase])
            average_anm_1_conv9 = total_anm_1_conv9 / len(dataloaders[phase])

            # DM Early stopping
            if (average_anm_1_conv9 < alpha_anm_min) & (phase == 'val'): 
                trigger_times += 1
                print ('Triggered Times: %d' % trigger_times)
                alpha_anm_min = average_anm_1_conv9
            if (trigger_times >= patient) & (phase == 'train'):
                print("DM Early Stopped Training At Epoch %d" % epoch)
                print ("1 - ALFPHA_ANM", alpha_anm_min)
                dd_activation = 0
            
            # wandb metrics logging
            wandb.log({phase + " loss" : average_loss, 'epoch': epoch})
            wandb.log({phase + " BG dsc" : average_dsc[0], 'epoch': epoch})
            wandb.log({phase + " aji ": average_aji, 'epoch': epoch})
            wandb.log({phase + " pq ": average_pq, 'epoch': epoch})
            wandb.log({phase + " anm conv9 ": average_anm_conv9, 'epoch': epoch})
            wandb.log({phase + " anm 1-conv9n": average_anm_1_conv9, 'epoch': epoch})
            
            if configs.model_params["nr_classes"] == 1:
                wandb.log({phase + " FG dsc " : average_dsc[1], 'epoch': epoch}) 
                print (f'--- {phase} Loss: {average_loss:.4f} DSC-BG: {average_dsc[0]:.4f} DSC-FG: {average_dsc[1]:.4f} AJI: {average_aji:.4f} PQ: {average_pq:.4f} ANM CONV9: {average_anm_conv9:.4f} ANM 1-CONV9: {average_anm_1_conv9:.4f}') 
            else: 
                wandb.log({phase + " POS dsc " : average_dsc[1], 'epoch': epoch}) 
                wandb.log({phase + " NEG dsc " : average_dsc[2], 'epoch': epoch})        
                wandb.log({phase + " avg dsc " : np.mean(average_dsc[1:]), 'epoch': epoch}) 
                print (f'--- {phase} Loss: {average_loss:.4f} DSC-BG: {average_dsc[0]:.4f} DSC-POS: {average_dsc[1]:.4f} DSC-NEG: {average_dsc[2]:.4f} DSC-AVG: {np.mean(average_dsc[1:]):.4f} AJI: {average_aji:.4f} PQ: {average_pq:.4f} ') #ANM CONV9: {average_anm_conv9:.4f} ANM 1-CONV9: {average_anm_1_conv9:.4f}

            if phase == 'val': 
                # Store validation results
                fold_dsc.append(average_dsc)
                fold_aji.append(average_aji)
                fold_pq.append(average_pq)

                if ((epoch%50 == 0) | (epoch == 199) | (epoch == (configs.train_params["epoch_num"]-1))):
                    wandb.log({"Validation predictions epoch: " + str (epoch) : val_table})

            # Saves checkpoint to disk
            if ((epoch%50 == 0) | (epoch == (configs.train_params["epoch_num"]-1))): 
                torch.save(model_s.state_dict(), os.path.join(CHECKPOINT_PATH, str(epoch) + '_' + wandb.run.id + '.h5'))
                # Saves checkpoint to wandb
                wandb.save(CHECKPOINT_PATH, base_path = configs.wandb_params["base_path"]) 

    print()

    # Calculate and print the average validation accuracy 
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
