"""Training UNet student model on SR-microCT and ditilling from histopathologies."""

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
from models.loss import DSCLoss
from loaders.datasets import *
from loaders.sampler import BonesAIDataSampler
from loaders.augmentation import albumentation_aug


c_wd = os.getcwd()

# parse input arguments
parser = argparse.ArgumentParser(description='LOTUS - Training',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--checkpoint-path', default='./checkpoint/bonesai',
                    help='path name of the checkpoint file')
parser.add_argument('--log-freq', type = int, default = 1, 
                    help='how many batches to wait before logging training status')
parser.add_argument('--dataset-path', default='./data',
                    help='path name of the training dataset to process')
parser.add_argument("--model_configs", type=str, default='config_s_bonesai.py',
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

dsc_criterion = DSCLoss() 
ce_criterion = nn.BCELoss()
ifv_criterion = CriterionIFV(klass = configs.train_params["klass_to_distill"][0]).to(device)
kd_criterion = CriterionKD().to(device)
align_layer = MLP()

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

# Images and masks path and ordering
imgs_path_list_t = glob('{}*{}'.format(os.path.join(args.dataset_path, configs.train_params["dataset_name_teacher"][0], '*/tissue images/'), 'tif'))
imgs_path_list_t.sort(key=lambda f: int(re.sub('\D', '', f)))
imgs_path_list_s = glob('{}*{}'.format(os.path.join(args.dataset_path, configs.train_params["dataset_name_student"][0], '*/tissue images/'), 'tif'))
imgs_path_list_s.sort(key=lambda f: int(re.sub('\D', '', f)))

masks_path_list_t = glob('{}*{}'.format(os.path.join(args.dataset_path, configs.train_params["dataset_name_teacher"][0], '*/mask binary/'), 'png'))
masks_path_list_t.sort(key=lambda f: int(re.sub('\D', '', f)))
masks_path_list_s = glob('{}*{}'.format(os.path.join(args.dataset_path, configs.train_params["dataset_name_student"][0], '*/mask binary/'), 'png'))
masks_path_list_s.sort(key=lambda f: int(re.sub('\D', '', f)))

# Consider image groups based on patient, to perform patient-wise CV
images_per_patient_t = get_dir_and_files_bonesai(os.path.join(args.dataset_path, configs.train_params["dataset_name_teacher"][0]))
groups_t = sum([[i]*images_per_patient_t[i] for i in range (len(images_per_patient_t))], [])

images_per_patient_s = get_dir_and_files_bonesai(os.path.join(args.dataset_path, configs.train_params["dataset_name_student"][0]))
groups_s = sum([[i]*images_per_patient_s[i] for i in range (len(images_per_patient_s))], [])

aug_params = configs.aug_params["augmentation"]
aug_s = albumentation_aug(aug_params["probability"], aug_params["crop_size_row"][0], aug_params["crop_size_col"])
aug_t = albumentation_aug(aug_params["probability"], aug_params["crop_size_row"][0], aug_params["crop_size_col"])

print ('--- Training start! ---')
per_fold_dsc = []
per_fold_aji = []
per_fold_pq = []

num_fold = configs.train_params["num_fold"]

with open(os.path.join(CHECKPOINT_PATH,  wandb.run.id + "_result.txt"), "a") as result_file:
    for fold in range(num_fold):
        print ('FOLD #', fold)
        fold_dsc = []
        fold_aji = []
        fold_pq = []

        # Create a new instance of the model for each fold
        model_ft = UNet(input_channels = configs.model_params["input_ch"], nclasses = configs.model_params["nr_classes"]) 

        # Create data handler for the current fold
        data_handler = MultiBonesAI(imgs_h_path_list = imgs_path_list_t, 
                                    masks_h_path_list = masks_path_list_t, 
                                    imgs_ct_path_list = imgs_path_list_s, 
                                    masks_ct_path_list = masks_path_list_s, 
                                    groups_h = groups_t, 
                                    groups_ct = groups_s,
                                    current_fold = fold, 
                                    aug_h = None, 
                                    aug_ct = aug_s, 
                                    num_fold = num_fold, 
                                    nr_patches_h = configs.train_params["nr_patches_teacher"][0], 
                                    nr_patches_ct = configs.train_params["nr_patches_student"][0], 
                                    img_dim_row = aug_params["crop_size_row"], 
                                    img_dim_col = aug_params["crop_size_col"], 
                                    seed = SEED)
            
        # Load the model
        pre_trained_model_path_s = os.path.join(args.checkpoint_path, configs.train_params["run_id_student"][0], configs.train_params["checkpoint_fold_student"][0], configs.train_params["run_id_student"][0] + '.h5')
        pre_trained_model_path_t = os.path.join(args.checkpoint_path, configs.train_params["run_id_teacher"][0], configs.train_params["checkpoint_fold_teacher"][0], configs.train_params["run_id_teacher"][0] + '.h5')
        model_t, model_s = load_separate_model(configs, device, pre_trained_model_path_s, pre_trained_model_path_t)
        dd_criterion = DiffDenCriterion(model_s, device)

        # Split the dataset into training and validation subsets
        train_s_data, train_s_indices, val_s_data, val_s_indices, train_t_data, train_t_indices = data_handler.get_splits() 
        # Create data loaders using the batch_size from the Config class
        dataloaders = {
        'train': DataLoader(data_handler, 
                                batch_sampler=BonesAIDataSampler((data_handler.__dataframe__()), train_s_indices, configs.train_params["batch_size"][0]),
                                worker_init_fn=np.random.seed(SEED)),
        'val': DataLoader(data_handler, 
                                batch_sampler=BonesAIDataSampler((data_handler.__dataframe__()), val_s_indices, configs.train_params["batch_size"][0]),
                                worker_init_fn=np.random.seed(SEED))}

        # Define a new optimizer for each fold to reset the model parameters
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

        # Make directory per run & fold
        os.makedirs(os.path.join(CHECKPOINT_PATH, "fold_" + str(fold)), exist_ok=True)

        wandb.watch(model_s, log = "all")

        # Training loop for the current fold
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
                    imgs_s = batch['img_ct'].to(device)
                    masks_s = batch['mask_ct'].to(device) 
                    imgs_t = batch['img_h'].to(device)
                    masks_t = batch['mask_h'].to(device) 

                    with torch.set_grad_enabled(phase == 'train'):
                        # Forward pass
                        masks_pred_prob_s, feat_s, feat_9s = model_s(imgs_s)
                        masks_pred_s = (masks_pred_prob_s > configs.model_params["probab_th"]).float()

                        # Copy and bring to cpu
                        masks_set_pred_s = masks_pred_s.cpu().numpy()
                        masks_set_gt_s = masks_s.cpu().numpy()

                        if epoch >= configs.train_params["distill_epoch"]:
                            with torch.no_grad():
                                masks_pred_prob_t, feat_t, feat_9t = model_t(imgs_t)
                                masks_pred_t = (masks_pred_prob_t > configs.model_params["probab_th"]).float()
                                
                        # Segmentation losses
                        loss_ce_s = ce_criterion (masks_pred_prob_s, masks_s) 
                        loss_dsc_s = dsc_criterion(masks_pred_s, masks_s)   

                        loss_kd = kd_criterion (feat_9s, feat_9t)

                        loss_diff, alpha_anm = dd_criterion([feat_9s], [feat_9t])
                        loss_ifv = ifv_criterion (feat_s, feat_t, masks_s, masks_t)
                        loss_s =  loss_dsc_s + .5*loss_ce_s + loss_ifv + loss_diff 

                        # DM Early stopping
                        if dd_activation == 1:
                            loss_diff, alpha_anm = dd_criterion([feat_9s], [feat_9t])
                            loss_s =  loss_dsc_s + .5*loss_ce_s + loss_diff 
                        else:
                            feat_s = align_layer(feat_s)
                            loss_ifv = ifv_criterion (feat_s, feat_t, masks_s, masks_t)
                            loss_s =  loss_dsc_s + .5*loss_ce_s + loss_ifv + loss_kd

                        if (((phase == 'val') & (epoch%5 == 0)) | ((phase == 'val') & (epoch == (configs.train_params["epoch_num"]-1)))):
                            log_count = 0
                            if log_count < configs.train_params["batch_size"][0]:
                                log_val_predictions(imgs_s, masks_s, masks_pred_s, val_table, glob_idx)
                                log_count += 1
                                glob_idx += 1
                        
                        # Segmentation metrics computation
                        for pred in range(len(masks_pred_prob_s)):
                            masks_set_gt_s[pred] = remap_label(masks_set_gt_s[pred])
                            
                            output_raw_0 = np.squeeze(masks_set_pred_s[pred].astype(np.uint8))
                            output_raw = skimage.morphology.label(output_raw_0)
                            output_raw = remove_small_objects(output_raw, min_size=50, connectivity=2)
                            output_raw = remap_label(output_raw)

                            DSC.append(dice(np.squeeze(masks_set_gt_s[pred].astype(np.uint8)), output_raw, labels = [0,1], include_zero=True))
                            AJI.append(fast_aji(np.squeeze(masks_set_gt_s[pred].astype(np.uint8)), output_raw))
                            PQ.append(fast_pq(np.squeeze(masks_set_gt_s[pred].astype(np.uint8)), output_raw)[0][2])

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
                wandb.log({phase + " loss f_" + str(fold): average_loss, 'epoch': epoch})
                wandb.log({phase + " BG dsc f_" + str(fold): average_dsc[0], 'epoch': epoch})
                wandb.log({phase + " FG dsc f_" + str(fold): average_dsc[1], 'epoch': epoch})        
                wandb.log({phase + " aji f_" + str(fold): average_aji, 'epoch': epoch})
                wandb.log({phase + " pq f_" + str(fold): average_pq, 'epoch': epoch})
                wandb.log({phase + " anm conv9 f_" + str(fold): average_anm_conv9, 'epoch': epoch})
                wandb.log({phase + " anm 1-conv9 f_" + str(fold): average_anm_1_conv9, 'epoch': epoch})

                print (f'--- {phase} Loss: {average_loss:.4f} DSC-BG: {average_dsc[0]:.4f} DSC-FG: {average_dsc[1]:.4f} AJI: {average_aji:.4f} PQ: {average_pq:.4f} ANM CONV9: {average_anm_conv9:.4f} ANM 1-CONV9: {average_anm_1_conv9:.4f}') #

                if phase == 'val': 
                    # Store validation results for the current fold
                    fold_dsc.append(average_dsc)
                    fold_aji.append(average_aji)
                    fold_pq.append(average_pq)

                    if ((epoch%5 == 0) | (epoch == 199) | (epoch == (configs.train_params["epoch_num"]-1))):
                        wandb.log({"Validation predictions epoch: " + str (epoch) + " fold: " + str(fold) : val_table})

                if ((epoch%10 == 0) | (epoch == (configs.train_params["epoch_num"]-1))): 
                    torch.save(model_s.state_dict(), os.path.join(CHECKPOINT_PATH, "fold_" + str(fold), str(epoch) + '_' + wandb.run.id + '.h5'))
                    # saves checkpoint to wandb
                    wandb.save(os.path.join("checkpoint", "fold_" + str(fold)), base_path = configs.wandb_params["base_path"]) 

        print()

        # Calculate and print the average validation accuracy across the current fold
        average_dsc = sum(fold_dsc) / int(configs.train_params["epoch_num"] - configs.train_params["start_epoch"])
        average_aji = sum(fold_aji) / int(configs.train_params["epoch_num"] - configs.train_params["start_epoch"])
        average_pq = sum(fold_pq) / int(configs.train_params["epoch_num"] - configs.train_params["start_epoch"])

        # Store validation results for each fold
        per_fold_dsc.append(average_dsc)
        per_fold_aji.append(average_aji)
        per_fold_pq.append(average_pq)

        result_file.writelines("\n%s\n" %(f'--- FOLD: {fold} DSC-BG: {average_dsc[0]:.4f} ({np.std(fold_dsc, axis = 0)[0]:.4f}) DSC-FG: {average_dsc[1]:.4f} ({np.std(fold_dsc, axis = 0)[1]:.4f}) AJI: {average_aji:.4f} ({np.std(fold_aji):.4f}) PQ: {average_pq:.4f} ({np.std(fold_pq):.4f})'))
        print (f'--- FOLD: {fold} DSC-BG: {average_dsc[0]:.4f} ({np.std(fold_dsc, axis = 0)[0]:.4f}) DSC-FG: {average_dsc[1]:.4f} ({np.std(fold_dsc, axis = 0)[1]:.4f}) AJI: {average_aji:.4f} ({np.std(fold_aji):.4f}) PQ: {average_pq:.4f} ({np.std(fold_pq):.4f})')
        print()

    # Calculate and print the average validation accuracy across all folds
    bg_dsc = [np.mean(per_fold_dsc, axis = 0)[0], np.std(per_fold_dsc, axis = 0)[0]]
    fg_dsc = [np.mean(per_fold_dsc, axis = 0)[1], np.std(per_fold_dsc, axis = 0)[1]]
    aji = [np.mean(per_fold_aji), np.std(per_fold_aji)]
    pq = [np.mean(per_fold_pq), np.std(per_fold_pq)]

    result_file.writelines("\n%s\n" %(f'{num_fold} FOLDS  DSC-BG: {bg_dsc[0]:.4f} ({bg_dsc[1]:.4f}) DSC-FG: {fg_dsc[0]:.4f} ({fg_dsc[1]:.4f}) AJI: {aji[0]:.4f} ({aji[1]:.4f}) PQ: {pq[0]:.4f} ({pq[1]:.4f})'))
    print (f'{num_fold} FOLDS  DSC-BG: {bg_dsc[0]:.4f} ({bg_dsc[1]:.4f}) DSC-FG: {fg_dsc[0]:.4f} ({fg_dsc[1]:.4f}) AJI: {aji[0]:.4f} ({aji[1]:.4f}) PQ: {pq[0]:.4f} ({pq[1]:.4f})')
    print()

