"""Training UNet teacher model on bones histopathologies."""

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
from models.metrics import *
from loaders.sampler import *
from loaders.datasets import *
from models.unet import UNet
from models.loss import DSCLoss
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
parser.add_argument("--model_configs", type=str, default='config_t_bonesai.py',
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
dsc_criterion = DSCLoss() 
ce_criterion = nn.BCELoss()

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
if configs.fine_tuning:
    imgs_path_list = glob('{}*{}'.format(os.path.join(args.dataset_path, configs.train_params["dataset_name"][0], '*/tissue images/'), 'tif'))
    # Consider image groups based on patient, to perform patient-wise CV
    images_per_patient = get_dir_and_files_bonesai(os.path.join(args.dataset_path, configs.train_params["dataset_name"][0]))
    groups = sum([[i]*images_per_patient[i] for i in range (len(images_per_patient))], [])
else: 
    imgs_path_list = glob('{}*{}'.format(os.path.join(args.dataset_path, configs.train_params["dataset_name"][0], '*/tissue images/'), 'png'))
masks_path_list = glob('{}*{}'.format(os.path.join(args.dataset_path, configs.train_params["dataset_name"][0], '*/mask binary/'), 'png'))

imgs_path_list.sort(key=lambda f: int(re.sub('\D', '', f)))
masks_path_list.sort(key=lambda f: int(re.sub('\D', '', f)))

# Augmentation 
aug_params = configs.aug_params["augmentation"]
aug = albumentation_aug(aug_params["probability"], aug_params["crop_size_row"][0], aug_params["crop_size_col"])

print ('--- Training start! ---')
per_fold_dsc = []
per_fold_aji = []
per_fold_pq = []

num_fold = configs.train_params["num_fold"][0]

with open(os.path.join(CHECKPOINT_PATH,  wandb.run.id + "_result.txt"), "a") as result_file:
    for fold in range(num_fold):
        
        fold_dsc = []
        fold_aji = []
        fold_pq = []

        # Create a new instance of the model for each fold
        model_ft = UNet(input_channels = configs.model_params["input_ch"], nclasses = configs.model_params["nr_classes"]) 

        if configs.fine_tuning: 
            # Create data handler for the current fold
            nr_patches = configs.train_params["nr_patches"][0]
            data_handler = BonesAI(imgs_path_list, masks_path_list, groups, fold, aug, num_fold, nr_patches, aug_params["crop_size_row"][0], aug_params["crop_size_col"], configs.modality, SEED)
            # Load the model to be finetuned on BonesAI
            model_ft.load_state_dict(torch.load(os.path.join(args.checkpoint_path, configs.train_params["run_id_fine_tuning"][0], configs.train_params["checkpoint_fold"][0], configs.train_params["run_id_fine_tuning"][0] + '.h5')))
            model = model_ft.to(device)
            # Split the dataset into training and validation subsets
            val_data, val_idxs, train_data, train_idxs = data_handler.get_splits() # change train and val for 1vs5
            # Create data loaders using the batch_size from the Config class
            dataloaders = {
            'train': DataLoader(data_handler, 
                                    batch_sampler=BonesAIDataSampler((data_handler.__dataframe__()), train_idxs, configs.train_params["batch_size"][0]),
                                    worker_init_fn=np.random.seed(SEED)),
            'val': DataLoader(data_handler, 
                                    batch_sampler=BonesAIDataSampler((data_handler.__dataframe__()), val_idxs, configs.train_params["batch_size"][0]),
                                    worker_init_fn=np.random.seed(SEED))}
        else: 
            # Create data handler for the current fold
            data_handler = Nuinsseg(imgs_path_list, masks_path_list, fold, aug, num_fold, SEED)
            model = model_ft.to(device)
            # Split the dataset into training and validation subsets
            train_data, val_data = data_handler.get_splits()
            # Create data loaders using the batch_size from the Config class
            dataloaders = {
            'train': DataLoader(train_data, batch_size=configs.train_params["batch_size"][0], shuffle=True), 
            'val': DataLoader(val_data, batch_size=configs.train_params["batch_size"][0]),
            }

        # Define a new optimizer for each fold to reset the model parameters
        if configs.train_params["optimizer"] == "adam":
            optimizer_params = configs.train_params["adam"]
            optimizer = optim.Adam(model.parameters(), lr=optimizer_params["lr"][0])
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=optimizer_params["step_size"], gamma=optimizer_params["gamma"])
        
        # Make directory per run & fold
        os.makedirs(os.path.join(CHECKPOINT_PATH, "fold_" + str(fold)), exist_ok=True)

        wandb.watch(model, log = "all")

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
                        masks_pred = (masks_pred_prob > configs.model_params["probab_th"]).float()

                        # Copy and bring to cpu
                        masks_set_pred = masks_pred.cpu().numpy()
                        masks_set_gt = masks.cpu().numpy()

                        # Calculate the loss using BCEWithLogitsLoss
                        loss = dsc_criterion(masks_pred, masks) + .5*ce_criterion (masks_pred_prob, masks)

                        # Validation step and image prediction wandb logging
                        if (((phase == 'val') & (epoch%25 == 0)) | ((phase == 'val') & (epoch == (configs.train_params["epoch_num"]-1)))):
                            log_count = 0
                            if log_count < configs.train_params["batch_size"][0]:
                                log_val_predictions(imgs, masks, masks_pred, val_table, glob_idx)
                                log_count += 1
                                glob_idx += 1
                        
                        # Segmentation metrics computation
                        for pred in range(len(masks_pred_prob)): 
                            masks_set_gt[pred] = remap_label(masks_set_gt[pred])
                            
                            output_raw_0 = np.squeeze(masks_set_pred[pred].astype(np.uint8))
                            output_raw = skimage.morphology.label(output_raw_0)
                            output_raw = remove_small_objects(output_raw, min_size=50, connectivity=2)
                            output_raw = remap_label(output_raw)

                            DSC.append(dice(np.squeeze(masks_set_gt[pred].astype(np.uint8)), output_raw, labels = [0,1], include_zero=True))
                            AJI.append(fast_aji(np.squeeze(masks_set_gt[pred].astype(np.uint8)), output_raw))
                            PQ.append(fast_pq(np.squeeze(masks_set_gt[pred].astype(np.uint8)), output_raw)[0][2])

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
                wandb.log({phase + " loss f_" + str(fold): average_loss, 'epoch': epoch})
                wandb.log({phase + " BG dsc f_" + str(fold): average_dsc[0], 'epoch': epoch})
                wandb.log({phase + " FG dsc f_" + str(fold): average_dsc[1], 'epoch': epoch})        
                wandb.log({phase + " aji f_" + str(fold): average_aji, 'epoch': epoch})
                wandb.log({phase + " pq f_" + str(fold): average_pq, 'epoch': epoch})
                
                print (f'--- {phase} Loss: {average_loss:.4f} DSC-BG: {average_dsc[0]:.4f} DSC-FG: {average_dsc[1]:.4f} AJI: {average_aji:.4f} PQ: {average_pq:.4f}')

                if phase == 'val': 
                    # Store validation results for the current fold
                    fold_dsc.append(average_dsc)
                    fold_aji.append(average_aji)
                    fold_pq.append(average_pq)

                    if ((epoch%25 == 0) | (epoch == (configs.train_params["epoch_num"]-1))):
                        wandb.log({"Validation predictions epoch: " + str (epoch) + " fold: " + str(fold) : val_table})

                # Saves checkpoint to disk
                if ((epoch%50 == 0) | (epoch == (configs.train_params["epoch_num"]-1))): 
                    torch.save(model.state_dict(), os.path.join(CHECKPOINT_PATH, "fold_" + str(fold), str(epoch) + '_' + wandb.run.id + '.h5'))
                    # Saves checkpoint to wandb
                    wandb.save(os.path.join(CHECKPOINT_PATH, "fold_" + str(fold)), base_path = configs.wandb_params["base_path"]) 

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
