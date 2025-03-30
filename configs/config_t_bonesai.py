"""Config file for training and testing UNet teacher model on bones histopathologies."""

class Config(object):
    # Name the image modality as appearing on the dataset
    modality = "histo"

    # wandb 
    wandb_params = {
    "base_path": "/content",
    "project_name":"lotus",
    "entity": "isabellapoles",
    "start_epoch": 0,
    "epoch_num" : 200
    }
    
    # Training details
    fine_tuning = True

    if fine_tuning:
        run_id = "",
        num_fold = 5,
        checkpoint_fold = "fold_0",
        run_id_fine_tuning = "1xf5tx6q", 
        nr_patches = 10, 
        dataset_name = "bonesai-histo",
        lr = 0.0001, 
        batch_size = 16,
        crop_size_row = 512, 
        crop_size_col = 512 
    else:
        run_id = "",
        num_fold = 5,
        checkpoint_fold = "",
        run_id_fine_tuning = "",
        nr_patches = 50,
        dataset_name = "bonesai-microct",
        lr = 0.0001, 
        batch_size = 16,
        crop_size_row = 512, 
        crop_size_col = 512

    # Training parameters
    train_params = {
        "run_id": run_id,
        "run_id_fine_tuning": run_id_fine_tuning,
        "num_fold": num_fold,
        "checkpoint_fold": checkpoint_fold,
        "nr_patches": nr_patches, 
        "dataset_name": dataset_name,
        "optimizer":"adam",
        "adam":{
            "lr" : lr,
            "step_size" : 1000,
            "gamma" : 0.7
        },
        "batch_size": batch_size,
        "start_epoch": 0,
        "epoch_num" : 200
    }

    # Model training parameters
    model_params = {
        "input_ch": 3,
        "nr_classes": 1, 
        "probab_th": .5
    }
    
     
    # Traditional data augmentation (for train) 
    aug_params = {
        "augmentation":{
            "probability" : 1.0,
            "crop_size_row" : crop_size_row, 
            "crop_size_col" : crop_size_col
        }
    }
    
