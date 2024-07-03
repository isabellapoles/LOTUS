"""Config file for training and testing UNet teacher model on IHCs."""

class Config(object):
    # Name the image modality as appearing on the dataset
    modality = "ihc"

    # wandb 
    wandb_params = {
        "base_path": "/home",
        "project_name":"lotus",
        "entity": "user",
        "start_epoch": 0,
        "epoch_num" : 300
        }
    
    # Training details
    run_id = "",
    dataset_name = "DeepLIIF-mm",
    lr = 0.001, 
    batch_size = 15,
    crop_size_row = 512, 
    crop_size_col = 512

    # Training parameters
    train_params = {
        "run_id": run_id,
        "dataset_name": dataset_name,
        "optimizer":"adam",
        "adam":{
            "lr" : lr,
            "step_size" : 1000,
            "gamma" : 0.7
        },
        "batch_size": batch_size,
        "start_epoch": 0,
        "epoch_num" : 300
    }

    # Model training parameters
    model_params = {
        "input_ch": 3,
        "nr_classes": 3, # 1 (binary), 3 (positive and negative cells classification)
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

    # Model testing parameters
    test_params = {
        "phase": "test",
        "run_id": "2srqu2wu", # gm7d6x8v (binary), 2srqu2wu (positive and negative cells classification)
        "dataset_name": "DeepLIIF-mm",
        "img_dim_row": 512, 
        "img_dim_col": 512, 
        "batch_size": 1
    }
    
    
