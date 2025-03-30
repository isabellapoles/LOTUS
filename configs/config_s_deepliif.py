"""Config file for training and testing UNet student model on DAPI."""

class Config(object):
    # Name the image modality as appearing on the dataset
    modality_s_1 = "dapi"
    modality_s_2 = "pm"
    modality_t = "ihc"

    # wandb 
    wandb_params = {
    "base_path": "/content",
    "project_name":"lotus",
    "entity": "isabellapoles",
    "start_epoch": 0,
    "epoch_num" : 1800
    }
    
    # Training details
    run_id = "", 
    run_id_teacher = "2srqu2wu", # gm7d6x8v (binary), 2srqu2wu (positive and negative cells classification)
    run_id_student = "xxxxxxxx", 
    dataset_name_teacher = "DeepLIIF-mm",
    dataset_name_student = "DeepLIIF-mm",
    klass_to_distill = [1, 2], # [1] (binary), [1, 2] (positive and negative cells classification)
    lr = 0.001, 
    batch_size = 15,
    crop_size_row = 512, 
    crop_size_col = 512 

    # Training parameters
    train_params = {
        "run_id": run_id,
        "run_id_teacher": run_id_teacher,
        "run_id_student": run_id_student,
        "dataset_name_teacher": dataset_name_teacher,
        "dataset_name_student": dataset_name_student,
        "klass_to_distill": klass_to_distill,
        "optimizer":"adam",
        "adam":{
            "lr" : lr,
            "step_size" : 1000,
            "gamma" : 0.7
        },
        "dd_early_stopping": {
            "dd_activation" : 1, 
            "alpha_anm_min" : 1, 
            "patient" : 100
        },
        "batch_size": batch_size,
        "start_epoch": 0,
        "distill_epoch": 0, 
        "epoch_num" : 1800
    }

    # Model parameters
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
        "run_id": "eruoe5fp", # 1ft1z7f9 (binary), eruoe5fp (positive and negative cells classification)
        "dataset_name": "DeepLIIF-mm",
        "img_dim_row": 512, 
        "img_dim_col": 512, 
        "batch_size": 1
    }