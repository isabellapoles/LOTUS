"""Config file for training and testing UNet student model on SR-microCT."""

class Config(object):
    # Name the image modality as appearing on the dataset
    modality_s = "microct"
    modality_t = "histo"

    # wandb 
    wandb_params = {
    "base_path": "/content",
    "project_name":"lotus",
    "entity": "isabellapoles",
    "start_epoch": 0,
    "epoch_num" : 200
    }
    
    # Training details
    num_fold = 5
    run_id = "", 
    run_id_teacher = "v8hmob38", 
    checkpoint_fold_teacher = "fold_4",
    run_id_student = "1xf5tx6q", 
    checkpoint_fold_student = "fold_0",
    nr_patches_teacher = 10, 
    nr_patches_student = 50,
    dataset_name_teacher = "bonesai-histo",
    dataset_name_student = "bonesai-microct",
    klass_to_distill = [1],

    lr = 0.001, 
    batch_size = 1, # Ex. batch_size for demo, original one = 16
    crop_size_row = 512, 
    crop_size_col = 512 

    # Training parameters
    train_params = {
        "num_fold": num_fold,
        "run_id": run_id,
        "run_id_teacher": run_id_teacher,
        "checkpoint_fold_teacher": checkpoint_fold_teacher,
        "run_id_student": run_id_student,
        "checkpoint_fold_student": checkpoint_fold_student,
        "nr_patches_teacher": nr_patches_teacher, 
        "nr_patches_student": nr_patches_student, 
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
            "patient" : 24
        },
        "batch_size": batch_size,
        "start_epoch": 0,
        "distill_epoch": 0, 
        "epoch_num" : 200
    }

    # Model parameters
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

    # Model testing parameters
    test_params = {
        "phase": "test",
        "run_id": "1ur01ynw",
        "checkpoint_fold": "fold_3",
        "nr_patches": 50,
        "probab_th": .5,
        "dataset_name": "bonesai-microct-test",
        "img_dim_row": 512, 
        "img_dim_col": 512, 
        "batch_size": 1
    }
    
