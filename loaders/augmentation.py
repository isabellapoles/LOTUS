"""Data augmentation transformations."""

from albumentations import *

def albumentation_aug(p, crop_size_row, crop_size_col):
    """Define the transformation you want to apply for augmentation

        Args:
            p (int): probability of the image transformation composition 
            crop_size_row (int): row dimension of the crop
            crop_size_row (int): column dimension of the crop
    """
    return Compose([
        Resize(crop_size_row, crop_size_row, always_apply=True, p=1),
        RandomCrop(crop_size_row, crop_size_col, always_apply=True, p=1),
        CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
        RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, brightness_by_max=True, p=0.4),
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.1),
        HorizontalFlip(always_apply=False, p=0.5),
        VerticalFlip(always_apply=False, p=0.5),
        RandomRotate90(always_apply=False, p=0.5),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, interpolation=1, 
                         border_mode=4, always_apply=False, p=0.1),

    ], p=p)