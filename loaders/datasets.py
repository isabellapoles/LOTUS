"""Objects to create the NuInsSeg-BonesAI-DeepLIIF monomodal and multimodal datasets."""

import random
import cv2 as cv
import numpy as np
import pandas as pd

import torch

from enum import Enum
from PIL import Image
from skimage import filters, transform
from sklearn.model_selection import KFold, GroupKFold

from torchvision import transforms
from torch.utils.data import Dataset, Subset


class Nuinsseg(Dataset):
    """Object representing the nuinsseg train/val datasets for k-fold cross validation."""

    def __init__(self, imgs_path_list, masks_path_list, current_fold, aug, num_fold, seed):
        super(Nuinsseg, self).__init__()
        """ Class representing an H&E image + nuclei mask segmentation. 
        
        Args:
            imgs_path_list (list): list of the image paths 
            masks_path_list (list): list of the mask paths 
            current_fold (int): number of the current fold during k-fold cross validation 
            aug (str): whether is required or not performing data augmentation
            num_fold (int): number of the k folds for k-fold cross validation 
            seed (int): seed to ensure reproducibility
        """
        self.imgs_path_list = imgs_path_list
        self.masks_path_list = masks_path_list
        self.current_fold = current_fold
        self.num_fold = num_fold
        self.aug = aug
        
        # Dataframe for easier access to the datasets paths
        self.df = pd.DataFrame(
                                list(zip(self.imgs_path_list, self.masks_path_list)),
                                columns=["imgs_path", "masks_path"]
                            )

        # Use KFold to split the dataset into 'num_fold' folds
        self.kf = KFold(n_splits=num_fold, shuffle=True, random_state=seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Returns a Dict (img, mask) correspond to batch #idx."""
        
        img_path = self.df.loc[idx, "imgs_path"]
        mask_path = self.df.loc[idx, "masks_path"]

        # Read image
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Read binary mask
        mask_path = self.masks_path_list[idx]
        mask = cv.imread(mask_path, -1)
        mask_temp = np.zeros((mask.shape[0], mask.shape[1]))
        mask_temp[mask == 255] = 1

        # Data augmenttaion 
        if ((self.aug != None) & (idx in self.train_indices)):
            augmented = self.aug(image = img, mask = mask_temp)
            x_img = augmented['image']
            x_mask = augmented['mask']
            x_img = x_img/255
        else:
            x_mask = mask_temp
            x_img = img/255    

        # To torch tensor
        img_tensor = transforms.ToTensor()(np.float32(x_img))
        mask_tensor = transforms.ToTensor()(np.float32(x_mask))

        return {'img': img_tensor, 'mask': mask_tensor}

    def get_splits(self):
        """
        Splits the dataset into training and validation subsets.

        Returns:
            tuple: A tuple containing the training and validation subsets.
        """
        # Dataset indexes split 
        fold_data = list(self.kf.split(self.df))
        self.train_indices, self.val_indices = fold_data[self.current_fold]

        # Datasets objects split
        train_data = self._get_subset(self.train_indices)
        val_data = self._get_subset(self.val_indices)

        return train_data, val_data

    def _get_subset(self, indices):
        """
        Returns a Subset of the dataset at the given indices.

        Args:
            indices (list): A list of indices specifying the subset of the dataset to return.

        Returns:
            Subset: A Subset of the dataset at the given indices.
        """
        return Subset(self, indices)


class BonesAI(Dataset): 
    """Object representing the bone H&E or microct propretary monomodal train/val datasets for k-fold cross validation."""

    def __init__(self, imgs_path_list, masks_path_list, groups, current_fold, aug, num_fold, nr_patches, img_dim_row, img_dim_col, modality, seed):
        """ Class representing a bone H&E or microct image + the corresponding segmentation. 
        
        Args:
            imgs_path_list (list): list of the image paths 
            masks_path_list (list): list of the mask paths 
            groups (list): list of indexes corresponding to patients images for fair stratification
            current_fold (int): number of the current fold during k-fold cross validation
            aug (str): whether is required or not performing data augmentation
            num_fold (int): number of the k folds for k-fold cross validation 
            nr_patches (int): number of patches you want to extract from the WSI
            img_dim_row (int): image row dimension
            img_dim_col (int): image col dimension 
            modality (str): 'histo' or 'microct' based on what your are processing
            seed (int): seed to ensure reproducibility
        """
        self.imgs_path_list = imgs_path_list
        self.masks_path_list = masks_path_list
        self.current_fold = current_fold
        self.nr_patches = nr_patches
        self.num_fold = num_fold
        self.img_dim_row = img_dim_row
        self.img_dim_col = img_dim_col
        self.groups = groups
        self.modality = modality
        self.aug = aug

        # Dataframe for easier access to the datasets paths
        self.df = pd.DataFrame(
                                list(zip(self.imgs_path_list, self.masks_path_list)),
                                columns=["imgs_path", "masks_path"]
                            )
        # Use KFold to split the dataset into 'num_fold' folds
        self.kf = GroupKFold(n_splits=num_fold) # , shuffle=True, random_state=seed

    def __dataframe__ (self):
            return self.df

    def __len__(self):
            return len(self.df.index)

    def __getitem__(self, idx):
        """Returns a Dict (img, mask) correspond to batch #idx."""

        img_b = []
        tiss_b = []
        mask_b = []

        img_path = self.df.loc[idx, "imgs_path"]
        mask_path = self.df.loc[idx, "masks_path"]

        # Read image
        img = cv.imread(img_path, -1)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Read binary mask
        mask_path = self.masks_path_list[idx]
        mask = cv.imread(mask_path, -1)

        # Patch size depending on the input image size and number of patches chosen 
        patch_size = max(int (np.array(img).shape[0] / self.nr_patches), int (np.array(img).shape[1] / self.nr_patches))

        # Tissue segmentation
        val = filters.threshold_otsu(np.asarray(img)[:, :, 1])
        if self.modality == 'histo': 
            seg_tissue = np.uint8((np.asarray(img[:, :, 1]) < val)*255)
        if self.modality == 'microct':
            seg_tissue = np.uint8((np.asarray(img[:, :, 1]) > val)*255)
        
        # Image patches extraction
        extractor = PatchExtractor(img=Image.fromarray(img), patch_size=patch_size, stride=patch_size)
        patches_img = extractor.extract_img_patches()
        # Tissue patches extraction
        extractor = PatchExtractor(img=Image.fromarray(seg_tissue), patch_size=patch_size, stride=patch_size)
        patches_tiss = extractor.extract_img_patches()
        # Segmentation patches extraction
        extractor = PatchExtractor(img=Image.fromarray(mask), patch_size=patch_size, stride=patch_size)
        patches_seg = extractor.extract_img_patches()

        # Patch selection (selection of the patches with a proportion of tissue higher than 90% than the total number of pizels in a patch)
        for i in range(len(patches_img)):
            if (((np.count_nonzero(np.asarray(patches_tiss[i]))/(np.asarray(patches_tiss[i]).shape[0]*np.asarray(patches_tiss[i]).shape[1])) > .9)): 
                img_b.append(patches_img[i])
                mask_b.append(patches_seg[i])

        # Stack of WSI patches 
        img_b = np.stack(img_b, axis=0)
        mask_b = np.stack (mask_b, axis=0) 
        mask_temp = np.zeros((mask_b.shape[0], mask_b.shape[1], mask_b.shape[2]))
        mask_temp[mask_b == 255] = 1

        # Data augmenttaion
        if ((self.aug != None) & (idx in self.train_indices)): 
            ind = random.randrange(0, img_b.shape[0], 1)
            augmented = self.aug(image = img_b[ind], mask = mask_temp[ind])
            x_img = augmented['image']
            x_mask = augmented['mask']
            x_img = x_img/255
        else:
            # Just image resize during validation
            ind = random.randrange(0, img_b.shape[0], 1) 
            x_mask = transform.resize(mask_temp[ind], (self.img_dim_row, self.img_dim_col), order = 0)
            x_img = transform.resize(img_b[ind], (self.img_dim_row, self.img_dim_col))    

        # To torch tensor
        img_tensor = transforms.ToTensor()(np.float32(x_img))
        mask_tensor = transforms.ToTensor()(np.float32(x_mask))

        return {'img': img_tensor, 'mask': mask_tensor}

    def get_splits(self):
        """
        Splits the dataset into training and validation subsets.

        Returns:
            tuple: A tuple containing the training and validation subsets.
        """

        # Dataset indexes split 
        fold_data = list(self.kf.split(self.df, groups = self.groups))
        self.train_indices, self.val_indices = fold_data[self.current_fold]

        # Datasets objects split
        train_data = self._get_subset(self.train_indices)
        val_data = self._get_subset(self.val_indices)

        return train_data, self.train_indices, val_data, self.val_indices

    def _get_subset(self, indices):
        """
        Returns a Subset of the dataset at the given indices.

        Args:
            indices (list): A list of indices specifying the subset of the dataset to return.

        Returns:
            Subset: A Subset of the dataset at the given indices.
        """
        return Subset(self, indices)


class BonesAITest(Dataset): 
    """Object representing the bone H&E or microct propretary monomodal test dataset."""

    def __init__(self, imgs_path_list, masks_path_list, nr_patches, img_dim_row, img_dim_col, seed, modality):
        """ Class representing a bone H&E or microct image + the corresponding segmentation. 
        
        Args:
            imgs_path_list (list): list of the image paths 
            masks_path_list (list): list of the mask paths 
            num_fold (int): number of the k folds for k-fold cross validation 
            nr_patches (int): number of patches you want to extract from the WSI
            img_dim_row (int): image row dimension
            img_dim_col (int): image col dimension 
            seed (int): seed to ensure reproducibility
            modality (str): 'histo' or 'microct' based on what your are processing
        """
        self.imgs_path_list = imgs_path_list
        self.masks_path_list = masks_path_list
        self.nr_patches = nr_patches
        self.modality = modality
        self.img_dim_row = img_dim_row
        self.img_dim_col = img_dim_col
        self.seed = seed
        
        # Dataframe for easier access to the datasets paths
        self.df = pd.DataFrame(
                                list(zip(self.imgs_path_list, self.masks_path_list)),
                                columns=["imgs_path", "masks_path"]
                            )

    def __dataframe__ (self):
            return self.df

    def __len__(self):
            return len(self.df.index)

    def __getitem__(self, idx):
        """Returns a Dict (img, mask) correspond to batch #idx."""

        img_b = []
        tiss_b = []
        mask_b = []

        img_path = self.df.loc[idx, "imgs_path"]
        mask_path = self.df.loc[idx, "masks_path"]

        # Read image
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Read binary mask
        mask_path = self.masks_path_list[idx]
        mask = cv.imread(mask_path, -1)

        # Patch size depending on the input image size and number of patches chosen 
        patch_size = max(int (np.array(img).shape[0] / self.nr_patches), int (np.array(img).shape[1] / self.nr_patches))

        # Tissue segmentation
        val = filters.threshold_otsu(np.asarray(img)[:, :, 1])
        if self.modality == 'histo': 
            seg_tissue = np.uint8((np.asarray(img[:, :, 1]) < val)*255)
        if self.modality == 'microct':
            seg_tissue = np.uint8((np.asarray(img[:, :, 1]) > val)*255)
        
        # Image patches extraction
        extractor = PatchExtractor(img=Image.fromarray(img), patch_size=patch_size, stride=patch_size)
        patches_img = extractor.extract_img_patches()
        # Tissue patches extraction
        extractor = PatchExtractor(img=Image.fromarray(seg_tissue), patch_size=patch_size, stride=patch_size)
        patches_tiss = extractor.extract_img_patches()
        # Segmentation patches extraction
        extractor = PatchExtractor(img=Image.fromarray(mask), patch_size=patch_size, stride=patch_size)
        patches_seg = extractor.extract_img_patches()

        # Patch selection (selection of the patches with a proportion of tissue higher than 90% than the total number of pizels in a patch)
        for i in range(len(patches_img)):
            if (((np.count_nonzero(np.asarray(patches_tiss[i]))/(np.asarray(patches_tiss[i]).shape[0]*np.asarray(patches_tiss[i]).shape[1])) > .9)):
                img_b.append(patches_img[i])
                mask_b.append(patches_seg[i])
        
        # Stack of original size WSI patches
        img_b = np.stack(img_b, axis=0)
        mask_b = np.stack (mask_b, axis=0) 
        mask_temp = np.zeros((mask_b.shape[0], mask_b.shape[1], mask_b.shape[2]))
        mask_temp[mask_b == 255] = 1

        # Prepare resized patches for each WSI to be tested 
        img_test = []
        mask_test = []
        for ind in range(img_b.shape[0]):
            img_test.append(transform.resize(img_b[ind], (self.img_dim_row, self.img_dim_col))) 
            mask_test.append(transform.resize(mask_temp[ind], (self.img_dim_row, self.img_dim_col), order = 0)) 
        x_img = np.stack(img_test, axis=0)
        x_mask = np.stack(mask_test, axis=0)

        # To torch tensor
        img_tensor = torch.stack([transforms.ToTensor()(np.float32(patch)) for patch in x_img])
        mask_tensor = torch.stack([transforms.ToTensor()(np.float32(patch)) for patch in x_mask])

        return {'img': img_tensor, 'mask': mask_tensor}


class SubtypeEnumBonesAI(Enum):
    """Object representing the bone H&E & microct propretary datasets labels."""

    # Healthy samples
    HTY = 0
    # Osteporotic samples
    OSTEO = 1
    # COVID19 samples
    CVD = 2


class MultiBonesAI(Dataset):
    """Object representing the bone H&E & microct propretary multimodal train/val datasets for k-fold cross validation."""
    def __init__(self, imgs_h_path_list, masks_h_path_list, imgs_ct_path_list, masks_ct_path_list, groups_h, groups_ct, current_fold, aug_h, aug_ct, num_fold, nr_patches_h, nr_patches_ct, img_dim_row, img_dim_col, seed):
        super(MultiBonesAI, self).__init__()
        """ Class representing a bone H&E or microct image + the corresponding segmentation. 
        
        Args:
            imgs_h_path_list (list): list of the histo image paths 
            masks_h_path_list (list): list of the histo mask paths
            imgs_ct_path_list (list): list of the microct image paths 
            masks_ct_path_list (list): list of the microct mask paths 
            groups_h (list): list of indexes corresponding to patients histo images for fair stratification
            groups_ct (list): list of indexes corresponding to patients histo images for fair stratification
            current_fold (int): number of the current fold during k-fold cross validation
            aug_h (str): whether is required or not performing histo data augmentation
            aug_ct (str): whether is required or not performing microctdata augmentation
            num_fold (int): number of the k folds for k-fold cross validation 
            nr_patches_h (int): number of patches you want to extract from the histo WSI
            nr_patches_ct (int): number of patches you want to extract from the microct WSI
            img_dim_row (int): image row dimension
            img_dim_col (int): image col dimension 
            seed (int): seed to ensure reproducibility
        """

        self.imgs_h_path_list = imgs_h_path_list
        self.masks_h_path_list = masks_h_path_list
        self.imgs_ct_path_list = imgs_ct_path_list
        self.masks_ct_path_list = masks_ct_path_list
        self.current_fold = current_fold
        self.nr_patches_h = nr_patches_h
        self.nr_patches_ct = nr_patches_ct
        self.img_dim_row = img_dim_row
        self.img_dim_col = img_dim_col
        self.num_fold = num_fold
        self.groups_h = groups_h
        self.groups_ct = groups_ct
        self.aug_h = aug_h
        self.aug_ct = aug_ct

        # Identify image classes for histo and microct WSI depending on the sample ID 
        image_labels_h = [0 for _ in range(len(self.imgs_h_path_list))]
        image_labels_ct = [0 for _ in range(len(self.imgs_ct_path_list))]

        for idx, filename in enumerate(self.imgs_h_path_list):
            if (('T1_S26' in filename) | ('FH5_S36' in filename)):
                label = SubtypeEnumBonesAI['HTY'].value    
            if (('T4_S31' in filename) | ('T2_S53' in filename) | ('FH2_S77' in filename) | ('FH4_S37' in filename)):
                label = SubtypeEnumBonesAI['OSTEO'].value
            if (('TC2_S13' in filename) | ('TC3_S7' in filename) | ('TC2_S42' in filename)):
                label = SubtypeEnumBonesAI['CVD'].value
            image_labels_h[idx] = label

        for idx, filename in enumerate(self.imgs_ct_path_list):
            if (('T1_S26' in filename) | ('FH5_S36' in filename)):
                label = SubtypeEnumBonesAI['HTY'].value    
            if (('T4_S31' in filename) | ('T2_S53' in filename) | ('FH2_S77' in filename) | ('FH4_S37' in filename)):
                label = SubtypeEnumBonesAI['OSTEO'].value
            if (('TC2_S13' in filename) | ('TC3_S7' in filename) | ('TC2_S42' in filename)):
                label = SubtypeEnumBonesAI['CVD'].value
            image_labels_ct[idx] = label

        # Dataframe for easier access to the datasets paths and labels
        self.df_h = pd.DataFrame(
                                list(zip(self.imgs_h_path_list, self.masks_h_path_list, image_labels_h)),
                                columns=["imgs_path", "masks_path", "label"]
                            )
        self.df_ct = pd.DataFrame(
                                list(zip(self.imgs_ct_path_list, self.masks_ct_path_list, image_labels_ct)),
                                columns=["imgs_path", "masks_path", "label"]
                            )
        # Use KFold to split the dataset into 'num_fold' folds
        self.kf = GroupKFold(n_splits=num_fold) 

    def __dataframe__ (self):
        return self.df_ct

    def __len__(self):
        return len(self.imgs_ct_path_list)

    def __getitem__(self, idx):
        """Returns a Dict (img_ct, mask_ct, img_h, mask_h) correspond to batch #idx."""

        img_ct_path = self.df_ct.loc[idx, "imgs_path"]
        mask_ct_path = self.df_ct.loc[idx, "masks_path"]

        # Read microct image
        img_ct = cv.imread(img_ct_path, -1)
        img_ct = cv.cvtColor(img_ct, cv.COLOR_BGR2RGB)

        # Read microct binary mask
        mask_ct = cv.imread(mask_ct_path, -1)

        # Read microct label
        klass_ct = self.df_ct.loc[idx, "label"]

        # Determine the index for the teacher histo to infer based on the student label
        idx_set_h = list(np.where(np.asarray(self.df_h.loc[:, "label"]) == klass_ct)[0])
        idx_h = random.choice(idx_set_h)
        img_h_path = self.df_h.loc[idx_h, "imgs_path"]
        mask_h_path = self.df_h.loc[idx_h, "masks_path"]

        # Read histo image
        img_h = cv.imread(img_h_path, -1)
        img_h = cv.cvtColor(img_h, cv.COLOR_BGR2RGB)

        # Read histo binary mask
        mask_h_path = self.masks_h_path_list[idx_h]
        mask_h = cv.imread(mask_h_path, -1)

        # Extract and select patches 
        img_h_b, mask_h_b = extract_and_select_patches(img_h, mask_h, self.nr_patches_h, 'histo')
        img_ct_b, mask_ct_b = extract_and_select_patches(img_ct, mask_ct, self.nr_patches_ct, 'ct')

        mask_h_temp = np.zeros((mask_h_b.shape[0], mask_h_b.shape[1]))
        mask_h_temp[mask_h_b == 255] = 1

        mask_ct_temp = np.zeros((mask_ct_b.shape[0], mask_ct_b.shape[1]))
        mask_ct_temp[mask_ct_b == 255] = 1

        # Microct data augmentation
        if ((self.aug_ct != None) & (idx in self.train_ct_indices)):
            augmented_ct = self.aug_ct(image = img_ct_b, mask = mask_ct_temp)
            x_ct_img = augmented_ct['image']
            x_ct_mask = augmented_ct['mask']
            x_ct_img = x_ct_img/255
        if ((self.aug_ct == None) | (idx not in self.train_ct_indices)): 
            x_ct_mask = transform.resize(mask_ct_temp, (self.img_dim_row[0], self.img_dim_col), order = 0)
            x_ct_img = transform.resize(img_ct_b, (self.img_dim_row[0], self.img_dim_col))
        
        # Histo data augmentation (it is avoided during the student training)
        if ((self.aug_h != None)):
            augmented_h = self.aug_h(image = img_h_b, mask = mask_h_temp)
            x_h_img = augmented_h['image']
            x_h_mask = augmented_h['mask']
            x_h_img = x_h_img/255
        if ((self.aug_h == None)):
            x_h_mask = transform.resize(mask_h_temp, (self.img_dim_row[0], self.img_dim_col), order = 0)
            x_h_img = transform.resize(img_h_b, (self.img_dim_row[0], self.img_dim_col))

        # To torch tensor
        img_h_tensor = transforms.ToTensor()(np.float32(x_h_img))
        mask_h_tensor = transforms.ToTensor()(np.float32(x_h_mask))
        img_ct_tensor = transforms.ToTensor()(np.float32(x_ct_img))
        mask_ct_tensor = transforms.ToTensor()(np.float32(x_ct_mask))
 
        return {'img_ct': img_ct_tensor, 'mask_ct': mask_ct_tensor, 'img_h': img_h_tensor, 'mask_h': mask_h_tensor}
    
    def get_splits(self):
        """
        Splits the dataset into training and validation subsets.

        Returns:
            tuple: A tuple containing the training and validation subsets.
        """
        # Microct dataset indexes split
        fold_ct_data = list(self.kf.split(self.df_ct, groups = self.groups_ct))
        self.train_ct_indices, self.val_ct_indices = fold_ct_data[self.current_fold]
        # Histo dataset indexes split
        fold_h_data = list(self.kf.split(self.df_h, groups = self.groups_h))
        self.train_h_indices, self.val_h_indices = fold_h_data[4] #self.current_fold

        # Microct dataset objects split
        train_ct_data = self._get_subset(self.train_ct_indices)
        val_ct_data = self._get_subset(self.val_ct_indices)
        # Histo dataset objects split
        val_h_data = self._get_subset(self.val_h_indices)

        return train_ct_data, self.train_ct_indices, val_ct_data, self.val_ct_indices, val_h_data, self.val_h_indices

    def _get_subset(self, indices):
        """
        Returns a Subset of the dataset at the given indices.

        Args:
            indices (list): A list of indices specifying the subset of the dataset to return.

        Returns:
            Subset: A Subset of the dataset at the given indices.
        """
        return Subset(self, indices)


class DeepLIIF(Dataset): 
    """Object representing the IHC DeepLIIF monomodal train/val datasets."""
    def __init__(self, imgs_path_list_1, masks_path_list_1, imgs_path_list_2, aug, img_dim_row, img_dim_col, classes, seed):
        """ Class representing a DeepLIIF IHC image + the corresponding segmentation. 
        
        Args:
            imgs_path_list (list): list of the image paths 
            masks_path_list (list): list of the mask paths
            aug (str): whether is required or not performing data augmentation
            img_dim_row (int): image row dimension
            img_dim_col (int): image col dimension
            classes (int): the number of masks labels (1: binary, 2: positive + negative) 
            seed (int): seed to ensure reproducibility
        """
        self.imgs_path_list_1 = imgs_path_list_1
        self.masks_path_list = masks_path_list_1
        self.imgs_path_list_2 = imgs_path_list_2
        self.img_dim_row = img_dim_row
        self.img_dim_col = img_dim_col
        self.classes = classes
        self.aug = aug
        

        # Dataframe for easier access to the datasets paths
        self.df_1 = pd.DataFrame(
                                list(zip(self.imgs_path_list_1, self.masks_path_list)),
                                columns=["imgs_path", "masks_path"]
                            )
        if self.imgs_path_list_2 != None: 
            self.df_2 = pd.DataFrame(
                            list(zip(self.imgs_path_list_2)),
                            columns=["imgs_path"]
                        )

    def __dataframe__ (self):
            return self.df_1

    def __len__(self):
            return len(self.df_1.index)

    def __getitem__(self, idx):
        """Returns a Dict (img, mask) correspond to batch #idx."""

        img_path_1 = self.df_1.loc[idx, "imgs_path"]
        mask_path = self.df_1.loc[idx, "masks_path"]

        # Read image
        img_1 = cv.imread(img_path_1, -1)
        img = cv.cvtColor(img_1, cv.COLOR_BGR2RGB)
        if self.imgs_path_list_2 != None: 
            img_path_2 = self.df_2.loc[idx, "imgs_path"]
            img_2 = cv.imread(img_path_2, -1)
            img_2 = cv.cvtColor(img_2, cv.COLOR_BGR2RGB)
            img = np.maximum(img, img_2)

        # Read mask and relabelling
        mask_path = self.masks_path_list[idx]
        mask = cv.imread(mask_path, -1)
        mask_class = np.zeros ((mask.shape[0], mask.shape[1]))
        if self.classes == 1: 
            mask_class [mask[:, :, 0]==255] = 1.
            mask_class [mask[:, :, 2]==255] = 1.
        else: 
            mask_class [mask[:, :, 0]==255] = 1.
            mask_class [mask[:, :, 2]==255] = 2.

        # Data augmentation
        if self.aug != None:
            augmented = self.aug(image = img, mask = mask_class)
            x_img = augmented['image']
            x_mask = augmented['mask'] 
            x_img = x_img/255
        else:
            # Just image resize during validation 
            x_img = transform.resize(img, (self.img_dim_row[0], self.img_dim_col[0]))    
            x_mask = transform.resize(mask_class, (self.img_dim_row[0], self.img_dim_col[0]), order = 0)

        # To torch tensor
        img_tensor = transforms.ToTensor()(np.float32(x_img))
        mask_tensor = transforms.ToTensor()(np.float32(x_mask))

        return {'img': img_tensor, 'mask': mask_tensor}


class SubtypeEnumDeepLIIF(Enum):
    """Object representing the organ DeepLIIF datasets labels."""

    # Bladder
    BLD = 0
    # Lung
    LNG = 1


class MultiDeepLIIF(Dataset):
    """Object representing the IHC and DAPI DeepLIIF multimodal train/val datasets."""
    def __init__(self, imgs_ihc_path_list, masks_ihc_path_list, imgs_dapi_path_list, imgs_pm_path_list , masks_dapi_path_list, aug_ihc, aug_dapi, img_dim_row, img_dim_col, classes, seed):
        super(MultiDeepLIIF, self).__init__()
        """ Class representing a IHC and DAPI image + the corresponding segmentation. 
        
        Args:
            imgs_ihc_path_list (list): list of the IHC image paths 
            masks_ihc_path_list (list): list of the IHC mask paths
            imgs_dapi_path_list (list): list of the DAPI image paths 
            imgs_pm_path_list (list): list of the PM image paths 
            masks_dapi_path_list (list): list of the DAPI mask paths 
            aug_ihc (str): whether is required or not performing IHC data augmentation
            aug_dapi (str): whether is required or not performing DAPI data augmentation
            img_dim_row (int): image row dimension
            img_dim_col (int): image col dimension
            classes (int): the number of masks labels (1: binary, 2: positive + negative) 
            seed (int): seed to ensure reproducibility
        """

        self.imgs_ihc_path_list = imgs_ihc_path_list
        self.masks_ihc_path_list = masks_ihc_path_list
        self.imgs_dapi_path_list = imgs_dapi_path_list
        self.imgs_pm_path_list = imgs_pm_path_list
        self.masks_dapi_path_list = masks_dapi_path_list
        self.img_dim_row = img_dim_row
        self.img_dim_col = img_dim_col

        self.aug_ihc = aug_ihc
        self.aug_dapi = aug_dapi

        self.classes = classes

        # Identify image classes for IHC and DAPI WSI depending on the sample ID 
        image_labels_ihc = [0 for _ in range(len(self.imgs_ihc_path_list))]
        image_labels_dapi = [0 for _ in range(len(self.imgs_dapi_path_list))]

        for idx, filename in enumerate(self.imgs_ihc_path_list):
            if ('Bladder' in filename):
                label = SubtypeEnumDeepLIIF['BLD'].value     
            if ('Lung' in filename):
                label = SubtypeEnumDeepLIIF['LNG'].value
            image_labels_ihc[idx] = label

        for idx, filename in enumerate(self.imgs_dapi_path_list):
            if ('Bladder' in filename):
                label = SubtypeEnumDeepLIIF['BLD'].value    
            if ('Lung' in filename):
                label = SubtypeEnumDeepLIIF['LNG'].value  
            image_labels_dapi[idx] = label

        # Dataframe for easier access to the datasets paths
        self.df_ihc = pd.DataFrame(
                                list(zip(self.imgs_ihc_path_list, self.masks_ihc_path_list, image_labels_ihc)),
                                columns=["imgs_path", "masks_path", "label"]
                            )
        self.df_dapi = pd.DataFrame(
                                list(zip(self.imgs_dapi_path_list, self.masks_dapi_path_list, image_labels_dapi)),
                                columns=["imgs_path", "masks_path", "label"]
                            )
        self.df_pm = pd.DataFrame(
                        list(zip(self.imgs_pm_path_list, self.masks_dapi_path_list, image_labels_dapi)),
                        columns=["imgs_path", "masks_path", "label"]
                    )

    def __dataframe__ (self):
        return self.df_dapi

    def __len__(self):
        return len(self.imgs_dapi_path_list)

    def __getitem__(self, idx):
        """Returns a Dict (img_dapi, mask_dapi, img_ihc, mask_ihc) correspond to batch #idx."""

        img_dapi_path = self.df_dapi.loc[idx, "imgs_path"]
        mask_dapi_path = self.df_dapi.loc[idx, "masks_path"]
        img_pm_path = self.df_pm.loc[idx, "imgs_path"]
        klass_dapi = self.df_dapi.loc[idx, "label"]

        # Read DAPI image
        img_dapi = cv.imread(img_dapi_path, -1)
        img_dapi = cv.cvtColor(img_dapi, cv.COLOR_BGR2RGB)
        img_pm = cv.imread(img_pm_path, -1)
        img_pm = cv.cvtColor(img_pm, cv.COLOR_BGR2RGB)
        img_merge = np.maximum(img_dapi, img_pm)

        # Read DAPI mask
        mask_dapi = cv.imread(mask_dapi_path, -1)

        # Determine the index for the teacher IHC to infer based on the student label
        idx_set_ihc = list(np.where(np.asarray(self.df_ihc.loc[:, "label"]) == klass_dapi)[0])
        idx_ihc = random.choice(idx_set_ihc)
        img_ihc_path = self.df_ihc.loc[idx_ihc, "imgs_path"]
        mask_ihc_path = self.df_ihc.loc[idx_ihc, "masks_path"]

        # Read IHC image
        img_ihc = cv.imread(img_ihc_path, -1)
        img_ihc = cv.cvtColor(img_ihc, cv.COLOR_BGR2RGB)

        # Read IHC mask
        mask_ihc_path = self.masks_ihc_path_list[idx_ihc]
        mask_ihc = cv.imread(mask_ihc_path, -1)

        mask_class_dapi = np.zeros ((mask_dapi.shape[0], mask_dapi.shape[1]))
        mask_class_ihc = np.zeros ((mask_ihc.shape[0], mask_ihc.shape[1]))

        # Mask and relabelling depending on the segmentation task
        if  self.classes == 1:  
            mask_class_dapi [mask_dapi[:, :, 0]==255] = 1.
            mask_class_dapi [mask_dapi[:, :, 2]==255] = 1.
            mask_class_ihc [mask_ihc[:, :, 0]==255] = 1.
            mask_class_ihc [mask_ihc[:, :, 2]==255] = 1.
        else: 
            mask_class_dapi [mask_dapi[:, :, 0]==255] = 1.
            mask_class_dapi [mask_dapi[:, :, 2]==255] = 2.
            mask_class_ihc [mask_ihc[:, :, 0]==255] = 1.
            mask_class_ihc [mask_ihc[:, :, 2]==255] = 2.

        # DAPI data augmentation 
        if self.aug_dapi != None:
            augmented = self.aug_dapi(image = img_merge, mask = mask_class_dapi)
            x_img = augmented['image']
            x_mask = augmented['mask']
            x_img = x_img/255
        else:
            # Just image resize during validation 
            x_img = transform.resize(img_merge, (self.img_dim_row, self.img_dim_col))    
            x_mask = transform.resize(mask_class_dapi, (self.img_dim_row, self.img_dim_col), order = 0)

        # IHC data augmentation (it is avoided during the student training)
        if self.aug_ihc != None:
            augmented_ihc = self.aug_ihc(image = img_ihc, mask = mask_class_ihc)
            x_ihc_img = augmented_ihc['image']
            x_ihc_mask = augmented_ihc['mask']
            x_ihc_img = x_ihc_img/255
        else:
            # Just image resize during validation 
            x_ihc_img = transform.resize(img_ihc, (self.img_dim_row, self.img_dim_col))    
            x_ihc_mask = transform.resize(mask_class_ihc, (self.img_dim_row, self.img_dim_col), order = 0)
        
        # To torch tensors
        img_ihc_tensor = transforms.ToTensor()(np.float32(x_ihc_img))
        mask_ihc_tensor = transforms.ToTensor()(np.float32(x_ihc_mask))
        img_tensor = transforms.ToTensor()(np.float32(x_img))
        mask_tensor = transforms.ToTensor()(np.float32(x_mask))
 
        return {'img': img_tensor, 'mask': mask_tensor, 'img_ihc': img_ihc_tensor, 'mask_ihc': mask_ihc_tensor}
    

# ----------------------------- Utilities for the datasets objects. ----------------------------- #

def extract_and_select_patches (img, mask, nr_patches, modality):
    """Utility function to group the patch extraction and selection from WSIs

        Args:
            img (arr): numpy array of the WSI to crop
            mask (arr): numpy array of the mask to crop
            nr_patches (int): number of patches you want to extract from the WSI
            modality (str): 'histo' or 'microct' based on what your are processing
        
        Returns:
            tuple: A tuple containing one image and mask patch.
    """

    img_b, tiss_b, mask_b  = [], [], []

    # Patch size depending on the input image size and number of patches chosen 
    patch_size = max(int (np.array(img).shape[0] / nr_patches), int (np.array(img).shape[1] / nr_patches))

    # Tissue segmentation
    val = filters.threshold_otsu(np.asarray(img)[:, :, 1])
    if modality == 'histo': 
        seg_tissue = np.uint8((np.asarray(img)[:, :, 1] < val)*255)
    elif modality == 'ct':
        seg_tissue = np.uint8((np.asarray(img)[:, :, 1] > val)*255)
    
    # Image patches extraction
    extractor = PatchExtractor(img=Image.fromarray(img), patch_size=patch_size, stride=patch_size)
    patches_img = extractor.extract_img_patches()
    # Tissue patches extraction
    extractor = PatchExtractor(img=Image.fromarray(seg_tissue), patch_size=patch_size, stride=patch_size)
    patches_tiss = extractor.extract_img_patches()
    # Segmentation patches extraction
    extractor = PatchExtractor(img=Image.fromarray(mask), patch_size=patch_size, stride=patch_size)
    patches_seg = extractor.extract_img_patches()

    # Patch selection 
    for i in range(len(patches_img)):
        if (((np.count_nonzero(np.asarray(patches_tiss[i]))/(np.asarray(patches_tiss[i]).shape[0]*np.asarray(patches_tiss[i]).shape[1])) > .9)):
            img_b.append(patches_img[i])
            mask_b.append(patches_seg[i])

    # Stack of WSI patches 
    img_b = np.stack(img_b, axis=0)
    mask_b = np.stack (mask_b, axis=0) 

    # Randomly pickying one index patch from the WSI
    ind = random.randrange(0, img_b.shape[0], 1)

    return img_b[ind], mask_b[ind]


class PatchExtractor:
    """Object to extract image patches from a WSI."""
    def __init__(self, img, patch_size, stride):
        """ Class to extract image patches from a WSI. 
        
        Args:
            img (py:class): WSI image '~PIL.Image.Image'
            patch_size (int): size of the patch
            stride (int): size of the stride 
        """
        self.img = img
        self.size = patch_size
        self.stride = stride

    def extract_img_patches(self):
        """Extracts all patches from an image.

        Returns:
            A list of :py:class:'~PIL.Image.Image' objects for the extracted patches. 
        """
        wp, hp = self.shape()
        return [self.extract_img_patch((w, h)) for h in range(hp) for w in range(wp)]

    def extract_img_patch(self, patch):
        """Extracts one patch from an input image.

        Args:
            patch (tuple): current indexes of the images crop. 

        Returns:
            An :py:class:`~PIL.Image.Image` object of the currect extracted patch.
        """
        return self.img.crop((
            patch[0] * self.stride,  # left
            patch[1] * self.stride,  # up
            patch[0] * self.stride + self.size,  # right
            patch[1] * self.stride + self.size  # down
        ))

    def shape(self):
        """Extracts one patch from an input image.

        Returns:
            A tuple of the image dimensions.
        """
        wp = int((self.img.width - self.size) / self.stride + 1)
        hp = int((self.img.height - self.size) / self.stride + 1)
        return wp, hp