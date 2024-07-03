"""Object to create the BonesAI dataloader sampler and regulate batching mechanisms."""

import random
import numpy as np

class BonesAIDataSampler():
    """Object to sample the patches from different image samples in each batch."""
    def __init__(self, dataframe, idx, batch_size):
        """ Class that samples the patches from different image samples in each batch.
        
        Args:
            dataframe (Dict): dataframe of the image and mask paths 
            idx (int): indexes of the dataset samples
            batch_size (int): number of desidered images per batch 
        """
        self.dataframe = dataframe
        self.idx = idx 
        self.n_batches = len(idx) 
        self.batch_size = batch_size
    
    def __len__(self):
        return self.n_batches

    def __iter__(self):
        """
        Returns:
            iter: An iterable object of the shuffled batches that constitute an epoch.
        """
        batches = []

        # Choose randomly from the dataset indexes to populate the batch
        for _ in range(self.n_batches):
            if self.batch_size >= len (self.idx):
                extended_idx_set = [*self.idx, *np.random.choice(self.idx, self.batch_size-len(self.idx))]
            else: 
                extended_idx_set = np.random.choice(self.idx, self.batch_size)
        
            # Shuffle each batch
            random.shuffle(extended_idx_set)
            batches.append(extended_idx_set)

        return iter(batches)