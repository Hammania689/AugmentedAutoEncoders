from typing import List, Tuple

import cv2
import gin.torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.aae.models import AugmentedAutoEncoder
from src.aae.dataset import OnlineRenderer
from src.aae.TransformUtils import aae_paper_views


class InfiniteIter:
    """
    Custom iterator that infinitely loops over a single provided data sample
    """
    def __init__(self, data, bs=64):
        self.__dict__.update(vars())
        
    def __iter__(self):
        self.n = 0
        return self
    
    def __next__(self):
        data = []
        for d in self.data:
            if isinstance(d, tuple):
                d = d[0]
            data.append(torch.stack([d] * self.bs))
        return data


@gin.configurable
def train_aae(num_workers: int=gin.REQUIRED,
              num_train_iters: int=gin.REQUIRED,
              cache_save_interval: int=gin.REQUIRED,
              device: str=gin.REQUIRED):
    
    
    BS = 64
    # batch_iters = aae_paper_views // BS
    batch_iters = 10 #aae_paper_views // BS

    dataset = OnlineRenderer()
    single_data_sample = dataset[0]
    infinite_dl = InfiniteIter(single_data_sample, BS)
    data = next(infinite_dl)

    model = AugmentedAutoEncoder(fixed_batch=data)
    model = model.to(device)
    
    ######################################
    # Optimization Step 
    ######################################
    for epoch in tqdm(range(num_train_iters), desc="AAE Training"):
        
        is_save_epoch = (epoch % cache_save_interval == 0)

        [model.optimize_params(data, device=device, cache_recon=is_save_epoch) for _ in tqdm(range(batch_iters), desc=f"Epoch: {epoch + 1}", leave=False)]

        if is_save_epoch:
            aux_dict = {'dataLoader': infinite_dl}
            model.save_state(epoch, aux_dict)

        model.log(epoch, is_save_epoch)
            

# gin.add_config_file_search_path('../')
gin.parse_config_file('../config/train/linemod/obj_0001.gin')
train_aae()
