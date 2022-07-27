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


@gin.configurable
def train_aae(num_workers: int=gin.REQUIRED,
              num_train_iters: int=gin.REQUIRED,
              cache_save_interval: int=gin.REQUIRED,
              device: str=gin.REQUIRED):
    
    
    BS = 64
    batch_iters = 1 #aae_paper_views // BS

    dataset = OnlineRenderer()
    dl = DataLoader(dataset,
               batch_size=dataset.batch_size,
               shuffle=True,
               num_workers=0)

    for data in dl:
        fixed_data = data
        break

    model = AugmentedAutoEncoder(fixed_batch=fixed_data)
    model = model.to(device)

    
    
    ######################################
    # Optimization Step 
    ######################################
    for epoch in tqdm(range(num_train_iters), desc="AAE Training"):
        
        is_save_epoch = (epoch % cache_save_interval == 0)

        # Create new dataset for each epoch
        dataset = OnlineRenderer()
        dl = DataLoader(dataset,
                   batch_size=dataset.batch_size,
                   shuffle=True,
                   num_workers=0)

        for data in tqdm(dl, desc=f"Epoch: {epoch + 1}", leave=False):
            model.optimize_params(data, device=device, cache_recon=is_save_epoch)

        if is_save_epoch:
            aux_dict = {'dataLoader': fixed_data}
            model.save_state(epoch, aux_dict)

        model.log(epoch, is_save_epoch)

gin.parse_config_file('./config/train/linemod/obj_0001.gin')
train_aae()
