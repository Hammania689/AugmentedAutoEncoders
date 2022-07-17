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

def plot_im(im, fig_size:Tuple[int, int]=(60, 30)):
    fig = plt.figure(figsize=fig_size, tight_layout=True)
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    fig.set_tight_layout(True)
    im = im.transpose(2, 1, 0) 
    plt.imshow(im)
    
    
def combine_in_out_imgs(imgs: List[np.array], axis: int) -> np.array:                                                                                                                      
    # NOTE: torch img dim (N, C, H, W) ¬
    return np.concatenate(imgs, axis=axis)     

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
              save_interval: int=gin.REQUIRED,
              cache_interval: int=gin.REQUIRED,
              device: str=gin.REQUIRED):
    
    model = AugmentedAutoEncoder()
    model = model.to(device)
    
    BS = 64
    dataset = OnlineRenderer()
    single_data_sample = dataset[0]
    infinite_dl = InfiniteIter(single_data_sample, BS)
    data = next(infinite_dl)
    iters = 200
    state_dict_hist = []
    
    ######################################
    # Preview Data
    ######################################
    view_limit = 1
    col_stack_view_size = (20, 20)
    
    aug, img, _ = data
    aug_stack = np.split(aug.cpu().numpy(), BS, axis=0)
    aug_stack = [a.squeeze() for a in aug_stack]
    aug_col_stack = combine_in_out_imgs(aug_stack[:view_limit], 2)
    # plot_im(aug_col_stack, col_stack_view_size)
    plot_im(aug_stack[0])
    
    ######################################
    # Optimization Step 
    ######################################
    for idx in tqdm(range(iters)):
        
        cache_recon = (idx % cache_interval == 0)
        if cache_interval:
            state_dict_hist.append((idx, model.state_dict))
        # aug, target, pose = next(infinite_dl)
        [model.optimize_params(data, device=device, cache_recon=cache_recon) for _ in tqdm(range(aae_paper_views // BS), leave=False)]
            
    
    
    
    ckpt_dict = {'model': model.state_dict(),
                 'prev_ckpts': state_dict_hist,
                 'dataloader': infinite_dl}
    return ckpt_dict

# gin.add_config_file_search_path('../')
gin.parse_config_file('../config/train/linemod/obj_0001.gin')
ckpt_dict = train_aae()

torch.save(ckpt_dict,'overfit_on_image.pth')
