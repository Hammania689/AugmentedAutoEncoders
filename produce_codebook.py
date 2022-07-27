import contextlib
import sys

import gin
import numpy as np
import torch
from tqdm.notebook import tqdm

from src.aae.models import AugmentedAutoEncoder
from src.datasets.render_tless_dataset import  tless_codebook_online_generator
# from src.datasets.concat_dataset import ConcatDataset
from src.ycb_render.tless_renderer_tensor import *
from src.config.config import cfg, cfg_from_file


class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout
    

@gin.configurable
def create_codebook(cfg_file:    str=gin.REQUIRED,
                    model_path:  str=gin.REQUIRED,
                    pose_path:  str=gin.REQUIRED,):

    with nostdout():
        cfg_from_file(cfg_file)
        cfg.MODE = 'TRAIN'
    
    
        model = AugmentedAutoEncoder(fixed_batch=None, log_to_wandb=False)

        ckpt = torch.load('./results/checkpoints/obj_01.pth')
        model.load_state_dict(ckpt['model'])



        dataset_code = tless_codebook_online_generator(model_path,
                                                       [model.cad_model_name],
                                                       cfg.TRAIN.RENDER_DIST[0],
                                                       output_size=(128, 128),
                                                       gpu_id=cfg.GPU_ID,
                                                       pose_list_path=pose_path)

    model.compute_codebook(dataset_code)
    
    
gin.parse_config_file('./config/test_codebook.gin')
