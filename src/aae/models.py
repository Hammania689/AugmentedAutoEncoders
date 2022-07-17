import time
from typing import List, Optional, Dict, Union
from pathlib import Path

import gin.torch
import numpy as np
import torch
import wandb
from torch import nn, optim
from torchvision.utils import save_image
from tqdm import tqdm

gin.external_configurable(torch.optim.Adam)


@gin.configurable
class BootstrapedMSEloss(nn.Module):
    def __init__(self,
                 b_factor: int=gin.REQUIRED):
        super(BootstrapedMSEloss, self).__init__()
        self.b_factor = b_factor

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        batch_size = pred.size(0)
        diff = torch.sum((target - pred)**2, 1)
        diff = diff.view(batch_size, -1)

        k = diff.shape[0] // self.b_factor
        diff = torch.topk(diff, k, dim=1)
        self.loss = diff[0].mean()
        return self.loss


class View(nn.Module):
    def __init__(self, shape):
        """
        Class that modularilizes tensor reshaping and can be called in an nn.Sequential list
        # reference: https://discuss.pytorch.org/t/how-to-build-a-view-layer-in-pytorch-for-sequential-models/53958/12
        """
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        batch_size = input.size(0)
        shape = (batch_size, *self.shape)
        out = input.view(shape)
        return out


class Encoder(nn.Module):
    def __init__(self,
                 code_dim: int):
        super(Encoder, self).__init__()

        self.net = nn.Sequential(nn.Conv2d(3, 128, 5, stride=2, padding=2),
                                 nn.ReLU(),
                                 nn.Conv2d(128, 256, 5, stride=2, padding=2),
                                 nn.ReLU(),
                                 nn.Conv2d(256, 256, 5, stride=2, padding=2),
                                 nn.ReLU(),
                                 nn.Conv2d(256, 512, 5, stride=2, padding=2),
                                 nn.ReLU(),
                                 View((512 * 8 * 8,)),
                                 nn.Linear(512 * 8 * 8, code_dim))


    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    
    def __init__(self,
                 code_dim: int):
        
        super(Decoder, self).__init__()

        self.net = nn.Sequential(nn.Linear(code_dim, 512 * 8 * 8),
                                 View((512, 8, 8)),
                                 nn.ConvTranspose2d(512, 256, 5, 2, padding=2, output_padding=1),
                                 nn.ReLU(),
                                 nn.ConvTranspose2d(256, 256, 5, 2, padding=2, output_padding=1),
                                 nn.ReLU(),
                                 nn.ConvTranspose2d(256, 128, 5, 2, padding=2, output_padding=1),
                                 nn.ReLU(),
                                 nn.ConvTranspose2d(128, 3, 5, 2, padding=2, output_padding=1))


    def forward(self, x):
        return self.net(x)


@gin.configurable
class AugmentedAutoEncoder(nn.Module):
    def __init__(self,
                 fixed_batch: List,
                 wandb_entity: Optional[str]=None,
                 log_to_wandb: bool=False,
                 code_dim: int=gin.REQUIRED,
                 opt: torch.optim=gin.REQUIRED,
                 lr: float=gin.REQUIRED,
                 loss: nn.Module=gin.REQUIRED,
                ):

        super(AugmentedAutoEncoder, self).__init__()

        self.fixed_batch = fixed_batch
        self.log_to_wandb = log_to_wandb

        self.encoder = Encoder(code_dim)
        self.decoder = Decoder(code_dim)

        self.opt = opt(self.parameters(), lr)
        self.loss = loss()
        
        self._comp_log = ()
        self._reset_logs()

        self._setup_output()
        if log_to_wandb:
            self._setup_wandb(wandb_entity)


    def _setup_output(self):
        try:
            res = gin.get_bindings('dataset.OnlineRenderer')['cad_model_path']
            self.cad_model_name = (Path(res).name).split('.')[0]
            
            self.output_dir = f"../results/checkpoints/{self.cad_model_name}/{time.time_ns()}"

            # Create path if doesn't exist already
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        except Exception as e:
            raise Exception(e)

        print_sys_out = f"{self.output_dir}/log.txt"
        self.print_sys_out =  open(print_sys_out, 'a')
        print(f"Training AAE for {self.cad_model_name:<40}\n{'='*100}", file=self.print_sys_out)


    def _setup_wandb(self, entity: str):
        wandb.init(project="AugmentedAutoEncoders-test", entity=entity)
        wandb.watch(self)


    def _reset_logs(self):
        self.running_loss = []
        self.cached_recon = []
        self.cached_fixed_recon = []


    def forward(self, x):
        code = self.encoder(x)
        recon = self.decoder(code)
        return (recon, code)


    def optimize_params(self, data, device, cache_recon: bool):

        self.opt.zero_grad()
        
        aug, label, _ = data
        aug = aug.to(device)
        label = label.to(device)

        recon, _ = self.forward(aug)

        recon_loss = self.loss(recon, label)
        self.running_loss.append(recon_loss)

        recon_loss.backward()
        self.opt.step()

        if cache_recon:
            self.cached_recon = list(zip( aug.cpu().numpy(),
                                          label.cpu().numpy(),
                                          recon.detach().cpu().numpy()))

    def reset_log(self):
        self.running_loss = []
    
    def produce_img_reel(self):
        
        pass
        """
        import numpy as np

        reels = [np.concatenate(disp) for disp in self.cached_recon]
        img_grid = np.vstack(reels)

        for disp in self.cached_recon:

            wandb.Im(np.concatenate(disp))
            wandb.Image()
        """


    def log_wandb(self, step):

        log = {'Training/AvgLoss': np.mean(running_loss),
               'Training/Reconstructions': produce_img_reel(),
               'step': step}

        wandb.log(log)

