try:
    from time import time_ns
except:
    import ctypes

    CLOCK_REALTIME = 0

    class timespec(ctypes.Structure):
        _fields_ = [
            ('tv_sec', ctypes.c_int64), # seconds, https://stackoverflow.com/q/471248/1672565
            ('tv_nsec', ctypes.c_int64), # nanoseconds
            ]

    clock_gettime = ctypes.cdll.LoadLibrary('libc.so.6').clock_gettime
    clock_gettime.argtypes = [ctypes.c_int64, ctypes.POINTER(timespec)]
    clock_gettime.restype = ctypes.c_int64    

    def time_ns():
        tmp = timespec()
        ret = clock_gettime(CLOCK_REALTIME, ctypes.pointer(tmp))
        if bool(ret):
            raise OSError()
        return tmp.tv_sec * 10 ** 9 + tmp.tv_nsec

from typing import List, Optional, Dict, Union
from pathlib import Path

import gin.torch
import numpy as np
import torch
try:
    import wandb
except ImportError as e:
    pass
from torch import nn, optim
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

gin.external_configurable(torch.optim.Adam)

def __print__(entry, f):
    tqdm.write(entry, file=f)


def get_path_to_config(gin_path)-> str:
    cur_path = Path(__file__).absolute().parent

    start_path = cur_path
    query_path = tuple()
    for p in Path(gin_path).parts:
        if p == '..':
            start_path = start_path.parent
        elif p == '.':
            continue
        else:
            query_path += (p,)

    query_path = "/".join(query_path)
    
    query_path = start_path / query_path
    return str(query_path)





def pairwise_distances(self, x, y=None):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    return torch.clamp(dist, 0.0, np.inf)


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
                 code_dim: int,
                 im_dim: int):
        super(Encoder, self).__init__()

        pre_latent_im_dim = (im_dim // 16) ** 2 
        self.net = nn.Sequential(nn.Conv2d(3, 128, 5, stride=2, padding=2),
                                 nn.ReLU(),
                                 nn.Conv2d(128, 256, 5, stride=2, padding=2),
                                 nn.ReLU(),
                                 nn.Conv2d(256, 256, 5, stride=2, padding=2),
                                 nn.ReLU(),
                                 nn.Conv2d(256, 512, 5, stride=2, padding=2),
                                 nn.ReLU(),
                                 View((512 * pre_latent_im_dim,)),
                                 nn.Linear(512 * pre_latent_im_dim, code_dim))


    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    
    def __init__(self,
                 code_dim: int,
                 im_dim: int):
        
        super(Decoder, self).__init__()

        
        pre_latent_im_dim = (im_dim // 16) ** 2 
        post_latent_im_shape =  (512, im_dim // 16, im_dim // 16)
        self.net = nn.Sequential(nn.Linear(code_dim, 512 * pre_latent_im_dim),
                                 View(post_latent_im_shape),
                                 nn.ConvTranspose2d(512, 256, 5, 2, padding=2, output_padding=1),
                                 nn.ReLU(),
                                 nn.ConvTranspose2d(256, 256, 5, 2, padding=2, output_padding=1),
                                 nn.ReLU(),
                                 nn.ConvTranspose2d(256, im_dim, 5, 2, padding=2, output_padding=1),
                                 nn.ReLU(),
                                 nn.ConvTranspose2d(im_dim, 3, 5, 2, padding=2, output_padding=1))


    def forward(self, x):
        return self.net(x)


@gin.configurable
class AugmentedAutoEncoder(nn.Module):
    def __init__(self,
                 fixed_batch: List,
                 wandb_entity: Optional[str]=None,
                 log_to_wandb: bool=False,
                 code_dim: int=gin.REQUIRED,
                 im_dim:   int=gin.REQUIRED,
                 opt: torch.optim=gin.REQUIRED,
                 lr: float=gin.REQUIRED,
                 loss: nn.Module=gin.REQUIRED,
                 cad_model_name: str=gin.REQUIRED,
                ):

        super(AugmentedAutoEncoder, self).__init__()

        self.__dict__.update(vars())

        self.encoder = Encoder(code_dim, im_dim)
        self.decoder = Decoder(code_dim, im_dim)


        self.opt = opt(self.parameters(), lr)
        self.loss = loss()
        
        self._comp_log = ()
        self._reset_logs()

        self._setup_output()
        if log_to_wandb:
            self._setup_wandb(wandb_entity)


    def _setup_output(self):
        try:
            target_path = f"../../results/checkpoints/{self.cad_model_name}/{time_ns()}"
            self.output_dir = get_path_to_config(target_path)

            # Create path if doesn't exist already
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        except Exception as e:
            raise Exception(e)

        self.log_path = f"{self.output_dir}/log.txt"
        print_sys_out = self.log_path
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
        
        aug, label = data
        aug = aug.to(device)
        label = label.to(device)

        recon, _ = self.forward(aug)

        recon_loss = self.loss(recon, label)
        self.running_loss.append(recon_loss.detach().cpu().numpy())

        recon_loss.backward()
        self.opt.step()

        if cache_recon:
            self.cached_recon = list(zip( aug.cpu(),
                                          label.cpu(),
                                          recon.detach().cpu()) )

            self.cached_fixed_recon = self.inference(self.fixed_batch, device)


    def inference(self, data: torch.utils.data.DataLoader, device: Union[str, torch.device]) -> List[np.array]:

        with torch.no_grad():

            aug, label = data
            aug = aug.to(device)
            label = label.to(device)

            recon, _ = self.forward(aug)

            return list(zip( aug.cpu(),
                             label.cpu(),
                             recon.detach().cpu()) )

    def produce_img_reel(self, cached_res):
        reels           = [make_grid(list(r), normalize=True) for r in cached_res]
        grid_of_reels   = make_grid(reels, nrows=64 // 4, normalize=True)
        return grid_of_reels


    def log(self, step: int, cache_recon: bool):
        # Log the avg reconstruction loss and visualization of reconstructions 
        avg_recon_loss = np.mean(self.running_loss)
        cur_log = {'Training/AvgLoss': avg_recon_loss,
                   'step': step}

        if cache_recon:
            cur_log.update(**{'Training/Random_Reconstruction_Visualizations': self.produce_img_reel(self.cached_recon),
                              'Training/Fixed_Reconstructions_Visualizations': self.produce_img_reel(self.cached_fixed_recon)})


        self._comp_log = self._comp_log + (cur_log,)

        print(f"Epoch {step:<4d} | Avg Recon Loss: {avg_recon_loss:4f}", file=self.print_sys_out)
        
        # If enabled then logged to wandb
        if self.log_to_wandb:
            # Properly encode the images
            for k, v in cur_log.items():
                if 'Visualizations' in k:
                    title = k.split('/')[-1]
                    title = " ".join(title.split('_'))
                    capt = f"{title:<30} | Epoch: {step + 1:4d}"
                    cur_log[k] = wandb.Image(v, caption=capt)

            wandb.log(cur_log)
            
        # Reset the current running logs
        self._reset_logs()

        
    def save_state(self, epoch: int, aux_dict: Optional[Dict]=None):
        
        # Create, (if necessary) update, and save ckpt object
        states = {'model': self.state_dict(),
                  'log': self._comp_log}
        
        if aux_dict:
            states.update(**aux_dict)

        ckpt_file = f"{self.output_dir}/{self.cad_model_name}_{epoch}.pth"

        torch.save(states, ckpt_file)
        print(f"Saved model to {ckpt_file}", file=self.print_sys_out)

        # Save image reconstructions
        def save_img_reel(reel, name, epoch):
            path = f"{name}_{epoch}.png"
            save_image(reel, path)
            return path

        rand_reel  = self.produce_img_reel(self.cached_recon)
        fixed_reel = self.produce_img_reel(self.cached_fixed_recon)
        
        rand_path  = save_img_reel(rand_reel , f"{self.output_dir}/random", epoch)
        fixed_path = save_img_reel(fixed_reel, f"{self.output_dir}/fixed", epoch)

        if self.log_to_wandb:
            [wandb.save(p) for p in [ckpt_file, rand_path, fixed_path, self.log_path]]

    # codebook generation and saving
    def compute_codebook(self, code_dataset, batch_sz=500):
            codebook_batch_size = batch_sz
            code_generator = torch.utils.data.DataLoader(code_dataset, batch_size=codebook_batch_size, shuffle=False,
                                                         num_workers=0)

            step = 0
            self = self.cuda()
            self.eval()
            n_single_object = int(len(code_dataset))
            codebook_cpt = torch.zeros(n_single_object, self.code_dim).cuda()
            codepose_cpt = torch.zeros(n_single_object, 7).cuda()

            for inputs in tqdm(code_generator, desc="Generate Codebook"):
                rgb, poses, _ = inputs

                poses = poses.cuda()
                rgb = rgb.cuda()

                _, code = self.forward(rgb)
                code    = code.detach().view(rgb.size(0), -1)

                codebook_cpt[step * codebook_batch_size:step * codebook_batch_size + code.size(0), :] = code
                codepose_cpt[step * codebook_batch_size:step * codebook_batch_size + code.size(0), :] = poses.squeeze(1)

                step += 1

            target_path    = f"../../results/codebooks/{self.cad_model_name}.pth"
            codebook_path  = Path(get_path_to_config(target_path))

            # Create path if doesn't exist already
            codebook_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save((codebook_cpt, codepose_cpt), codebook_path)
            if self.log_to_wandb:
                wandb.save(codebook_path)

            print('code book is saved to {}'.format(codebook_path))

    def pairwise_cosine_distances(self, x, y, eps=1e-8):
        """
        :param x: batch of code from the encoder (batch size x code size)
        :param y: code book (codebook size x code size)
        :return: cosine similarity matrix (batch size x code book size)
        """
        dot_product = torch.mm(x, torch.t(y))
        x_norm = torch.norm(x, 2, 1).unsqueeze(1)
        y_norm = torch.norm(y, 2, 1).unsqueeze(1)
        normalizer = torch.mm(x_norm, torch.t(y_norm))

        return dot_product / normalizer.clamp(min=eps)

