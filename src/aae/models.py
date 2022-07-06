import gin.torch
import torch
import wandb
from torch import nn, optim

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
                 code_dim: int=gin.REQUIRED,
                 opt: torch.optim=gin.REQUIRED,
                 lr: float=gin.REQUIRED,
                 loss: nn.Module=gin.REQUIRED,
                ):

        super(AugmentedAutoEncoder, self).__init__()

        self.encoder = Encoder(code_dim)
        self.decoder = Decoder(code_dim)

        self.opt = opt(self.parameters(), lr)

        self.reset_log()


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

        self.opt.backward()

        if cache_recon:
            self.cached_recon = zip([aug, label, recon])

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

