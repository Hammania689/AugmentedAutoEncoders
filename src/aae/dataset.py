import gin
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from src.aae.Renderer import Renderer

@gin.configurable
class OnlineRenderer(Dataset):

    def __init__(self, 
                 batch_size: int=gin.REQUIRED,
                 cad_model_path: str=gin.REQUIRED):

        self.__dict__.update(locals())
        self._r = Renderer(cad_model_path)


    def __len__(self):
        return len(self._r.poses)


    def __getitem__(self, idx):
        aug, target, poses = self._r.produce_batch_images(batch_size=1)

        aug = ToTensor()(aug)
        target = ToTensor()(target)
        poses = poses
        return aug, target, poses
