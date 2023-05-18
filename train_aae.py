import contextlib
import sys
import warnings

import gin
from torch.utils.data import DataLoader
from torchvision.transforms import functional as T
from tqdm import tqdm

from src.aae.models import AugmentedAutoEncoder
from src.aae.Visualizations import plot_batch, plot_img
from src.datasets.render_tless_dataset import *
from src.datasets.concat_dataset import ConcatDataset
from src.ycb_render.tless_renderer_tensor import *
from src.config.config import cfg, cfg_from_file, get_output_dir, write_selected_class_file


class InfiniteIter:
    """
    Custom iterator that infinitely loops over a DataLoader
    """
    def __init__(self, dataloader):
        self.__dict__.update(vars())
        self.__reset__()
        
    def __iter__(self):
        return self

    def __reset__(self):
        self.dataIter = iter(self.dataloader)
    
    def __next__(self, size):
        try:
            data = next(self.dataIter)
        except StopIteration:
            self.__reset__()
            data = next(self.dataIter)
            
        if data.size(0) < size:
            self.__reset__()
            data = next(self.dataIter)
        elif data.size(0) > size:
            return data[:size]
        return data

class DummyFile(object):
    def write(self, x): 
        pass

    def flush(self):
        pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout
    
# TODO (ham) make this more general
@gin.configurable
def load_ycbrenderer_dataset(dis_dir:     str=gin.REQUIRED,
                             cfg_file:    str=gin.REQUIRED,
                             im_dim:      int=gin.REQUIRED,
                             model_path:  str=gin.REQUIRED,):

    cfg_from_file(cfg_file)
    cfg.MODE = 'TRAIN'
    cfg.TRAIN.RENDER_SZ = im_dim
    cfg.TRAIN.INPUT_IM_SIZE = [im_dim, im_dim]

    print(cfg.TRAIN.OBJECTS)
    print(cfg.TRAIN.RENDER_SZ)
    print(cfg.TRAIN.INPUT_IM_SIZE)

    # set up render
    models = cfg.TRAIN.OBJECTS[:]
    renderer = TLessTensorRenderer(cfg.TRAIN.RENDER_SZ, cfg.TRAIN.RENDER_SZ)
    if cfg.TRAIN.USE_OCCLUSION:
        with open('./src/datasets/tless_classes.txt', 'r') as class_name_file:
            class_names_all = class_name_file.read().split('\n')
            for class_name in class_names_all:
                if class_name not in models:
                    models.append(class_name)

        class_colors_all = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
                            (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
                            (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64),
                            (155, 0, 0), (0, 155, 0), (0, 0, 155), (155, 155, 0), (155, 0, 155), (0, 155, 155),
                            (200, 0, 0), (0, 200, 0), (0, 0, 200), (200, 200, 0),
                            (200, 0, 200), (0, 200, 200)
                            ]


        obj_paths = ['{}/tless_models/{}.ply'.format(model_path, item) for item in models]
        texture_paths = ['' for cls in models]
        renderer.load_objects(obj_paths, texture_paths, class_colors_all)
        renderer.set_projection_matrix(cfg.TRAIN.RENDER_SZ, cfg.TRAIN.RENDER_SZ, cfg.TRAIN.FU, cfg.TRAIN.FV,
                                       cfg.TRAIN.RENDER_SZ/2.0, cfg.TRAIN.RENDER_SZ/2.0, 0.01, 10)
        renderer.set_camera_default()
        renderer.set_light_pos([0, 0, 0])

    # synthetic dataset
    dataset_train = tless_multi_render_dataset(model_path, cfg.TRAIN.OBJECTS, renderer,
                                             render_size=cfg.TRAIN.RENDER_SZ,
                                             output_size=cfg.TRAIN.INPUT_IM_SIZE)

    # background dataset
    dataset_dis = DistractorDataset(dis_dir, cfg.TRAIN.CHM_RAND_LEVEL,
                                    size_crop=(cfg.TRAIN.INPUT_IM_SIZE[1],
                                               cfg.TRAIN.INPUT_IM_SIZE[0]))
    
    return dataset_train, dataset_dis


@gin.configurable
def train_aae(num_workers: int=gin.REQUIRED,
              num_train_iters: int=gin.REQUIRED,
              cache_save_interval: int=gin.REQUIRED,
              batch_size: int=gin.REQUIRED,
              device: str=gin.REQUIRED):
    
    
    print("Loading Data....")
    with nostdout():
        dataset_train, dataset_dis = load_ycbrenderer_dataset()

    dl = DataLoader(dataset_train,
                   batch_size=batch_size,
                   shuffle=True,
                   num_workers=num_workers)

    bg_dl = InfiniteIter(DataLoader(dataset_dis,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers))


    def produce_augmented_data(data, bg_dl, device):
        im, target, _, mask, _, _, _, _, _, roi_affine, *_= data
        
        im         = im.to(device)
        target     = target.to(device)
        mask       = mask.to(device)
        roi_affine = roi_affine.to(device)
        
        grids  = F.affine_grid(roi_affine, im.size())
        images = F.grid_sample(im, grids)
        mask   = F.grid_sample(mask, grids)
        mask   = 1 - mask
        

        cur_batch_sz    = im.size(0)
        bg              = bg_dl.__next__(cur_batch_sz).to(device)
        imdisp          = im + mask * bg[:cur_batch_sz]
        noise_level     = np.random.uniform(0, 0.05)
        imdisp          += torch.randn_like(imdisp) * noise_level
        
        return (imdisp, target)
    
    
    for rend in tqdm(dl, desc="AAE Training"):
        fixed_data = produce_augmented_data(rend, bg_dl, device)
        break
    
        
    model = AugmentedAutoEncoder(fixed_batch=fixed_data)
    model = model.to(device)
    cache_save_interval = 2

    ######################################
    # Optimization Step 
    ######################################
    for epoch in tqdm(range(num_train_iters), desc="AAE Training"):

        is_save_epoch = ((epoch + 1) % cache_save_interval == 0)


        for rend in tqdm(dl, desc=f"Epoch: {epoch + 1}", leave=False):
            data = produce_augmented_data(rend, bg_dl, device)
            model.optimize_params(data, device=device, cache_recon=is_save_epoch)

        if is_save_epoch:
            aux_dict = {'dataLoader': fixed_data}
            model.save_state(epoch, aux_dict)
            
        model.log(epoch, is_save_epoch)
        
    return model
    
        
gin.parse_config_file('./config/train/linemod/obj_0001.gin')
model = train_aae()
