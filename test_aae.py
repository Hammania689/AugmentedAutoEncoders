import glob
import copy

import gin
import pprint

from src.datasets.tless_dataset import *
from src.aae.PoseEvaluator import PoseEvaluator
from src.aae.models import AugmentedAutoEncoder
from src.config.config import cfg, cfg_from_file

def get_abs_from_relative(gin_path)-> str:
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


@gin.configurable
def evaluate_aae(sequence:          int=gin.REQUIRED,
                 ckpt_path:         str=gin.REQUIRED,
                 path_to_codebook:  str=gin.REQUIRED,
                 pf_cfg_dir:        str=gin.REQUIRED,
                 train_cfg_dir:     str=gin.REQUIRED,
                 test_cfg_file:     str=gin.REQUIRED,
                 dataset_dir:       str=gin.REQUIRED,
                 device:            str="cuda:0",
                 model_dir:         str=gin.REQUIRED):

    # load the configurations
    cfg_from_file(test_cfg_file)

    # load the test objects
    print('Testing with objects: ')
    print(cfg.TEST.OBJECTS)
    obj_list = cfg.TEST.OBJECTS

    print('Test on TLESS Dataset ... ')
    object_category = 'tless'
    with open('./src/datasets/tless_classes.txt', 'r') as class_name_file:
        obj_list_all = class_name_file.read().split('\n')

    # pf config files
    print(pf_cfg_dir)
    pf_config_files = sorted(glob.glob(pf_cfg_dir + '*yml'))
    cfg_list = []

    for obj in obj_list:
        obj_idx = obj_list_all.index(obj)
        train_config_file = train_cfg_dir + '{}.yml'.format(obj)
        pf_config_file = pf_config_files[obj_idx]
        cfg_from_file(train_config_file)
        cfg_from_file(pf_config_file)
        cfg_list.append(copy.deepcopy(cfg))
    pprint(cfg_list)

    # Load checkpoints and codebooks
    aae = AugmentedAutoEncoder(fixed_batch=None)
    aae_ckpt = torch.load(ckpt_path)
    aae.load_state_dict(aae_ckpt['model'])

    aae.to(device)

    
    path_to_codebook = Path(get_abs_from_relative(path_to_codebook))

    if not path_to_codebook.exists():
        # TODO (ham): Compute code book and save to file
        pass
    
    codebook = torch.load(path_to_codebook)

    # setup the poserbpf
    poseEval = PoseEvaluator(aae,  codebook, obj_list, obj_idx, 
                             cfg_list,  modality='rgb', cad_model_dir=model_dir)

    # test the system on ycb or tless datasets
    target_obj = "obj_01"
    test_list_file = './datasets/TLess/{}/{}.txt'.format(target_obj, sequence)
    dataset_test   = tless_dataset(class_ids=[0],
                                 object_names=[target_obj],
                                 class_model_num=1,
                                 path=dataset_dir,
                                 list_file=test_list_file)

    poseEval.eval_dataset(dataset_test, sequence)


gin.parse_config_file("./config/test_obj01.gin")
evaluate_aae()
