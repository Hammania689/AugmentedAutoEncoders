# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

from datasets.ycb_video_dataset import *
import ruamel.yaml as yaml
from pathlib import Path
import png
import matplotlib.pyplot as plt

def load_info(path):
    with open(path, 'r') as f:
        info = yaml.load(f, Loader=yaml.CLoader)
        for eid in info.keys():
            if 'cam_K' in info[eid].keys():
                info[eid]['cam_K'] = np.array(info[eid]['cam_K']).reshape(
                    (3, 3))
            if 'cam_R_w2c' in info[eid].keys():
                info[eid]['cam_R_w2c'] = np.array(
                    info[eid]['cam_R_w2c']).reshape((3, 3))
            if 'cam_t_w2c' in info[eid].keys():
                info[eid]['cam_t_w2c'] = np.array(
                    info[eid]['cam_t_w2c']).reshape((3, 1))
    return info

def load_gt(path):
    with open(path, 'r') as f:
        gts = yaml.load(f, Loader=yaml.CLoader)
        for im_id, gts_im in gts.items():
            for gt in gts_im:
                if 'cam_R_m2c' in gt.keys():
                    gt['cam_R_m2c'] = np.array(gt['cam_R_m2c']).reshape((3, 3))
                if 'cam_t_m2c' in gt.keys():
                    gt['cam_t_m2c'] = np.array(gt['cam_t_m2c']).reshape((3, 1))
    return gts

def load_depth(depth_path):
    depth = cv2.imread(depth_path, -1)

    if len(depth.shape) == 3:
        depth16 = np.uint16(depth[:, :, 1]*256) + np.uint16(depth[:, :, 2])
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'

    return depth16


class fit_dataset(data.Dataset):
    def __init__(self, class_ids, object_names, class_model_num, path, list_file):
        self.dataset_type = 'fit'
        self.path = path

        list_file = open(list_file)
        name_list = list_file.read().splitlines()

        file_start = name_list[0]
        file_end = name_list[-1]

        import pdb; pdb.set_trace()
        start_num = int(file_start.split('_')[0])
        end_num = int(file_end.split('_')[0])

        """
        file_num = np.arange(start_num, end_num+1)

        file_list = []
        file_list_depth = []
        for i in range(len(file_num)):
            file_list.append(file_start.split('/')[0] + '/'+file_start.split('/')[1] + '/{:04}'.format(file_num[i]))
            file_list_depth.append(file_start.split('/')[0] + '/' + 'depth/{:04}'.format(file_num[i]))

        self.file_list = file_list

        self.files = [path+'/'+item+'.png' for item in file_list]
        self.files_depth = [path + item + '.png' for item in file_list_depth]
        """

        with open('./datasets/fit_classes.txt', 'r') as class_name_file:
            self.class_names_all = class_name_file.read().split('\n')

        assert len(object_names)==1, "current only support loading the information for one object !!!"
        self.object_names = object_names
        self.class_ids = class_ids
        self.class_model_number = class_model_num
        print('class id = ', self.object_names)

        # Load in the ground truth pose, camera info, and bbox predictions 
        self.file_gt_poses = path+'/'+file_start.split('/')[0]+'/gt.yml'    # Ground Truth Pose info
        self.file_cam_info = path+'/'+file_start.split('/')[0]+'/info.yml'  # Cam info
        self.file_bbox = path+'/'+file_start.split('/')[0]+'/bbox_pred.yml' # Bounding Box Predictions


        self.gt_poses = load_gt(self.file_gt_poses)
        self.cam_infos = load_info(self.file_cam_info)
        self.bbox_estimation = load_gt(self.file_bbox)

        self.objId = int(self.object_names[0][:3])
        # TODO (ham) This is hard coded but will need to be updated 
        # self.objId = int(0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # image, bbox = self.load(idx)
        image, pose, intrinsics, bbox, depth = self.load(idx)

        class_info = torch.zeros(128, 128, self.class_model_number)
        class_info[:, :, self.class_ids[0]] = 1
        instance_mask = torch.zeros(3 * self.class_model_number)
        instance_mask[self.class_ids[0]*3 : self.class_ids[0]*3 + 3] = 1
        class_mask = (instance_mask==1)

        bbox = np.asarray(bbox, dtype=np.float32)

        is_kf = True

        return image, depth, pose, intrinsics, class_mask, self.file_list[idx], is_kf, bbox
        # return image, bbox

    def load(self, idx):
        img = np.array(Image.open(self.files[idx]))

        img = img.astype(np.float32) / 255.0

        gt_pose = self.gt_poses[idx]
        cam_info = self.cam_infos[idx]

        # depth = load_depth(self.files_depth[idx])
        # depth = depth * 0.1 / 1000
        # depth = np.expand_dims(depth, 2)
        depth = [0., 0., 1.]

        intrinsics = cam_info['cam_K']

        index_obj = [ind for ind, obj in enumerate(gt_pose) if obj['obj_id'] == self.objId]

        pose = np.zeros((3,4))
        pose[:, :3] = gt_pose[index_obj[0]]['cam_R_m2c']
        pose[:, 3] = gt_pose[index_obj[0]]['cam_t_m2c'].squeeze(1)

        bbox = self.gt_poses[idx][index_obj[0]]['obj_bb']
        bbox_est_all = self.bbox_estimation[idx]
        index_bbox = [ind for ind, obj in enumerate(bbox_est_all) if obj['obj_id'] == self.objId]

        if len(index_bbox) > 0:
            bbox_est = bbox_est_all[index_bbox[0]]['obj_bb']
        else:
            bbox_est = [0.0, 0.0, 0.0, 0.0]

        # TODO (ham) Depth hardcoded to None for prototype rgb demo. One should change this in the future.
        return img, pose, intrinsics, bbox_est, depth


