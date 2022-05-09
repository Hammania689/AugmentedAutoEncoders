import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from typing import Tuple
from pathlib import Path

import cv2
import gin
import numpy as np
import pyrender
import trimesh
from transforms3d import euler

from imgaug import augmenters as iaa



from aae.TransformUtils import produce_pose_samples

@gin.configurable
class Renderer(object):
    def __init__(self, 
                 W: int=gin.REQUIRED,
                 H: int=gin.REQUIRED,
                 render_dims: Tuple=gin.REQUIRED,
                 K: list=gin.REQUIRED,
                 z_near: int=gin.REQUIRED,
                 z_far: int=gin.REQUIRED,
                 render_dist: int=gin.REQUIRED,
                 voc_path: str=gin.REQUIRED,
                 light_strength: float=gin.REQUIRED,
                ):
        """
        Class to produce images of objects from equivariant sampled poses 
        """
        
        self.__dict__.update(locals())
        
        self._r = pyrender.OffscreenRenderer(*render_dims)
        
        (FU, _, U0, 
         _, FV, V0, 
         _,  _, _)   = self.K

        self.K = np.array(self.K).reshape((3,3))

        self._cam = pyrender.IntrinsicsCamera(FU, FV, U0, V0, 
                                              znear=z_near, zfar=z_far)

        self.poses = produce_pose_samples()

        self.voc_paths = list(Path(voc_path).rglob('*.jpg'))
        self.resize_dims = (self.H, self.W)



        self.seq = iaa.Sequential([iaa.Sometimes(0.5, iaa.Affine(scale=(1.0, 1.2))),
                                   iaa.Sometimes(0.5, iaa.CoarseDropout( p=0.2, size_percent=0.05) ),
                                   iaa.Sometimes(0.5, iaa.GaussianBlur(1.2*np.random.rand())),
                                   iaa.Sometimes(0.5, iaa.Add((-25, 25), per_channel=0.3)),
                                   iaa.Sometimes(0.3, iaa.Invert(0.2, per_channel=True)),
                                   iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4), per_channel=0.5)),
                                   iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4))),
                                   iaa.Sometimes(0.5, iaa.ContrastNormalization((0.5, 2.2), per_channel=0.3))],
                                   random_order=False)


    def _produce_image(self):

        # Create a basic scene
        scene = pyrender.Scene(bg_color=(0, 0, 0, 1))

        # Add light to scene
        light = pyrender.PointLight(color=np.ones(3), intensity=self.light_strength)
        T_cam = np.eye(4)

        scene.add(light)
        scene.add(self._cam, name="cam")

        # Load object and add it to the scene
        tmp_obj = trimesh.load(self.data_path)
        obj = pyrender.Mesh.from_trimesh(tmp_obj)

        # Convert scale from millimeters to meters
        T_obj_in_cam = np.eye(4)
        T_obj_in_cam[:3, :3] /= 1000
        T_obj_in_cam[2, 3] = -self.render_dist / 1000 


        # Sample pose and generate
        sample = np.random.randint(0, self.poses.shape[0], size=1)[0]
        cur_pose = self.poses[sample]

        R = euler.euler2mat(*cur_pose)
        T_obj_in_cam[:3, :3] = R @ T_obj_in_cam[:3, :3]
        scene.add(obj, pose=T_obj_in_cam)

        # Render the scene to RGB image
        rgb, depth = self._r.render(scene)
        original = rgb.copy()

        # Add background
        bg_sample = np.random.randint(0, len(self.voc_paths), size=1)[0]
        bg_img = str(self.voc_paths[bg_sample])
        bg = cv2.imread(bg_img)[:, :, ::-1]

        
        # Rescale and Overlay
        bg = cv2.resize(bg, self.resize_dims)
        rgb = cv2.resize(rgb, self.resize_dims)
        depth = cv2.resize(depth, self.resize_dims)

        # Combine masked out images to produce result
        depth_three_chan = np.dstack((depth,)*3)
        rgb_bg = bg * (depth_three_chan==0.0).astype(np.uint8) + rgb*(depth_three_chan>0).astype(np.uint8)

        # Delete scene and other items
        del scene
        return (original, rgb_bg, cur_pose)


    def produce_batch_images(self, batch_size: int=32):
        data = [self._produce_image() for _ in range(batch_size)]
        original, rgb_bg, poses = zip(*data)

        aug_rgb = self.seq(images=rgb_bg)

        if batch_size == 1:
            original = np.squeeze(original)
            rgb_bg = np.squeeze(rgb_bg)
            aug_rbg = np.squeeze(aug_rgb)

        return  aug_rgb, original, poses

