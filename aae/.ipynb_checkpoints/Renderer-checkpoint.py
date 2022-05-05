import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from typing import Tuple
from pathlib import Path

import gin
import numpy as np
import pyrender
import trimesh

@gin.configurable
class Renderer(object):
    def __init__(self, 
                 W: int=gin.REQUIRED,
                 H: int=gin.REQUIRED,
                 C: int=gin.REQUIRED,
                 render_dims: Tuple=gin.REQUIRED,
                 radius: int=gin.REQUIRED,
                 K: list=gin.REQUIRED,
                 z_near: int=gin.REQUIRED,
                 z_far: int=gin.REQUIRED,
                 render_dist: int=gin.REQUIRED,
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

    def produce_image(self):

        # Create a basic scene
        scene = pyrender.Scene()

        # Add light to scene
        
        
        # Add the camera and set it's pose
        pose = np.eye(4)
        scene.add(self._cam, name="cam", pose=pose)

        # Load object and add it to the scene
        data_path = "../data/t_less/models_cad/obj_01.ply"
        obj = trimesh.load(data_path)

        obj = pyrender.Mesh.from_trimesh(obj)

        T_obj_in_cam = np.eye(4)
        T_obj_in_cam[2, 3] = -self.render_dist

        scene.add(obj, pose=T_obj_in_cam)

        # Render the scene to RGB image
        color, _ = self._r.render(scene)

        return color