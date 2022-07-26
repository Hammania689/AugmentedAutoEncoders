from pathlib import Path
import gin
import numpy as np
import torch
from transforms3d.euler import euler2quat

aae_paper_views = 92232
step_size = np.floor(aae_paper_views ** (1/3)).astype(int)

def get_path_to_config(gin_path)-> str:
    cur_path = Path().absolute()

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
def produce_pose_samples(
        num_steps: int=step_size,
        output_path: str=gin.REQUIRED):

    intervals = np.deg2rad(np.linspace(-180, 180, num_steps))
    axis_steps = (intervals, intervals, intervals)
    poses = np.stack(np.meshgrid(*axis_steps))

    poses = poses.reshape(3, -1).T
    poses = np.stack([euler2quat(*p) for p in poses])
    position = np.stack([np.array([0, 0, -.16]) for _ in poses])
    poses = np.c_[position, poses]
    poses = torch.from_numpy(poses)
    
    torch.save(poses, get_path_to_config(output_path))
    return poses

if __name__ == "__main__":
    produce_pose_samples()
