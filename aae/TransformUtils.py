import gin
import numpy as np

aae_paper_views = 92232
step_size = np.floor(aae_paper_views ** (1/3)).astype(int)

@gin.configurable
def produce_pose_samples(
        num_steps: int=step_size,
        output_path: str=gin.REQUIRED):

    # intervals = np.linspace(0, 360, num_steps)
    intervals = np.deg2rad(np.linspace(-180, 180, num_steps))
    axis_steps = (intervals, intervals, intervals)
    poses = np.stack(np.meshgrid(*axis_steps))

    poses = poses.reshape(3, -1).T
    np.save(output_path, poses)

    return poses

if __name__ == "__main__":
    produce_pose_samples()
