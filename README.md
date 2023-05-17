# Augmented AutoEncoders 🖼️

Hameed Abdul (hameeda2@illinois.edu) | Spring 22' [CS 498 Introduction into Machine Perception][cs498]

This is a re-implementation of [Sundermeyer's et al.'s Augmented AutoEncoders][aae_paper]$^1$. 


- :scroll: This accompanying [manuscript][overleaf] describe in further detail this re-implementation.
- 📚 Supplementary visuals can be found [here][supp] 
- 🏋🏿‍♂️ Official BOP Challenge Submision can be found [here][bop]



https://user-images.githubusercontent.com/20171200/181167929-119342c1-3a12-41bf-9286-c34f87e3acda.mp4


## Environment Setup 👷🏿 :earth_africa:
```bash
git clone --recursive https://github.com/Hammania689/AugmentedAutoEncoders.git
cd AugmentedAutoEncoders
```


### Docker 🐳
❗*We **strongly** suggest that Docker be used*❗
<details>
<summary>Click to expand....</summary>


### Prequisites
- [Docker][docker]
- [Nvidia-docker2][nv2]
- [nvidia-container-runtime][ncr]

[docker]: https://docs.docker.com/install/
[nv2]: https://github.com/nvidia/nvidia-docker/wiki/
[hardware]: http://wiki.ros.org/docker/Tutorials/Hardware%20Acceleration
[ncr]: https://github.com/nvidia/nvidia-container-runtime#ubuntu-distributions

### Running Commands

```bash
cd {path_to_this_repo}/pose_estimation/poserbpf
# Build image 
bash Docker/build_image

# Build and start container
bash Docker/build_container
```

The previous command will start an interactive shell session with the `stable_pose_aae` docker container that was just built.


#### Post Container Setup
```bash
cd src/ycb_render
pip install -r requirement.txt
pip install -e .

# ROI Align Setup
cd ../../src/RoIAlign
pip install -e .
```


To start and connect to the built container 
# Access the running container in another terminal
```bash 
Docker/start_container
```

*This will start another interactive shell session with the running `stable_pose_aae` container that was built. Running this is equivalent to opening a new terminal window. **So prior to running the roslaunch or rosrun commands outline below you will need to run `docker exec -it stable_pose_aae bash`***

#### Helpful Resources for Extending and Debugging Docker with ROS, NVIDIA, and GUI passthrough
- [Official MoveIt! 1 Docker Install Documentation][moveit]
- [ROS' Docker Hardware Accleration][ros_docker_doc]
- [How to Use Basler USB Cameras in Docker Container][basler_dock]

[moveit]:https://moveit.ros.org/install/docker/
[basler_dock]:https://www.baslerweb.com/en/sales-support/knowledge-base/frequently-asked-questions/how-to-use-basler-usb-cameras-in-docker-container/588488/
[ros_docker_doc]:http://wiki.ros.org/docker/Tutorials/Hardware%20Acceleration

</details>


### Local Setup with Conda 🖥

<details>
<summary> Click to expand....</summary>

#### Conda setup
```bash
conda env create -f env.yml
conda activate aae
pip install -e .
```

#### YCB Renderer & ROI Align Setup
```bash
cd src/ycb_render
sudo apt-get install libassimp-dev
pip install -r requirement.txt
# additionally, you need to install nvidia OpenGL drivers and make them visible
export LD_LIBRARY_PATH=/usr/lib/nvidia-<vvv>:$LD_LIBRARY_PATH
pip install -e .

cd ../../src/RoIAlign
pip install -e .
```
</details>


## Download Data and Checkpoints :floppy_disk:

`bash .scripts/download_data.sh`

## Run Commands 🏇🏿
|Command| Description| Arguments / Configurations|
|--|--|--|
|`python train_aae.py`| Train an AAE to for obj_01 from the T-LESS dataset. Throughout the training process, visualizations and checkpoints are saved to disk (and optionally [wandb][wandb]) at a predifined interval. | see [obj_00001.gin][train_cfg] |
|`python produce_codebook.py`| This script uses a trained AAE to produce a codebook of latent vectors and their corresponding poses | see [codebook.gin][cb_cfg] |
|`python test_aae.py`| Given a sequence of images with ground truth poses information, this script evaluates the Visible Surface Discrepency ($e_{vsd}$) and axes wise rotation error of the AAE's codebook matching estimated poses. Visualizations are saved to disk and can be logged to [wandb][wandb] | see [test_obj01.gin][test_cfg] |


####  References :book:
1. Sundermeyer, Martin, et al. ["Implicit 3d orientation learning for 6d object detection from rgb images."][aae_paper] _Proceedings of the european conference on computer vision (ECCV)_. 2018.


[overleaf]: https://www.overleaf.com/read/xrjynfnswxqn
[aae_paper]:https://arxiv.org/abs/1902.01275
[vid]: https://drive.google.com/file/d/1I_XpvzuptCkVtKkc63rGoDXEgP39ZJn-/view?usp=sharing
[supp]: https://bit.ly/3zel9gK
[cs498]: https://shenlong.web.illinois.edu/teaching/cs498spring22/
[wandb]: https://wandb.ai

[cb_cfg]: https://github.com/Hammania689/AugmentedAutoEncoders/blob/main/config/codebook.gin
[train_cfg]: https://github.com/Hammania689/AugmentedAutoEncoders/blob/main/config/train/linemod/obj_0001.gin
[test_cfg]: https://github.com/Hammania689/AugmentedAutoEncoders/blob/main/config/test_obj01.gin
[bop]: https://bop.felk.cvut.cz/sub_info/2427/
