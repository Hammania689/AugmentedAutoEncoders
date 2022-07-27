# Augmented AutoEncoders 🖼️

Hameed Abdul (hameeda2@illinois.edu) | Spring 22' [CS 498 Introduction into Machine Perception][cs498]

This is a re-implementation of [Sundermeyer's et al.'s Augmented AutoEncoders][aae_paper]$^1$. 
This accompanying [manuscript][overleaf] describe in further detail this re-implementation.
Supplementary visuals can be found [here][supp] 



https://user-images.githubusercontent.com/20171200/181167929-119342c1-3a12-41bf-9286-c34f87e3acda.mp4



## Setup 👷🏿

### Environment :earth_africa:
```bash
git clone --recursive https://github.com/Hammania689/AugmentedAutoEncoders.git
cd AugmentedAutoEncoders

conda env create -f env.yml
conda activate aae
pip install -e .

# YCB Renderer Setup
cd src/ycb_render
sudo apt-get install libassimp-dev
pip install -r requirement.txt
# additionally, you need to install nvidia OpenGL drivers and make them visible
export LD_LIBRARY_PATH=/usr/lib/nvidia-<vvv>:$LD_LIBRARY_PATH
pip install -e .

# ROI Align Setup
cd ../../src/RoIAlign
pip install -e .
```
### Data and Checkpoints :floppy_disk:

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
