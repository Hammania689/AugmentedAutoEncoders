# Please refer to the resources above
# This image was constrcuted following instructions outlined: http://wiki.ros.org/docker/Tutorials/Hardware%20Acceleration
FROM nvidia/cuda:11.1.1-devel-ubuntu18.04
##########################################
# Nvidia's Opengl drivers
# Equivalent to using nvidia/opengl:1.1-glvnd-devel
# Base Instructions copied from: https://gitlab.com/nvidia/container-images/opengl/blob/ubuntu18.04/base/Dockerfile
# Devel Instructions copied from: https://gitlab.com/nvidia/container-images/opengl/blob/ubuntu18.04/glvnd/devel/Dockerfile 
########################################## 
RUN dpkg --add-architecture i386 && \
    apt-get update && apt-get install -y --no-install-recommends \
        libxau6 libxau6:i386 \
        libxdmcp6 libxdmcp6:i386 \
        libxcb1 libxcb1:i386 \
        libxext6 libxext6:i386 \
        libx11-6 libx11-6:i386 && \
    rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTORCH_VERSION=1.10.2

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
     ${NVIDIA_VISIBLE_DEVICES:-all}

ENV NVIDIA_DRIVER_CAPABILITIES \
     ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics


ENV LD_LIBRARY_PATH=/usr/lib/nvidia-<vvv>:$LD_LIBRARY_PATH
 
# Install prerequisites
# See python https://ubuntu.pkgs.org/18.04/ubuntu-main-amd64/libpython3.6_3.6.5-3_amd64.deb.html
RUN ROS_PYTHON_VERSION=3 \
    && apt-get update && apt-get install -y lsb-release && apt-get clean all \
    && sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
    && apt install -y curl  \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && apt update \
    && apt install -y ros-melodic-desktop-full \
    && echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc \
    && source ~/.bashrc \
    && apt install -y python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential \
    && apt-get install -y software-properties-common python-pip python3-pip ros-melodic-ros-control ros-melodic-ros-controllers ros-melodic-moveit ros-melodic-trac-ik-kinematics-plugin \
    && add-apt-repository ppa:sdurobotics/ur-rtde && apt-get update && apt install -y librtde librtde-dev \
    && apt install -y python3-dev \
	&& update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \ 
	&& python -m pip install --upgrade pip 


# NOTE: See this link for torch and torch vision version compatiblity: https://pypi.org/project/torchvision/0.15.2/
ARG TORCH_MAJOR_VERSION=1.10
ARG TORCH_MINOR_VERSION=2
# NOTE: This must match or be lower than the cuda version of the docker image AND host machine
# NOTE: There is no period (e.g. CUDA 10.2 is cu102 and CUDA 11.1 is cu111)
ARG TORCH_CUDA_VERSION=111
# NOTE See the prebuilt binaries for:
# Torch: https://download.pytorch.org/whl/torch/
# Torchvision: https://download.pytorch.org/whl/torchvision/
# Detectron2 v0.6: https://github.com/facebookresearch/detectron2/releases/tag/v0.6
ARG TORCHVISION_MAJOR=0.11
ARG TORCHVISION_MINOR=3

RUN apt-get install -y libpython3.6 ros-melodic-catkin python-catkin-tools apt-utils \
	&& python -m pip install pip setuptools wheel \
	&& python -m pip install rospkg catkin_pkg matplotlib scipy empy opencv-python==4.5.2.54 \
    && apt install python-rosdep \
    && rosdep init \
    && rosdep update

WORKDIR /tmp
RUN apt update \
    && apt install -y wget python3-tk \
    && echo https://download.pytorch.org/whl/cu${TORCH_CUDA_VERSION}/torch-${TORCH_MAJOR_VERSION}.${TORCH_MINOR_VERSION}%2Bcu${TORCH_CUDA_VERSION}-cp36-cp36m-linux_x86_64.whl \
    && cd /tmp && wget https://download.pytorch.org/whl/cu${TORCH_CUDA_VERSION}/torch-${TORCH_MAJOR_VERSION}.${TORCH_MINOR_VERSION}%2Bcu${TORCH_CUDA_VERSION}-cp36-cp36m-linux_x86_64.whl \
    && python -m pip install torch-${TORCH_MAJOR_VERSION}.${TORCH_MINOR_VERSION}+cu${TORCH_CUDA_VERSION}-cp36-cp36m-linux_x86_64.whl \
    && cd /tmp && wget https://download.pytorch.org/whl/cu${TORCH_CUDA_VERSION}/torchvision-${TORCHVISION_MAJOR}.${TORCHVISION_MINOR}%2Bcu${TORCH_CUDA_VERSION}-cp36-cp36m-linux_x86_64.whl \
    && python -m pip install torchvision-${TORCHVISION_MAJOR}.${TORCHVISION_MINOR}+cu${TORCH_CUDA_VERSION}-cp36-cp36m-linux_x86_64.whl \
    && python -m pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu${TORCH_CUDA_VERSION}/torch${TORCH_MAJOR_VERSION}/index.html \
    && python -m pip install easydict Cython filterpy numba imgaug ruamel.yaml pypng ninja open3d gin-config netifaces wandb 

WORKDIR /opt
# Prebuild install and necessary link creation
# install eigen if not installed already
RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.3.8/eigen-3.3.8.tar.gz \
    && tar -xvf eigen-3.3.8.tar.gz \
    && rm eigen-3.3.8.tar.gz \
    && cd eigen-3.3.8 \
    && mkdir -p build && cd build \
    && cmake .. \
    && make -j8 install

# RUN apt install -y ccache libgoogle-glog-dev \
#     && git clone https://ceres-solver.googlesource.com/ceres-solver ceres-solver \
#     && cd ceres-solver \
#     && git reset --hard b0aef211db734379319c19c030e734d6e23436b0 \
#     && mkdir -p build \
#     && cd build \
#     && ccache -s \
#     && cmake -DCMAKE_CXX_COMPILER_LAUNCHER=ccache .. \
#     && make -j8 \
#     && make install
# 
# RUN apt-get install -y libfmt-dev \
#     && wget -O Sophus-1.22.10.tar.gz https://github.com/strasdat/Sophus/archive/refs/tags/1.22.10.tar.gz \
#     && tar -xvf Sophus-1.22.10.tar.gz \
#     && rm Sophus-1.22.10.tar.gz \
#     && cd Sophus-1.22.10 \
#     && mkdir -p build && cd build \
#     && cmake .. -DUSE_BASIC_LOGGING=ON \
#     && make -j8 install 

# Install pip dependecies
# Compile ycb renderer
# RUN apt-get update \ 
#    && apt-get install git \
#    && git clone https://github.com/NVlabs/PoseRBPF.git --recursive \
#    && cd PoseRBPF \
#    && apt-get install -y libassimp-dev \
#    && cd ycb_render \ 
#    && pip install -r requirement.txt \
#    && pip install -e . \
#    && cd ../ \
#    && ./scripts/download_demo.sh 

WORKDIR /cobot_aae
#COPY build.sh /cobot_aae/build.sh
#COPY utils /cobot_aae/utils
#COPY ycb_render /cobot_aae/ycb_render
#COPY fit_scripts /cobot_aae/fit_scripts
#COPY scripts /cobot_aae/scripts
    #  && cd ../ \

# CMD  cd /cobot_aae/ycb_render \ 
#      && pip install -r requirement.txt \
#      && python setup.py develop \
#      && bash build.sh \
#      && bash fit_scripts/download_pipeline_files

# WORKDIR /

# 
# 
# 
# WORKDIR /PoseRBPF

ENV LC_ALL=C.UTF-8 
ENV LANG=C.UTF-8
RUN python -m pip install transforms3d
