# FROM nvcr.io/nvidia/tensorflow:18.06-py3
# FROM nvcr.io/nvidia/tensorflow:18.10-py3
# FROM nvcr.io/nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
FROM nvcr.io/nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV cwd="/home/"
WORKDIR $cwd

RUN apt-get -y update && apt-get install -y \
    software-properties-common \
    build-essential \
    checkinstall \
    cmake \
    pkg-config \
    yasm \
    git \
    vim \
    curl \
    wget \
    gfortran \
    libjpeg8-dev \
    libpng-dev \
    libtiff5-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libdc1394-22-dev \
    libxine2-dev \
    sudo \
    apt-transport-https \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    dbus-x11 \
    vlc \
    iputils-ping \
    python3-dev \
    python3-pip \
    pigz

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata python3-tk
ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove

### APT END ###

RUN pip3 install --no-cache-dir --upgrade pip 

RUN pip3 install --no-cache-dir \
    numpy==1.15.3 \
    GPUtil \
    tqdm \
    requests \
    protobuf

RUN pip3 install --no-cache-dir  \
    scipy==1.0.0 \
    matplotlib \
    Pillow==5.3.0 \
    opencv-python \
    scikit-image

RUN pip3 install --no-cache-dir \
    torch==1.4.0 \
    torchvision==0.5.0

RUN pip3 install --no-cache-dir jupyter
RUN echo 'alias jup="jupyter notebook --allow-root --no-browser"' >> ~/.bashrc

RUN pip3 install --no-cache-dir tensorboard==1.14
RUN pip3 install --no-cache-dir python-dotenv

# DETECTRON2 DEPENDENCY: PYCOCOTOOLS 
RUN pip3 install --no-cache-dir cython
RUN git clone https://github.com/pdollar/coco
RUN cd coco/PythonAPI \
    && python3 setup.py build_ext install \
    && cd ../.. \
    && rm -r coco


# INSTALL DETECTRON2
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5"
# A100s will require 8.0
# ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0"
RUN git clone https://github.com/facebookresearch/detectron2.git /detectron2
RUN cd /detectron2 &&\
    git checkout 185c27e4b4d2d4c68b5627b3765420c6d7f5a659 &&\
    python3 -m pip install -e .
# RUN rm -r detectron2