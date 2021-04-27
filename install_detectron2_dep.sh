#!/bin/bash
pip3 install \
    torch==1.7.0 \
    torchvision==0.8.1

# DETECTRON2 DEPENDENCY: PYCOCOTOOLS 
pip3 install cython
git clone https://github.com/pdollar/coco \
    && cd coco/PythonAPI \
    && python3 setup.py build_ext install \
    && cd ../.. \
    && rm -rf coco