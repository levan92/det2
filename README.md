# Inference wrapper for Detectron2

## Dependencies

- detectron2:

```bash
./install_detectron2_dep.sh && python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2 && python3 -m pip install -e .

# Or if you are on macOS
# CC=clang CXX=clang++ python -m pip install -e .
```
- cv2

## Installation

- In the main project folder, install `det2` as a python package using `pip` or as an editable package if you like (`-e` flag after `pip`)

```bash
cd deep_sort_realtime && pip3 install .
```

## Docker

- Get docker:`docker pull levan92/det2` OR Build your own docker from `Dockerfile`, you will need the tensorrt installation [deb file](https://drive.google.com/file/d/10NT4GYOAOjrwdSGPJS6v6uyVtduW-Pa3/view?usp=sharing)
- Use docker: `./run_docker.sh` look into the script to change paths accordingly

## Get weights

```bash
cd weights
./get_weights.sh
```

## Example usage

See `example_image.py` and `example_video.py`
