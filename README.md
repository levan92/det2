
# Object Container for Detectron2

## Docker

- Get docker:`docker pull levan92/cv-suite`
- Use docker: `./run_docker.sh` look into the script to change flags accordingly

## Get weights

```bash
cd weights
./get_weights.sh
```

## Example usage

See `example_image.py` and `example_video.py`

## Dependencies

- Detectron2:

```bash
python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2 && python3 -m pip install -e .

# Or if you are on macOS
# CC=clang CXX=clang++ python -m pip install -e .
```
