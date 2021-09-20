from setuptools import setup, find_packages

setup(name='det2', 
      version='1.0', 
      packages=find_packages(),
      install_requires=['opencv-python',
                        'detectron2 @ git+https://github.com/facebookresearch/detectron2.git@185c27e4b4d2d4c68b5627b3765420c6d7f5a659'
                        ])
