from setuptools import setup, find_packages

setup(
        name='det2',
        version='1.0',
        packages=find_packages(exclude=("test",)),
        install_requires=[
            'opencv-python',
            'detectron2',
            ]
    )
