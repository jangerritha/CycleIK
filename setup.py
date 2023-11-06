#!/usr/bin/env python

from distutils.core import setup

setup(name='CycleIK',
      version='0.1.0',
      description='CycleIK - Neuro-inspired Inverse Kinematics - Neuro-Genetic Robot Kinematics Toolkit for PyTorch',
      author='Jan-Gerrit Habekost',
      author_email='jan-Gerrit.habekost@uni-hamburg.de',
      #url='https://www.inf.uni-hamburg.de/en/inst/ab/wtm.html',
      license='BSD 2-clause',
      packages=['cycleik_pytorch'],
      install_requires=[
          'pillow',
          'tqdm',
          'matplotlib',
          'opencv-python==4.1.2.30; python_version<="3.8"',
          'opencv-python; python_version>"3.8"',
          'numpy==1.21.6; python_version<="3.8"',
          'numpy==1.22.4; python_version>"3.8"',
          'torch',
          'networkx',
          'open3d',
          'torchvision',
          'gdown',
          'optuna',
          #'pytorch_kinematics @ git+https://github.com/UM-ARM-Lab/pytorch_kinematics@bee97ab0d20d21145df2519b1a6db346f20f78d4#egg=pytorch_kinematics-0.4.0'
      ],

     )
