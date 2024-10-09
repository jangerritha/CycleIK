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
          'opencv-python',
          'numpy',
          'torch',
          'networkx',
          'open3d',
          'torchvision',
          'gdown',
          'optuna',
          'pytorch_kinematics'
      ],

     )
