# -*- coding: utf-8 -*-
import numpy as np
from os.path import realpath
from pathlib import Path


def linear_kernel(x):
    return np.c_[np.ones(len(x)), x]


def load_pw_mapping():
    return np.load(Path(realpath(__file__)).parent / 'pw_mapping.npy')
