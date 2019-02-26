# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 19:12:05 2019

@author: ator-
"""

import pytest
from sortseriate import *
import numpy as np

def test_freq():
    a = np.array([[1, 0, 0],
                  [0, 0, 0],
                  [1, 1, 1]])
    freq_a = FreqSeriation().fit_transform(a)
    expected = np.array([[1, 1, 1],
                         [1, 0, 0],
                         [0, 0, 0]])
    assert (freq_a == expected).all()
    
    
def test_conf():
    a = np.array([[1, 1, 1, 0, 0],
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1]])
    conf_a = ConfSeriation().fit_transform(a)
    expected = np.array([[0, 0, 0, 1, 1],
                         [0, 0, 0, 1, 1],
                         [1, 1, 1, 0, 0]])
    assert (conf_a == expected).all()