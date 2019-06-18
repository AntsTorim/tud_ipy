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
    

def test_conf2d():
    a = np.array([[0, 0, 1, 0, 1],
                  [1, 1, 0, 1, 0],
                  [0, 0, 1, 0, 1]])
    conf_a = Conf2DSeriation().fit_transform(a)
    expected = np.array([[1, 1, 0, 0, 0],
                         [1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1]])
    assert (conf_a == expected).all()

    
def test_liseriate():
    a = np.array([[0, 1, 0, 0, 1],
                  [1, 1, 1, 0, 1],
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1]])
    lis_a = LexiIterSeriation().fit_transform(a)
    expected = np.array([[1, 1, 1, 1, 0],
                         [1, 1, 0, 0, 0],
                         [1, 0, 0, 0, 1],
                         [1, 0, 0, 0, 1]])
    assert (lis_a == expected).all()
    # Test with refiller
    r = Refiller()
    r.fit(a)
    b = np.concatenate((np.zeros((1, 5)), a[1:,:]))
    lis_b = LexiIterSeriation(refiller=r).fit_transform(b)
    expected_b = np.array([[1, 1, 1, 1, 0],
                         [1, 0, 0, 0, 1],
                         [1, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0]])
    assert (lis_b == expected_b).all()
    c = np.array([[0, 1, 0, 0, 1],
                  [0, 0, 1, 0, 1],
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1]])
    lis = LexiIterSeriation(refiller=r)
    lis.fit(c)
    lis_ca = lis.transform(a)
    assert (lis_ca == expected).all()


def test_refill():
    a_0 = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [1, 1, 1]])
    a = np.array([[0, 0, 0],
                  [0, 1, 0],
                  [0, 1, 0]])
    r = Refiller()
    r.fit(a_0)
    refilled = r.transform(a)
    expected = np.array([[0, 0, 0],
                         [0, 1, 0],
                         [1, 1, 0]])
    assert (refilled == expected).all()
    refilled_c = r.transform(a, axis=0, row_i=[0,1,2],col_i=[1,0,2])
    expected = np.array([[0, 1, 0],
                         [0, 1, 0],
                         [0, 1, 0]])
    assert (refilled_c == expected).all()
    r.arrange(row_i=[0,1,2], col_i=[1,0,2])
    refilled_ar = r.transform(a, axis=0)
    assert (refilled_ar == expected).all()
    refilled_arc = r.transform(a, axis=1, row_i=[1,0,2],col_i=[1,2,0])
    expected = np.array([[0, 0, 0],
                         [0, 0, 0],
                         [1, 1, 0]])
    


def test_FL_seriate():
    a = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0]])
    fls_a = FreqLexiSeriation(preseriate=Conf2DSeriation).fit_transform(a)
    expected = np.array([[1, 0, 0, 0],
                         [1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]])
    assert (fls_a == expected).all()
    
    # Test with refill, fls_r corner will be 1 in a0
    a0 = np.array([[1, 1, 1, 0],
                   [1, 1, 0, 1],
                   [1, 0, 0, 0],
                   [1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 1, 0, 0]])
    r = Refiller()
    r.fit(a0)
    fls_r = FreqLexiSeriation(preseriate=Conf2DSeriation, refiller=r).fit_transform(a)
    expected = np.array([[0, 1, 0, 0],
                         [1, 0, 0, 0],
                         [1, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]])
    assert (fls_r == expected).all()
    
    
if __name__ == "__main__":
    pytest.main(args=["-v"])
    #test_FL_seriate()