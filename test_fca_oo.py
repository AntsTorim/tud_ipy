# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 10:28:55 2016

@author: ator-

To run right click project and open command prompt. 
Type py.test.
"""

import pytest

from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as st_np

import numpy as np

import pandas as pd

import kernel_fca_oo as krn




@pytest.fixture(scope='function')
def kernel_classes():
    return [krn.KernelSystemNP, krn.KernelSystemDF, krn.FCASystemDF]


arr_shape_strat = st.tuples(st.integers(min_value=2, max_value=5), 
                            st.integers(min_value=2, max_value=5))
bin_int_strat = st.integers(min_value=0, max_value=1)
arr_strat = arr_shape_strat.flatmap(lambda t: st_np.arrays(np.int8, (t[0], t[1]), 
                                                           elements=bin_int_strat))
intent_strat = st.shared(arr_strat, key=1).flatmap(lambda a: st.sets(st.integers(0, a.shape[1]-1)))


@given(st.shared(arr_strat, key=1), intent_strat)
def test_ei(kernel_classes, arr, gen_intent):
    #print(arr)
    #print(arr2)
    #print('_______')
    df = pd.DataFrame(arr)
    for K in kernel_classes:
        if K == krn.KernelSystemNP:
            ks = K(arr)
        else:
            ks = K(df)
        extent = ks.extent(gen_intent)
        intent = set(ks.intent(extent))
        assert intent >= gen_intent
        extent2 = ks.extent(intent)
        assert extent == extent2
        
   # assert len(arr) > 9

@given(arr_strat)
def test_mf(kernel_classes, arr):
    df = pd.DataFrame(arr)
    for K in kernel_classes:
        if K == krn.KernelSystemNP:
            ks = K(arr)
        else:
            ks = K(df)
        mf = ks.minusframe()
        assert len(mf) == arr.shape[0], 'error in ' + K.__name__
        for i in mf.index:
            row = mf.loc[i]
            assert row['w'] >= np.sum(arr[i]), 'error in ' + K.__name__
            assert row['w'] <= np.sum(arr[i])*len(arr), 'error in ' + K.__name__


def test_cc():
    bin_slr = np.array([[0, 0, 1, 1, 1, 1, 0],                   
                   [0, 0, 0, 1, 1, 1, 1],
                   [1, 0, 1, 0, 1, 1, 1],
                   [1, 0, 1, 1, 1, 0, 0],
                   [1, 1, 1, 0, 1, 0, 0],
                   [1, 1, 1, 1, 1, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0],
                   [1, 1, 1, 0, 0, 0, 0]])
    bin_three_aspects = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                                  [1, 1, 0, 0, 1, 1, 1, 0],
                                  [1, 1, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 1, 0, 0, 1, 1, 1, 0],
                                  [0, 0, 0, 0, 1, 1, 1, 0],
                                  [0, 0, 0, 0, 1, 1, 1, 0],
                                  [0, 0, 0, 0, 1, 1, 1, 0]])
    for System in [krn.FCASystemDF, 
                   krn.FCAPathSystemDF, 
                   krn.FreqLexiSeriateSystem, 
                   krn.ConfLexiSeriateSystem,
                   krn.LexiSystem,
                   krn.Lexi2System,
                   lambda data: krn.FreqLexiSeriateSystem(data, refill=True),
                   lambda data: krn.ConfLexiSeriateSystem(data, refill=True)]:               
        ks = System(pd.DataFrame(bin_slr))  
        ccc = ks.conceptchaincover()
        assert ccc[0][0].area() > 15
    #    ks = System(bin_slr)  
        if not isinstance(ks, krn.LexiSystem):
            ccc2 = ks.conceptchaincover_v2()
            assert ccc2[0][0].area() > 15
        
        ks2 = System(pd.DataFrame(bin_three_aspects))
        chainlist, _ = ks2.conceptchaincover(uncovered=0.0)
        assert 1 < len(chainlist) < 6  # Ideally 3 chains
        
        
    ks = krn.FCAPathSystemDF(pd.DataFrame(bin_slr))
    cr = ks.conceptrec([])
    assert cr.intent == set()
    assert cr.extent == set(range(8))
    cr1 = ks.conceptrec([1])
    assert ks.conceptdist(cr, cr1) == 4*2
    
    for System in [krn.LexiSystem, krn.Lexi2System]:
        ks = System(pd.DataFrame(bin_three_aspects))
        ccc, _ = ks.conceptchaincover(uncovered=0.0, min_cost=True)
        assert 0 < len(ccc) < 3 # Test cost minimization

    
    
    
def test_ConceptChain():
    bin_slr = np.array([[0, 0, 1, 1, 1, 1, 0],                   
                   [0, 0, 0, 1, 1, 1, 1],
                   [1, 0, 1, 0, 1, 1, 1],
                   [1, 0, 1, 1, 1, 0, 0],
                   [1, 1, 1, 0, 1, 0, 0],
                   [1, 1, 1, 1, 1, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0],
                   [1, 1, 1, 0, 0, 0, 0]])
    ks = krn.FCASystemDF(pd.DataFrame(bin_slr))  
    cc = krn.ConceptChain([1, 2, 3, 6], ks)
    assert len(cc) == 3
    assert [(len(e), len(i)) for e, i in cc] == [(1,4), (2,3), (6,1)]
    ccT = krn.ConceptChain.intent_init([1, 2, 3, 6], ks)
    assert len(ccT) == 3
    assert [(len(e), len(i)) for e, i in ccT] == [(1,5), (3,3), (4,2)]
    #assert ccT.intent_labels() == [1, 2, 3, 6]


@given(st.integers(), st.integers())
def test_ints_are_commutative(x, y):
    assert x + y == y + x
    assert x * y == y * x
    #assert x / y == 1 / (y / x)

@pytest.fixture(scope='function')
def some_list():
    return list(range(5))

def test_answer():
    assert 5 == 5

def test_answer2(some_list):
    assert len(some_list) == 5

if __name__ == "__main__":
    pytest.main()
   