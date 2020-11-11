# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 13:16:45 2020

@author: ator-
"""

from hypothesis import given, settings
import hypothesis.strategies as st
import hypothesis.extra.numpy as st_np

import numpy as np

import pandas as pd

import kernel_fca_oo as krn


bin_int_strat = st.integers(min_value=0, max_value=1)

results = []

@given(st_np.arrays(np.int8, (20, 20), elements=bin_int_strat))
@settings(max_examples=100, deadline=5000)
def show_T_superiority(arr):
    #print(arr)
    data1 = pd.DataFrame(arr)
    data2 = pd.DataFrame(arr.T)
    ls1 = krn.LexiSystem(data1, full_lexi=False)
    ls2 = krn.LexiSystem(data2, full_lexi=False)
    ccc1, _ = ls1.conceptchaincover(uncovered=0.0)
    ccc2, _ = ls2.conceptchaincover(uncovered=0.0)
    #print(ccc1, ccc2)
    l1 = len(ccc1)
    l2 = len(ccc2)
    #if l1 != l2:
      #  print(l1, l2)
    results.append((l1, l2))
    
show_T_superiority()
print(results)
print(len([x for x,y in results if x!=y]) / len(results))
    