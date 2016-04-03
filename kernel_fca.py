"""
Greedy FCA concept cover for binary arrays. Uses iterative MS kernels.
MS minus technique and kernel calculation.
FCA intent, extent calculation.

A. Torim,  2016
"""

import numpy as np
import pandas as pd


def minusframe(bin_arr):
    """
    Returns w(eight) i(ndex of removal) minus technique dataframe 
    for bin_arr 0, 1 numpy array
    """
    r, c = bin_arr.shape
    freq_v = np.dot(np.ones(r), bin_arr)
    weight_v = np.dot(freq_v * bin_arr, np.ones(c))
    result = pd.DataFrame({'w': weight_v})
    result['i'] = -1
    for i in range(r):
        min_idx = (result[result['i'] < 0] )['w'].idxmin()
        result.loc[min_idx, 'i'] = i
        freq_v -= bin_arr[min_idx]
        result.loc[result['i'] < 0, 'w'] = pd.Series(np.dot(freq_v * bin_arr, np.ones(c)))
    return result
    
def kernel(bin_arr): 
    """
    Returns kernel row indices for bin_arr
    """
    mf = minusframe(bin_arr)
    mf.sort('i', inplace=True, ascending=False) # prefer smaller extents
    max_wi = mf['w'].idxmax()
    max_i = mf.loc[max_wi, 'i']
    return mf[mf['i'] >= max_i].index

def intent(bin_arr, extent):
    bin_ext = np.prod(bin_arr[extent], axis=0)
    return [i for i, b in enumerate(bin_ext) if b==1]

def extent(bin_arr, intent):
    bin_ext = np.prod(bin_arr[:, intent], axis=1)
    return [i for i, b in enumerate(bin_ext) if b==1]

def removed(bin_arr, extent, intent):
    """
    Returns bin_arr copy with 1-s replaced with 0-s for 
    (extent, intent) concept.
    """
    result = np.copy(bin_arr)
    ree = result[extent]
    ree[:, intent] = 0
    result[extent] = ree
    return result

def conceptcover(bin_arr, limit=1, uncovered=0.1):
    """
    Returns a set of concepts covering bin_arr.
    Limit is minimal area.
    Uncovered is a ratio of 1-s that can be left uncovered.
    """
    arr = np.copy(bin_arr)
    arr_sum = np.sum(arr)
    result = []
    while True:
        k = kernel(arr)
        i = intent(bin_arr, k)
        e = extent(bin_arr, i)
        if len(e)*len(i) < limit or (e, i) in result: break
        result.append((e, i))
        arr = removed(arr, e, i)
        if np.sum(arr)/arr_sum < uncovered: break
    return result
        
    
    
if __name__ == "__main__":
    """
    mf = minusframe(bin_arr)
    #mf.loc[mf['w'] > 4, 'w'] = 99
    print(mf)
    k = kernel(bin_arr)
    print(k)
    i = intent(bin_arr, k)
    print(i)
    e = extent(bin_arr, i)
    print(e)
    #print(bin_arr[e] [:, i])
    new_arr = removed(bin_arr, e, i)
    print(new_arr)
    print(np.sum(new_arr))    
    """
    bin_arr = np.array([[1, 0, 1, 1],
                   [1, 0, 1, 1],
                   [0, 0, 0, 1],
                   [1, 1, 0, 0],
                   [1, 0, 0, 0]])
    cc = conceptcover(bin_arr)
    print(cc)
    cc_t = conceptcover(np.transpose(bin_arr))
    print(cc_t)
    bin_slr = np.array([[0, 0, 1, 1, 1, 1, 0],                   
                   [0, 0, 0, 1, 1, 1, 1],
                   [1, 0, 1, 0, 1, 1, 1],
                   [1, 0, 1, 1, 1, 0, 0],
                   [1, 1, 1, 0, 1, 0, 0],
                   [1, 1, 1, 1, 1, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0],
                   [1, 1, 1, 0, 0, 0, 0]])
    cc = conceptcover(bin_slr)
    for e, i in cc: print(e, i)
    print("-------------------")
    cc_t = conceptcover(np.transpose(bin_slr))
    for i, e in cc_t: print(e, i)
    
    
