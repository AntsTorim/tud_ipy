"""
Greedy FCA concept cover for binary arrays. Uses iterative MS kernels.
MS minus technique and kernel calculation.
FCA intent, extent calculation.

A. Torim,  2016
"""

import numpy as np
import pandas as pd



class KernelSystemNP:
    
    
    def __init__(self, data):
        self.data = data
        self.factory =KernelSystemNP
    
    def minusframe(self):
        """
        Returns w(eight) i(ndex of removal) minus technique dataframe 
        for data 0, 1 numpy array
        """
        r, c = self.data.shape
        freq_v = np.dot(np.ones(r), self.data)
        weight_v = np.dot(freq_v * self.data, np.ones(c))
        result = self.new_minusframe(weight_v)
        for i in range(r):
            min_idx = (result[result['i'] < 0] )['w'].idxmin()
            result.loc[min_idx, 'i'] = i
            freq_v -= self.datarow(min_idx)
            result.loc[result['i'] < 0, 'w'] = pd.Series(np.dot(freq_v * self.data, np.ones(c)), index=result.index)
        return result
     
    def new_minusframe(self, weight_v):
         result = pd.DataFrame({'w': weight_v})
         result['i'] = -1
         return result
     
    def datarow(self, i):
        """
        Data row with index i
        """
        return self.data[i]
        
    def kernel(self): 
        """
        Returns kernel row indices for data
        """
        mf = self.minusframe()
        mf.sort('i', inplace=True, ascending=False) # prefer smaller extents
        max_wi = mf['w'].idxmax()
        max_i = mf.loc[max_wi, 'i']
        return mf[mf['i'] >= max_i].index
    
    def intent(self, extent):
        bin_int = np.prod(self.data[extent], axis=0)
        return [i for i, b in enumerate(bin_int) if b==1]
    
    def extent(self, intent):
        bin_ext = np.prod(self.data[:, intent], axis=1)
        return [i for i, b in enumerate(bin_ext) if b==1]
    
    def removed(self, extent, intent):
        """
        Returns data copy with 1-s replaced with 0-s for 
        (extent, intent) concept.
        """
        result = self.datacopy()
        ree = result[extent]
        ree[:, intent] = 0
        result[extent] = ree
        return result
    
    def datacopy(self):
        """
        Return the copy of self.data
        """
        return np.copy(self.data)
   
    def totalsum(self, arr): return np.sum(arr)
   
    def conceptcover(self, limit=1, uncovered=0.1):
        """
        Returns a set of concepts covering data.
        Limit is minimal area.
        Uncovered is a ratio of 1-s that can be left uncovered.
        """
        arr = self.datacopy()
        arr_sum = self.totalsum(arr)
        result = []
        while True:
            ks = self.factory(arr)
            k = ks.kernel()
            #print(k)
            #print('new intent len:', len(ks.intent(k)))
            i = self.intent(k)
            e = self.extent(i)
            if len(e)*len(i) < limit or (e, i) in result: 
              #  print("break (e,i):", e, i, len(result))
                break
            result.append((e, i))
            arr = ks.removed(e, i)
            #print(self.totalsum(arr))
            if self.totalsum(arr)/arr_sum < uncovered: 
               # print("Total uncovered: ", self.totalsum(arr), "/", arr_sum)
                break
        return result


        
class KernelSystemDF(KernelSystemNP):
    
    def __init__(self, data):
        self.data = data
        self.factory =KernelSystemDF
    
    def intent(self, extent):
        bin_int = np.prod(self.data.loc[extent], axis=0).astype(bool) # .loc
        return list(self.data.columns[bin_int]) # ?
    
    def extent(self, intent):
        bin_ext = np.prod(self.data.loc[:, intent], axis=1).astype(bool) # .loc
        return list(self.data.index[bin_ext])
                
    def datarow(self, i):
        """
        Data row with index i
        """
        return self.data.loc[i]
    
    def new_minusframe(self, weight_v):
         result = pd.DataFrame({'w': weight_v}, index=self.data.index)
         result['i'] = -1
        # print(result)
         return result
         
    def datacopy(self): return self.data.copy()
    
    def removed(self, extent, intent):
        """
        Returns data copy with 1-s replaced with 0-s for 
        (extent, intent) concept.
        """
        result = self.datacopy()
        result.loc[extent, intent] = 0
        return result
        
    def totalsum(self, arr):
        return arr.sum().sum()


class FCASystemDF(KernelSystemDF):

        def __init__(self, data):
            self.data = data
            self.factory =FCASystemDF
            
        def kernel(self): 
            """
            Returns extent for the largest area concept in minus sequence
            """
            mf = self.minusframe()
            mf.sort('i', inplace=True, ascending=False) # prefer smaller extents
            best_area = -1
            current_extent = []
            for r in mf.index:
                current_seq = current_extent + [r]
                current_i = self.intent(current_seq)
                current_e = self.extent(current_i)
                current_area = len(current_i) * len(current_e)
                if current_area > best_area:
                    best_area = current_area
                    result = current_e
            return result
                

        
if __name__ == "__main__":
    """
    mf = minusframe(data)
    #mf.loc[mf['w'] > 4, 'w'] = 99
    print(mf)
    k = kernel(data)
    print(k)
    i = intent(data, k)
    print(i)
    e = extent(data, i)
    print(e)
    #print(data[e] [:, i])
    new_arr = removed(data, e, i)
    print(new_arr)
    print(np.sum(new_arr))    
    """
    
    data = np.array([[1, 0, 1, 1],
                   [1, 0, 1, 1],
                   [0, 0, 0, 1],
                   [1, 1, 0, 0],
                   [1, 0, 0, 0]])
    df = pd.DataFrame(data, index=['i','ii', 'iii', 'iv', 'v'], columns=['a','b','c','d'])
    #df = pd.DataFrame(data, columns=['a','b','c','d'])
    #ks = KernelSystemDF(df)
    ks = FCASystemDF(df)
    select = np.array([0,1,1,0]).astype(bool)
    i = ks.intent(["i", "ii"])
    #i = ks.intent([0, 1])
    e = ks.extent(i)
    print(e, i)
    print(ks.removed(e, i))
    mf = ks.minusframe()
    print(mf)
    k = ks.kernel()
    print("Kernel:", k)
    cc = ks.conceptcover()
    print(cc)
    
    ks = KernelSystemNP(data)
    cc = ks.conceptcover()
    print(cc)
    ks = KernelSystemNP(np.transpose(data))
    cc_t = ks.conceptcover()
    print(cc_t)
    bin_slr = np.array([[0, 0, 1, 1, 1, 1, 0],                   
                   [0, 0, 0, 1, 1, 1, 1],
                   [1, 0, 1, 0, 1, 1, 1],
                   [1, 0, 1, 1, 1, 0, 0],
                   [1, 1, 1, 0, 1, 0, 0],
                   [1, 1, 1, 1, 1, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0],
                   [1, 1, 1, 0, 0, 0, 0]])
    ks = KernelSystemNP(bin_slr)
    cc = ks.conceptcover()
    for e, i in cc: print(e, i)
    print("-------------------")
    ks = KernelSystemNP(np.transpose(bin_slr))
    cc_t = ks.conceptcover()
    for i, e in cc_t: print(e, i)
    
    
