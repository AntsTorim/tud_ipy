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
        extent = np.array(list(extent)) # transform
        if len(extent) > 0:
            subdata = self.data[extent]
        else: # empty extent
            return list(range(self.data.shape[1]))
        bin_int = np.prod(subdata, axis=0)
        return [i for i, b in enumerate(bin_int) if b==1]
    
    def extent(self, intent):
        intent = np.array(list(intent)) # transform
        if len(intent) > 0:
            subdata = self.data[:, intent]
        else: # empty intent
            return list(range(self.data.shape[0]))
        bin_ext = np.prod(subdata, axis=1)
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
        if len(extent) > 0:
            subdata = self.data.loc[extent]
        else: # empty extent
            return list(self.data.columns)
        bin_int = np.prod(subdata, axis=0).astype(bool) # .loc
        #bin_int = np.prod(self.data.loc[extent], axis=0).astype(bool) # .loc
        return list(self.data.columns[bin_int]) # ?
    
    def extent(self, intent):
        if len(intent) > 0:
            subdata = self.data.loc[:, intent]
        else: # empty extent
           # return list(range(self.data.shape[0]))
            return list(self.data.index)
        bin_ext = np.prod(subdata, axis=1).astype(bool) # .loc
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
            self.factory = FCASystemDF
            
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
                
        def conceptchaincover(self, uncovered=0.1):
            """
            Returns a set of concept chains covering the data and a list of 
            corresponding ratios of data table left uncovered.
  
            Uncovered is a ratio of 1-s that can be left uncovered.
            """
            arr = self.datacopy()
            old_sum = arr_sum = self.totalsum(arr)
            result = []
            uncovered_list = []
            while True:
                ks = self.factory(arr)
                cc = ks.get_conceptchain()
                result.append(cc)
                for e, i in cc:
                    ks = self.factory(arr)
                    arr = ks.removed(e, i)
                new_sum = self.totalsum(arr)
                uncovered_list.append(new_sum/arr_sum)
                if (old_sum == new_sum) or (new_sum/arr_sum < uncovered): 
                   # print("Total uncovered: ", self.totalsum(arr), "/", arr_sum)
                    break
                old_sum = new_sum
            return result, uncovered_list

        def get_conceptchain(self):
            """
            Get the best conceptchain.  Here we use minus technique.
            """
            mf = self.minusframe()
            cc = ConceptChain(mf, self)
            return cc
            

class ConceptChain(list):
    """
    A chain of concepts
    """
    
    def __init__(self, objectseq, ks):
        """
        ks is a kernel system
        objectseq is either a minusframe or a sequence of index labels for data
        """
        # If objectseq is minusframe-like then turn it into a sequence of index labels
        try:
            objectseq = list(objectseq.sort('i', ascending=False).index)
        except:
            pass
        self.seq = objectseq
        
        # Generate concepts
        extent = set()
        intent = set(ks.data.columns)
        for obj in objectseq:
            # calculate {object}'
            obj_intent = set(ks.intent([obj]))
            # if it does not contain the whole intent:
            if len(intent - obj_intent) > 0:
                # add previous concept to list
                if len(extent) > 0:
                    self.append((extent, intent))
                    
                # generate the next concept:  extent = extent union {object}, intent = [intent intersection {object}']
                extent = extent  | {obj}
                intent = intent & obj_intent
            # else add object to extent
            else:
                extent.add(obj)
        # if final concept has non-zero intent then add it
        if len(intent) > 0:
            self.append((extent, intent))
            
        
    def extent_labels(self):
        prev_ext = set()
        result = []
        for ext, _ in self:
            result.append(ext - prev_ext)
            prev_ext = ext
        return result

        
    def intent_labels(self):
        prev_int = set()
        result = []
        for _, int in reversed(self):
            result.insert(0, int - prev_int) # should it be reversed?
            prev_int = int
        return result
        
    def concept_areas(self):
        return [len(e) * len(i) for e, i in self]
    
    def area(self):
        """ Total area for the chain """
        checked_rows = 0
        result = 0
        for e, i in self:
            result += len(i) * (len(e) - checked_rows)
            checked_rows = len(e)
        return result
    
    def local_maxima(self):
        """
        Returns indices of concepts with locally maximal areas in the chain
        """
        result = []
        areas = self.concept_areas()
        if len(self) == 0:
            return []
        elif len(self) == 1:
            return [0]
        elif areas[0] > areas[1]:
            result.append(0)
        for i in range(1, len(self)-1):
            if areas[i-1] <= areas[i] > areas[i+1]:
                result.append(i)
        if areas[-1] >= areas[-2]:
            result.append(len(self)-1)
        return result


class ConceptChainFromRec(ConceptChain):

    def __init__(self, bottomrec):
        activerec = bottomrec
        while activerec != None:
            self.append((activerec.extent, activerec.intent))
            activerec = activerec.prev



class FCAPathSystemDF(FCASystemDF):


    def __init__(self, data):
        self.data = data
        self.factory = FCASystemDF
   
   
    def get_conceptchain(self):
        """
        Get the best conceptchain.  Here we use Dijkstras algorithm.
        """
        # Initialize top (start) and bottom (target) concepts
        top = self.conceptrec([])
        bottom = self.conceptrec(self.data.columns)
        
        # Initialize the concept dict (intent: extent, distance, previous, visited
        concepts = {top.intent: top}
        
        # Initialize top-level concept as current
        current = top
        
        while True:
            attributes = set(self.data.columns) - current.intent
            checked_concepts = []
            # For each attribute\concept neighbouring current:
            for attr in attributes:
                # Generate the concept
                ca = self.conceptrec(current.intent | {attr})

                # If it has been checked, then continue to next attr
                if ca.intent in checked_concepts:
                    continue
                checked_concepts.append(ca.intent)

                # If it is not visited 
                if not (ca.intent in concepts and concepts[ca.intent].visited):
                    # Calculate its distance to current & start
                    dist = current.dist + self.conceptdist(current, ca)
                    # If concept in dict and new distance is smaller than distance in dict
                    if ca.intent in concepts and (dist < concepts[ca.intent].dist):
                        # Update dict with new data
                        concepts[ca.intent].dist = dist
                        concepts[ca.intent].prev = current
                    elif not ca.intent in concepts:
                        # add data to dict
                        ca.dist = dist
                        ca.prev = current
                        concepts[ca.intent] = ca
            
            current.visited = True            
            # Select active concept with minimum distance as current
            current = min([c for c in concepts.values() if not c.visited], 
                          key=lambda c: c.dist)
            # If current is bottom:
            if current.intent == bottom.intent:
                c = current
                while c != None:
                    print("current", c.extent)
                    c = c.prev
                # generate conceptchain and return it
                return ConceptChainFromRec(current)

                 


    def conceptrec(self, protointent):
        extent = self.extent(protointent)
        intent = self.intent(extent)
        return ConceptRec(intent, extent)


    def conceptdist(self, c1, c2):
        """ 
        'Distance' from concept c1 to concept c2. c2 should be below c1.
        Distance is |c2.intent - c1.intent| * |data| - |c2.extent|
        """
        assert c2.extent <= c1.extent
        return len(c2.intent - c1.intent) * (len(self.data) - len(c2.extent)) 

        
class ConceptRec:

    def __init__(self, intent, extent, dist=0, prev=None, visited=False):
        self.intent = frozenset(intent)
        self.extent = frozenset(extent)
        self.dist = dist
        self.prev = prev
        self.visited = visited
        
    def __repr__(self): return str(tuple(self.intent))


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
    #for row in mf.index: print(mf.loc[row])
    #cca = ConceptChain(mf, ks)
    cca = ConceptChain(['ii', 'i', 'iv'], ks)
    print(cca.seq)
    print(cca)
    print(cca.extent_labels(), cca.intent_labels(), cca.concept_areas())
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
    for e, i in cc: print(e, i)
    print("-------------------")
    ks = KernelSystemNP(np.transpose(bin_slr))
    cc_t = ks.conceptcover()
    for i, e in cc_t: print(e, i)
    
    k_big = FCASystemDF(pd.DataFrame(bin_slr))   
    cca = ConceptChain(k_big.minusframe(), k_big)
    print(cca.extent_labels(), cca.concept_areas())
    print('Chain area: ', cca.area())
    print(cca.local_maxima())
    ccc, uc = k_big.conceptchaincover()
    for c, u in zip(ccc, uc):
        #print(c)
        print(c.intent_labels(), c.concept_areas(), c.local_maxima())
        print(u)
        
    print("\nTest Path System\n")
    for System in FCAPathSystemDF, FCASystemDF:
        ps = System(df)
        print(ps.intent([]))
        #print(list(df.index))
        #cr = ps.conceptrec([])
        #print(cr.intent, cr.extent)
        cc = ps.get_conceptchain()
        print(cc)
        print(cc.concept_areas())
        ccc, uc = ps.conceptchaincover()
        for c, u in zip(ccc, uc):
            print(c.intent_labels(), c.concept_areas(), c.local_maxima())
            print(u)

    
    
