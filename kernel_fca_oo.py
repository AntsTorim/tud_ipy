"""
Greedy FCA concept cover for binary arrays. Uses iterative MS kernels.
MS minus technique and kernel calculation.
FCA intent, extent calculation.

A. Torim,  2016
"""

import numpy as np
import pandas as pd
import heapq
import sortseriate
from sklearn.cluster import KMeans
import sklearn.base as sk
import copy

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
        mf.sort_values('i', inplace=True, ascending=False) # prefer smaller extents
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
            mf.sort_values('i', inplace=True, ascending=False) # prefer smaller extents
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
                
        def conceptchaincover(self, uncovered=0.1, max_cc=20):
            """
            Returns a set of concept chains covering the data and a list of 
            corresponding ratios of data table left uncovered.
  
            Uncovered is the maximal allowed ratio of uncovered 1-s.
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
                if (old_sum == new_sum) or (new_sum/arr_sum < uncovered) or len(result) >= max_cc: 
                   # print("Total uncovered: ", self.totalsum(arr), "/", arr_sum)
                    break
                old_sum = new_sum
            return result, uncovered_list


        def conceptchaincover_v2(self, uncovered=0.1, max_cc=20):
            """
            Returns a set of concept chains covering the data and a list of 
            corresponding ratios of data table left uncovered.
            For finding new concept chains connecting elements,
            that is 1-s that have any uncovered 1s in their rows and columns,
            are retained (connected array).
  
            Uncovered is the maximal allowed ratio of uncovered 1-s.
            More complex than v1, cover comparison with it mixed.
            """
            arr = connected = self.datacopy()
            old_sum = arr_sum = self.totalsum(arr)
            result = []
            uncovered_list = []
            while True:
                ks = self.factory(connected)
                cc = ks.get_conceptchain()
                result.append(cc)
                for e, i in cc:
                    ks = self.factory(arr)
                    arr = ks.removed(e, i)
                new_sum = self.totalsum(arr)
                connected = self.data * arr.any(axis=0)
                connected = connected.mul(arr.any(axis=1), axis='index')
                connected = connected + arr
                uncovered_list.append(new_sum/arr_sum)
                if (old_sum == new_sum) or (new_sum/arr_sum < uncovered) or len(result) >= max_cc: 
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
    
    @staticmethod
    def intent_init(attr_seq, ks): 
        ksT = copy.deepcopy(ks)
        ksT.data = ksT.data.T
        return ConceptChain(attr_seq, ksT).T
        
    
    def __init__(self, objectseq, ks):
        """
        ks is a kernel system
        objectseq is either a minusframe or a sequence of index labels for data
        """
        # If objectseq is minusframe-like then turn it into a sequence of index labels
        try:
            objectseq = list(objectseq.sort_values('i', ascending=False).index)
        except:
            pass
        self.seq = objectseq
        
        # Generate concepts
        extent = set()
        extent_obj = set()
        for obj in objectseq:
            extent_obj = extent  | {obj} 
            # If object is not in previous extent
            if len(extent_obj) > len(extent):
                # generate a concept
                intent = set(ks.intent(extent_obj))
                # We don' t deal with zero-coverage concepts
                if len(intent) == 0: break
                extent = set(ks.extent(intent))
                # add the concept
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
    
    def transpose(self):
        """
        Return transposed concept chain, where extent and intent have been swapped
        """
        result = copy.deepcopy(self)
        result.clear()
        for e, i in self:
            #result.append((i, e))
            result.insert(0, (i, e))
        return result

    @property
    def T(self): return self.transpose()
    


class _ConceptChainOld(ConceptChain):
    """
    Old implementation of CC generation.
    Does not add all objects into first and last extent.
    """
    
    def __init__(self, objectseq, ks):
        """
        ks is a kernel system
        objectseq is either a minusframe or a sequence of index labels for data
        """
        # If objectseq is minusframe-like then turn it into a sequence of index labels
        try:
            objectseq = list(objectseq.sort_values('i', ascending=False).index)
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
                #extent = extent  | set(ks.extent(obj_intent))
                intent = intent & obj_intent
            # else add object to extent
            else:
                extent.add(obj)
        # if final concept has non-zero intent then add it
        if len(intent) > 0:
            self.append((extent, intent))


class ConceptChainFromRec(ConceptChain):

    def __init__(self, bottomrec):
        activerec = bottomrec
        while activerec != None:
            self.append((activerec.extent, activerec.intent))
            activerec = activerec.prev


class _SlowFCAPathSystemDF(FCASystemDF): # For historic interest


    def __init__(self, data):
        self.data = data
        self.factory = _SlowFCAPathSystemDF
   
   
    def dijkstra_gen(self, top):
        """
        Yield concepts where shortest path has been found
        """
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
            candidates = [c for c in concepts.values() if not c.visited]
            if len(candidates)>0:
                current = min(candidates, key=lambda c: c.dist)
                yield current
            else:
                raise StopIteration
       
       
   
    def get_conceptchain(self):
        """
        Get the best conceptchain.  Here we use Dijkstras algorithm.
        """
        # Initialize top (start) and bottom (target) concepts
        top = self.conceptrec([])
        bottom = self.conceptrec(self.data.columns)
        for current in self.dijkstra_gen(top):
            # If current is bottom:
            if current.intent == bottom.intent:
                return ConceptChainFromRec(current)

                 


    def conceptrec(self, protointent):
        extent = self.extent(protointent)
        intent = self.intent(extent)
        return ConceptRec(intent, extent)


    def conceptdist(self, c1, c2):
        """ 
        'Distance' (not correct mathematical distance metric) from 
        concept c1 to concept c2. c2 should be below c1.
        Distance is |c2.intent - c1.intent| * (|data| - |c2.extent|)
        """
        assert c2.extent <= c1.extent
        return len(c2.intent - c1.intent) * (len(self.data) - len(c2.extent))


class PriorityQueue:
    """@author: Teacode
    https://teacode.wordpress.com/2013/08/02/algo-week-5-heap-and-dijkstras-shortest-path/
    Priority queue based on heap, capable of inserting a new node with
    desired priority, updating the priority of an existing node and deleting
    an abitrary node while keeping invariant"""
 
    def __init__(self, heap=[]):
        """if 'heap' is not empty, make sure it's heapified"""
 
        heapq.heapify(heap)
        self.heap = heap
        self.entry_finder = dict({i[-1]: i for i in heap})
        #self.REMOVED = '<remove_marker>'
        self.REMOVED = ConceptRec(['remove',],[])
  
    def __contains__(self, node):
        """
        Override 'in' operator
        """
        return node in self.entry_finder      
  
    def __getitem__(self, key):
        """
        Override pq[key] access
        """
        return self.entry_finder[key][1]
  
    def insert(self, node, priority=0):
        """'entry_finder' bookkeeps all valid entries, which are bonded in
        'heap'. Changing an entry in either leads to changes in both."""
 
        if node in self:
            self.delete(node)
        entry = [priority, node]
        self.entry_finder[node] = entry
        heapq.heappush(self.heap, entry)
 
    def delete(self, node):
        """Instead of breaking invariant by direct removal of an entry, mark
        the entry as "REMOVED" in 'heap' and remove it from 'entry_finder'.
        Logic in 'pop()' properly takes care of the deleted nodes.
        
        Returns the priority.
        
        """
 
        entry = self.entry_finder.pop(node)
        entry[-1] = self.REMOVED
        return entry[0]
 
    def pop(self):
        """Any popped node marked by "REMOVED" does not return, the deleted
        nodes might be popped or still in heap, either case is fine."""
 
        while self.heap:
            priority, node = heapq.heappop(self.heap)
            if node != self.REMOVED:
                del self.entry_finder[node]
                return priority, node
        raise KeyError('pop from an empty priority queue')


class FCAPathSystemDF(FCASystemDF):


    def __init__(self, data):
        self.data = data
        self.factory = FCAPathSystemDF
   
   
    def dijkstra_gen(self, top):
        """
        Yield concepts where shortest path has been found
        """
        # Initialize the concept dict (intent: extent, distance, previous, visited
        visited_dict = {top.intent: top}
        
        # Initialize candidates priority Queue
        candidates = PriorityQueue()

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
                if not ca.intent in visited_dict:
                    # Calculate its distance to current & start
                    dist = current.dist + self.conceptdist(current, ca)
                    # If concept not in candidates or new distance is smaller than old distance
                    if not (ca in candidates) or (candidates[ca].dist > dist):
                        # add data to candidates
                        ca.dist = dist
                        ca.prev = current
                        candidates.insert(ca, dist)
            try:
                dist, current = candidates.pop()          
                yield current
            except KeyError: # Empty queue
                raise StopIteration
       
       
   
    def get_conceptchain(self):
        """
        Get the best conceptchain.  Here we use Dijkstras algorithm.
        """
        # Initialize top (start) and bottom (target) concepts
        top = self.conceptrec([])
        bottom = self.conceptrec(self.data.columns)
        for current in self.dijkstra_gen(top):
            # If current is bottom:
            if current.intent == bottom.intent:
                return ConceptChainFromRec(current)
            

    def conceptrec(self, protointent):
        extent = self.extent(protointent)
        intent = self.intent(extent)
        return ConceptRec(intent, extent)


    def conceptdist(self, c1, c2):
        """ 
        'Distance' (not correct mathematical distance metric) from 
        concept c1 to concept c2. c2 should be below c1.
        Distance is |c2.intent - c1.intent| * (|data| - |c2.extent|)
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

    def __hash__(self): return hash(self.intent)
    
    def __eq__(self, other): 
        try:
            return self.intent == other.intent
        except:
            return False
        
    def __lt__(self, other): # for heapq
        try:
            return self.intent < other.intent
        except:
            return False


class FCAPathSystem2Way(FCAPathSystemDF):
    
    def __init__(self, data):
        self.data = data
        self.factory = FCAPathSystem2Way
    
    def conceptdist(self, c1, c2):
        """
        Symmetrical distance function (top to bottom, bottom to top).
        Sum of distances is the area of data table not covered by the concept chain.
        Middle square between c1 and c2 has weight 1 and two projections have weight 1/2.
        """
        if c2.extent > c1.extent:
            c1, c2 = c2, c1
        de = len(c1.extent-c2.extent)
        di = len(c2.intent-c1.intent)
        lg, lm = self.data.shape
        weight = (di * de) + (de * (lm - len(c2.intent)) / 2) + (di * (lg - len(c1.extent)) / 2)
        return weight


class FreqLexiSeriateSystem(FCASystemDF):
    
    
    def __init__(self, data, refill=False, refiller=None):
        #import pdb
        self.data = data
        #self.factory = FreqLexiSeriateSystem
        self.refiller = refiller
        if refill: 
            #pdb.set_trace()
            if not refiller:
                self.refiller = sortseriate.Refiller()
                self.refiller.fit(data.values)
            self.factory = lambda data: self.factory_class()(data, refill=True, refiller=self.refiller)
        else:
            self.factory = self.factory_class()
    
    def factory_class(self): return FreqLexiSeriateSystem
    
    def get_conceptchain(self):
        """
        Get the best conceptchain. 
        """
        #fls = sortseriate.FreqLexiSeriation()
        fls = self.preseriate()
        fls.fit(self.data.values)
        cc = ConceptChain(self.data.index[fls.row_i], self)
        return cc
     
    def preseriate(self):
        return sortseriate.FreqLexiSeriation(refiller=self.refiller)


class ConfLexiSeriateSystem(FreqLexiSeriateSystem):
    
    #def __init__(self, data):
    #    self.data = data
    #    self.factory = ConfLexiSeriateSystem
    
    def factory_class(self): return ConfLexiSeriateSystem
    
    def preseriate(self):
        return sortseriate.FreqLexiSeriation(preseriate=sortseriate.Conf2DSeriation, refiller=self.refiller)



class LexiCCTransformer(sk.TransformerMixin):
    """
    An alternative sklearn style way to generate concept chains.
    fit(), transform() split should work well with fit(reduced), transform(original)
    """
    
    def __init__(self, refiller=None, transform="CL"):
        """
        Transform: CL (Conformism-Lexi), FL (Frequency-Lexi)
        """
        if transform == "CL":
            self.fls = sortseriate.FreqLexiSeriation(preseriate=sortseriate.Conf2DSeriation, refiller=refiller)
        elif transform == "FL":
            self.fls = sortseriate.FreqLexiSeriation(refiller=refiller)
            
       
    def fit(self, X):
        """
        X should be pandas DataFrame
        """
        self.fls.fit(X.values)
        return self
        
   
    def transform(self, X):
        """
        X should be pandas DataFrame
        """
        cc = ConceptChain(X.index[self.fls.row_i], FCASystemDF(X)) #ConceptChain takes a kernel system so FCASystemDF is good enough
        return cc
        
    
class Lexi2CCTransformer(LexiCCTransformer):
    """
    Adds second lexicographic sort (instead of refill) 
    while transforming full data
    """
    
    
    def transform(self, X):
        """
        X should be pandas DataFrame
        """
        lis = sortseriate.LexiIterSeriation()
        lis.fit(self.fls.transform(X.values))
        row_i = self.fls.row_i[lis.row_i]
        cc = ConceptChain(X.index[row_i], FCASystemDF(X)) #ConceptChain takes a kernel system so FCASystemDF is good enough
        return cc
    
    
    
class LexiSystem(FCASystemDF):
    """
    A new style FL, CL seriate system
    Has own methods to get cc and to get cc cover
    """    
    
      
    def __init__(self, data, transform = "CL", full_lexi=True, refill=False, refiller=None):
        #import pdb
        self.data = data
        self.refiller = refiller
        if refill and not refiller:
            self.refiller = sortseriate.Refiller()
            self.refiller.fit(data.values)
        if full_lexi:
            self.transformer =  Lexi2CCTransformer(refiller=self.refiller, transform=transform)
        else:
            self.transformer =  LexiCCTransformer(refiller=self.refiller, transform=transform)
                
    
    def get_conceptchain(self):
        """
        Get the best conceptchain. 
        """
        return self.transformer.fit_transform(self.data)
     
    
    def conceptchaincover(self, uncovered=0.1, max_cc=20, min_cost=False):
        """
        Returns a set of concept chains covering the data and a list of 
        corresponding ratios of data table left uncovered.
  
        Uncovered is the maximal allowed ratio of uncovered 1-s.
        min_cost sets the cost regularization cutoff
        """
        #arr = self.datacopy()
        #old_sum = arr_sum = self.totalsum(arr)
        result = []
        uncovered_list = []
        if min_cost:
            l = min(self.data.shape)
            c_min = 1.0
            k_min = 0
        #while True:
        for cc, u in self.gen_conceptchaincover():
           # self.transformer.fit(arr)
           # cc = self.transformer.transform(self.data)
            result.append(cc)
            uncovered_list.append(u)
            #for e, i in cc:
                # remove concept 1s
             #   arr.loc[e, i] = 0
           # new_sum = self.totalsum(arr)
           # uncovered_list.append(new_sum/arr_sum)
            k = len(result)
            if min_cost:
                kl = (k/l)**2 
                if kl >= c_min:
                    return result[:k_min], uncovered_list[:k_min]
                elif (kl + u**2) < c_min:
                    c_min = kl + u**2
                    k_min = len(result)
            elif (u < uncovered) or len(result) >= max_cc: 
               # print("Total uncovered: ", self.totalsum(arr), "/", arr_sum)
                break
            #old_sum = new_sum
        return result, uncovered_list    


    def gen_conceptchaincover(self):
        """
        Yields pairs of concept chains and uncovered ratios.
  
        """
        arr = self.datacopy()
        old_sum = arr_sum = self.totalsum(arr)
        while True:
            cc = self.chain_from_array(arr) 
            for e, i in cc:
                # remove concept 1s
                arr.loc[e, i] = 0
            new_sum = self.totalsum(arr)
            u = new_sum/arr_sum
            yield cc, u
            if (old_sum == new_sum) or new_sum == 0: 
               # print("Total uncovered: ", self.totalsum(arr), "/", arr_sum)
                break
            old_sum = new_sum
        

    def chain_from_array(self, arr):
        """ Get the best chain from (reduced) array"""
        self.transformer.fit(arr)
        return self.transformer.transform(self.data)
        


class Lexi2System(LexiSystem):
    """
    Adds second phase seriation on full array
    """
    
    pass


class KMeansSystem(KernelSystemDF):
    
    def __init__(self, data, n_chains=8, random_state=None):
        self.data = data
        self.n_chains = n_chains
        self.random_state = random_state


    def conceptchaincover(self, uncovered=0.1):
        """
        n_chains: Number of chains based on Kmeans clusters
        random_state: None: undeterministic; int: deterministic seed
        Uncovered, max_cc are ignored.
        """
                        
        # Find clusters
        n_chains = self.n_chains
        kmeans = KMeans(n_clusters=n_chains, random_state=self.random_state).fit(self.data)
        chains = []
        
        # Initialize array and vars for uncovered % calculation
        arr = self.datacopy()
        arr_sum = self.totalsum(arr)
        uncovered_list = []
        
        # For each cluster
        for i in range(n_chains):
            
            # Find the concept for the cluster    
            extent = list(self.data[kmeans.labels_ == i].index)
            intent = self.intent(extent)
            extent = self.extent(intent)
            
            # Find chain up
            chain_up = self.chain(extent)
            
            # Find chain down
            chain_down = self.chain(intent, is_intent=True)
            
            # Combine
            cc = chain_up
            for c in chain_down[-2::-1]: #backwards, ignore last (identical)
                cc.append(c)
            
            # Calculate uncovered
            for extent, intent in cc:
                arr.loc[extent, intent] = 0
            uncovered_list.append(arr.values.sum() / arr_sum)
                        
            # Add chain to the cover
            chains.append(cc)
            
        # Return the cover
        return chains, uncovered_list
            
     
    def chain(self, extintent, is_intent=False):
        """
        Return the concept chain corresponding to extent or intent.
        If we have extent then chain to top, if intent then chain to bottom.
        """
        
        # If intent then transpose
        if is_intent:
            ks = copy.deepcopy(self)
            ks.data = ks.data.T
        else:
            ks = self
            
        # Find selection
        selection_df = ks.data.loc[extintent]
        
        # Seriate it 
        sort_extintent = self.seriate(selection_df)
        
        # Generate chain
        cc = ConceptChain(sort_extintent, ks)
        
         # If intent, then transpose back
        if is_intent: cc = cc.T
        
        #print("chain", extintent, is_intent, cc)
        return cc
  
      
    def seriate(self, df):
        """
        Return seriated indices for dataframe
        TBD: has turned buggy!        
        """
        sort_i = sortseriate.ConfSeriation().fit(df.values).sortindices_
        return df.iloc[sort_i].index
        

def main():
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
    for System in FCAPathSystemDF, FCAPathSystem2Way, FCASystemDF:
        print("\nSystem ", System)
        ps = System(df)
        print("Intent ", ps.intent([]))
        #print(list(df.index))
        #cr = ps.conceptrec([])
        #print(cr.intent, cr.extent)
        cc = ps.get_conceptchain()
        print("Chain ",  cc)
        print("Chain areas ",  cc.concept_areas())
        ccc, uc = ps.conceptchaincover()
        print("CC cover (intent, areas, maxima):")
        for c, u in zip(ccc, uc):
            print(c.intent_labels(), c.concept_areas(), c.local_maxima())
            print("Uncovered", u)



def main_sk_style():
    andmed = np.array([[0, 0, 1, 1, 1, 1, 0],                   
                   [0, 0, 0, 1, 1, 1, 1],
                   [1, 0, 1, 0, 1, 1, 1],
                   [1, 0, 1, 1, 1, 0, 0],
                   [1, 1, 1, 0, 1, 0, 0],
                   [1, 1, 1, 1, 1, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0],
                   [1, 1, 1, 0, 0, 0, 0]])
    #andmed = pd.DataFrame(andmed) # Teeme DataFrameks
    andmed = pd.DataFrame(andmed, index=['i','ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii'], 
                           columns=['a','b','c', 'd', 'e', 'f', 'g'])
    #lcct = LexiCCTransformer()
    #lcct.fit(andmed)
    #cc = lcct.transform(andmed)
    ls = LexiSystem(andmed, refill=True)
    for kontsept in ls.get_conceptchain():
        ekstent, intent = kontsept
        print("Ekstent:", ekstent)
        print("Intent:", intent)
    ahelate_list, katmata_protsendi_list = ls.conceptchaincover()
    print(katmata_protsendi_list)
    for ahel in ahelate_list:
        print("\nAhel:")
        for kontsept in ahel:
            ekstent, intent = kontsept
            print("Ekstent:", ekstent)
            print("Intent:", intent)

    
     
    

def simple_main():
     andmed = np.array([[0, 0, 1, 1, 1, 1, 0],                   
                   [0, 0, 0, 1, 1, 1, 1],
                   [1, 0, 1, 0, 1, 1, 1],
                   [1, 0, 1, 1, 1, 0, 0],
                   [1, 1, 1, 0, 1, 0, 0],
                   [1, 1, 1, 1, 1, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0],
                   [1, 1, 1, 0, 0, 0, 0]])
     #andmed = np.array([[1, 0, 1], [0, 0, 1]])
     
     #andmed = pd.DataFrame(andmed) # Teeme DataFrameks
     andmed = pd.DataFrame(andmed, index=['i','ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii'], 
                           columns=['a','b','c', 'd', 'e', 'f', 'g'])
     #katte_systeem = FCAPathSystem2Way(andmed) # Dijkstra algoritmi pohine
     #katte_systeem = FCASystemDF(andmed) # Monotoonsets syst miinustehnika pohine
     #katte_systeem = FreqLexiSeriateSystem(andmed)
     katte_systeem = ConfLexiSeriateSystem(andmed)
     kate = katte_systeem.conceptchaincover()
     """
     import cProfile
     cProfile.runctx('katte_systeem.conceptchaincover()', 
                     globals={'katte_systeem': katte_systeem},
                     locals={})
     """
     print(kate)
     
     # Jalutame katte elemendid ykshaaval labi
     ahelate_list, katmata_protsendi_list = kate
     print(katmata_protsendi_list)
     for ahel in ahelate_list:
         print("\nAhel:")
         for kontsept in ahel:
             ekstent, intent = kontsept
             print("Ekstent:", ekstent)
             print("Intent:", intent)
     

def simple_main_kmeans():
     andmed = np.array([[0, 0, 1, 1, 1, 1, 0],                   
                   [0, 0, 0, 1, 1, 1, 1],
                   [1, 0, 1, 0, 1, 1, 1],
                   [1, 0, 1, 1, 1, 0, 0],
                   [1, 1, 1, 0, 1, 0, 0],
                   [1, 1, 1, 1, 1, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0],
                   [1, 1, 1, 0, 0, 0, 0]])
     #andmed = np.array([[1, 0, 1], [0, 0, 1]])
     
     #andmed = pd.DataFrame(andmed) # Teeme DataFrameks
     andmed = pd.DataFrame(andmed, index=['i','ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii'], 
                           columns=['a','b','c', 'd', 'e', 'f', 'g'])
     katte_systeem = KMeansSystem(andmed, n_chains=3)
     kate = katte_systeem.conceptchaincover()
     ahelate_list, katmata_protsendi_list = kate
     for ahel, katmata in zip(ahelate_list, katmata_protsendi_list):
         print("\nAhel:")
         for kontsept in ahel:
             ekstent, intent = kontsept
             print("Ekstent:", ekstent)
             print("Intent:", intent)
         print("Katmata:", katmata)
             

def simple_lexi():
     andmed = np.array([[0, 0, 1, 1, 1, 1, 0],                   
                   [0, 0, 0, 1, 1, 1, 1],
                   [1, 0, 1, 0, 1, 1, 1],
                   [1, 0, 1, 1, 1, 0, 0],
                   [1, 1, 1, 0, 1, 0, 0],
                   [1, 1, 1, 1, 1, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0],
                   [1, 1, 1, 0, 0, 0, 0]])

     andmed = pd.DataFrame(andmed, index=['i','ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii'], 
                           columns=['a','b','c', 'd', 'e', 'f', 'g'])
     katte_systeem = Lexi2System(andmed)
            
     # Jalutame katte elemendid ykshaaval labi

     for ahel, u in katte_systeem.gen_conceptchaincover():
         print("\nAhel, katmata %f:" % u)
         for kontsept in ahel:
             ekstent, intent = kontsept
             print("Ekstent:", ekstent)
             print("Intent:", intent)


if __name__ == "__main__":
    simple_lexi()
    #simple_main_kmeans()
    #simple_main()
    #main_sk_style()
    #main()

    
    
