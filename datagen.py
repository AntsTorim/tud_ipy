# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 20:06:58 2019

@author: ator-
"""
from random import randrange, random
from numpy import fromfunction, vectorize
from pandas import DataFrame
from kernel_fca_oo import ConceptChain, FCASystemDF

def make_test_results(n_samples=100, 
                      n_tests=20, 
                      n_aspects=3,
                      aspect_range = 800,
                      sample_aspect0=-1200,
                      test_aspect0=-2400):
     """
     Generate a n_samples*n_tests binary data table. Samples and deatures have aspect strengths.
     Table is filled by comparing sample aspect strengths against test
     aspect strengths.
     
     Parameters
     ----------
     n_samples : int, optional (default=100)
        The number of samples/rows.

     n_features : int, optional (default=20)
        The total number of features/columns.
        
     n_aspects : int, optional (default=20)
       The number of aspects samples/features belong to.
       
    
     aspect_range : int, optional (default=-800)
       How many elo points is the variation range for primary aspects.  
       
     sample_aspect0 : int, optional (default=-800)
       How many elo points is non-primary aspect in a sample below the average elo.
       
       
     test_aspect0 : int, optional (default=-1200)
       How many elo points is a non-primary aspect in a test below the average elo.
       
     Returns
     -----------
     X: the data table
     Y_samples: aspect strengths for samples
     Y_tests: aspect strengths for tests
     """
     Y_samples = [rnd_aspects(n_aspects, aspect_range, sample_aspect0) for _ in range(n_samples)]
     Y_tests = [rnd_aspects(n_aspects, aspect_range, test_aspect0) for _ in range(n_tests)]
     f = vectorize(lambda i, j: int(aspect_pass(Y_samples[i], Y_tests[j])))
     X = fromfunction(f, shape=(n_samples, n_tests), dtype=int)
     return X, Y_samples, Y_tests
     
     
     
def rnd_aspects(n_aspects, aspect_range, aspect0, avg=1500):
    result =  [avg+aspect0] * n_aspects
    primary_i = randrange(n_aspects)
    result[primary_i] = randrange(avg-aspect_range, avg+aspect_range)
    return result
     


def elo_prob(elo1, elo2):
    q1 = 10** (elo1 / 400)
    q2 = 10** (elo2 / 400)
    return q1 / (q1 + q2)

def rnd_pass(sample_elo, test_elo): 
    return elo_prob(sample_elo, test_elo) > random()
    

def aspect_pass(sample_aspects, test_aspects):
    return all([rnd_pass(s,t) for s,t in zip(sample_aspects, test_aspects)])





class MockSkillTestSystem(FCASystemDF):

        def __init__(self, X, Y_samples, Y_tests, use_samples=True):
            self.data = DataFrame(X)
            self.Y_samples = Y_samples
            self.Y_tests = Y_tests
            self.use_samples = use_samples
  
                
        def conceptchaincover(self, uncovered=0.1, max_cc=20):
            """
            Returns a set of concept chains generated from Y_samples.
            Ignores uncovered, max_cc
            """
            arr = self.datacopy()
            arr_sum = self.totalsum(arr)
            result = []
            uncovered_list = []
            if self.use_samples:
                Y = self.Y_samples
            else:
                Y = self.Y_tests
            aspect_i = range(len(Y[0])) 
            #aspect_i = range(len(self.Y_samples[0])) 
            for i in aspect_i:
                cc_seqtuple = [(max(s), n) for n, s in enumerate(Y) if s.index(max(s))==i]
                #print("sorted_inf", sorted(cc_seqtuple, reverse=True))
                cc_seq = [n for _, n in sorted(cc_seqtuple, reverse=self.use_samples)]
                if self.use_samples:
                    cc = ConceptChain(cc_seq, self)
                else:
                    cc = ConceptChain.intent_init(cc_seq, self)
                result.append(cc)
                for (e, i) in cc:
                    arr.loc[e, i] = 0
                new_sum = self.totalsum(arr)
                uncovered_list.append(new_sum/arr_sum)
            return result, uncovered_list



if __name__ == "__main__":
   # f = lambda i, j: aspect_pass(Y_samples[i], Y_tests[j])
   # fv = vectorize(f)
    X, Ys, Yt = make_test_results()
    print(X)
    print(Ys)
    print(Yt)
    ms = MockSkillTestSystem(X, Ys, Yt)
    msts = MockSkillTestSystem(X, Ys, Yt, use_samples=False)
    for s in [ms, msts]:
        print("use samples:", s.use_samples)
        ccc, uc = s.conceptchaincover()
        print(uc)
        for cc in ccc:
            print(cc)
    """
    for i in range(20):
        s = rnd_aspects(3, 400, -600)
        t = rnd_aspects(3, 400, -1200)
        print(s, t, aspect_pass(s, t))
    """
























