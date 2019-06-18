# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 19:15:46 2018

@author: ator-
"""
import numpy as np
import numpy.random as rnd
import sklearn.base as sk

class FreqSeriation(sk.TransformerMixin):
    """
    Seriate by frequencies, rows only
    """
    
    def fit(self, X):
        self.freq_= X @ np.ones(X.shape[1])
        self.sortindices_ = np.argsort(self.freq_)[::-1]
        return self
        
    def transform(self, X):
        return X[self.sortindices_]
        

class Freq2DSeriation(sk.TransformerMixin):
    """
    Seriate by frequencies, rows and columns
    """
    
    def fit(self, X):
        self.row_freq = X @ np.ones(X.shape[1])
        self.row_i = np.argsort(self.row_freq)[::-1]
        self.col_freq = np.ones(X.shape[0]) @ X
        self.col_i = np.argsort(self.col_freq)[::-1]
        return self
        
    def transform(self, X):
        return X[self.row_i][:, self.col_i]


class LexiSeriation(sk.TransformerMixin):
    """
    Seriate lexicographically, once
    """
    
    def fit(self, X):
        #self.sortindices_ = np.lexsort(X)
        self.sortindices_ = sorted(range(X.shape[0]), key=lambda i: tuple(X[i]), reverse=True)
        return self
        
    def transform(self, X):
        return X[self.sortindices_]
  
      
class LexiIterSeriation(sk.TransformerMixin):
    
    """
    Seriate iteratively lexicographically, by rows and columns
    until stable solution is reached. 
    """
    
    def __init__(self, refiller=None):
        self.refiller = refiller
    
    def fit(self, X):
        X_new = None
        transposed = False
        self.row_i, self.col_i  = np.arange(X.shape[0]), np.arange(X.shape[1])
        self.stepcount = 0
        while True:
            if self.refiller:
                if transposed:
                    X = self.refiller.transform(X.T, axis=0, row_i=self.row_i, col_i=self.col_i).T
                else:
                    X = self.refiller.transform(X, axis=1, row_i=self.row_i, col_i=self.col_i)
            ls = LexiSeriation()
            X_new = ls.fit_transform(X)
            if transposed:
                self.col_i = self.col_i[ls.sortindices_]
            else:
                self.row_i = self.row_i[ls.sortindices_]
            self.stepcount += 1
            if (X == X_new).all() and self.stepcount>1: 
                break
            X = X_new.transpose()
            transposed = not transposed
        return self
        
    def transform(self, X):
        return X[self.row_i][:, self.col_i]
        

class FreqLexiSeriation(sk.TransformerMixin):
    
    def __init__(self, preseriate=Freq2DSeriation, refiller=None):
        self.preseriate = preseriate
        self.refiller = refiller
    
    def fit(self, X):
        f2s = self.preseriate()
        Y = f2s.fit_transform(X)
        if self.refiller:
            self.refiller.arrange(row_i=f2s.row_i, col_i=f2s.col_i)
        lis = LexiIterSeriation(refiller=self.refiller)
        lis.fit(Y)
        self.row_i = f2s.row_i[lis.row_i]
        self.col_i = f2s.col_i[lis.col_i]
        return self
        
    def transform(self, X):
        return X[self.row_i][:, self.col_i]


    
class ConfSeriation(sk.TransformerMixin):
    """
    Seriate by conformism, rows only
    """
    
    def fit(self, X):
        """
        X must be a binary data table
        """
        self.freq_= np.add.reduce(X)
        self.weight_ = np.add.reduce(self.freq_*X, axis=1) 
        self.sortindices_ = np.argsort(self.weight_)[::-1]
        return self
        
    def transform(self, X):
        return X[self.sortindices_]


class Conf2DSeriation(sk.TransformerMixin):
    """
    Seriate by conformism
    """
    
    def fit(self, X):
        """
        X must be a binary data table
        """
        self.col_freq = np.add.reduce(X)
        self.row_conf = np.add.reduce(self.col_freq * X, axis=1) 
        self.row_i = np.argsort(self.row_conf)[::-1]
        
        self.row_freq = np.add.reduce(X, axis=1)
        self.col_conf = np.add.reduce(X * self.row_freq[:, np.newaxis], axis=0) 
        self.col_i = np.argsort(self.col_conf)[::-1]
        
        return self
        
    def transform(self, X):
        return X[self.row_i][:, self.col_i]
    

class Refiller(sk.TransformerMixin):
    """
    Refill the streaks of ones in original up to the ones in a new array
    """
    
    def fit(self, X):
        self.original = np.asarray(X)
        self.row_i = np.arange(X.shape[0])
        self.col_i = np.arange(X.shape[1])
        return self
        
    
    def arrange(self, row_i, col_i): 
        """
        Seriate original matrix by row_i, col_i
        """
        self.row_i = np.array(row_i)
        self.col_i = np.array(col_i)
        
    def transform(self, X, axis=1, row_i=None, col_i=None):
        """
        Refill along the axis seriating the possibly arranged original by row_i and col_i if not None
        """
        if row_i is None  or col_i is None:
            row_i, col_i = self.row_i, self.col_i
            
        else:
            row_i, col_i = self.row_i[row_i], self.col_i[col_i]
        X_0 = self.original[row_i][:, col_i]
        streaks = np.logical_and.accumulate(X_0, axis=axis) #A
        
        # add extra row/col of ones
        if axis == 1:
            streaks = np.c_[np.ones(streaks.shape[0], dtype='bool'), streaks[:, :-1]]
        elif axis == 0:
            new_row = np.ones(streaks.shape[1], dtype='bool')[np.newaxis,...]
            streaks = np.concatenate((new_row, streaks[:-1, :]), axis=0)

        fillpoint = streaks & X.astype(bool)
        to_fill = np.flip(np.logical_or.accumulate(np.flip(fillpoint, axis=axis), axis=axis), axis=axis)
        return np.where(to_fill, 1, X)
        

if __name__ == "__main__":
    a = np.array([[0, 1, 0, 0, 1],
                  [1, 1, 1, 0, 1],
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1]])
    r = Refiller()
    r.fit(a)
    c = np.array([[0, 1, 0, 0, 1],
                  [0, 0, 1, 0, 1],
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1]])
    lis_c = LexiIterSeriation(refiller=r).fit_transform(c)
    print(lis_c)
    """
    X = np.array([[1, 1, 0, 1, 1],
                  [0, 1, 0, 1, 1],
                  [1, 1, 0, 0, 0],
                  [0, 1, 1, 0, 0],
                  [0, 1, 0, 1, 1]])
    """
    """
    X = np.array([[1, 1, 1, 1, 0, 1],
                  [0, 1, 1, 1, 0, 0],
                  [0, 1, 1, 1, 0, 0],
                  [0, 0, 1, 1, 1, 0],
                  [0, 0, 1, 1, 1, 0]])
    """
    X = rnd.randint(2, size=(12,11))
    """
    X_new = None
    transposed = False
    row_i, col_i  = np.arange(X.shape[0]), np.arange(X.shape[1])
    stepcount = 0
    while True:
        ls = LexiSeriation()
        X_new = ls.fit_transform(X)
        if transposed:
            col_i = col_i[ls.sortindices_]
        else:
            row_i = row_i[ls.sortindices_]
        #print(X_new)
        stepcount += 1
        if (X == X_new).all() and stepcount>1: break
        X = X_new.transpose()
        transposed = not transposed
        #print(X)
        #print("____\n")
    if transposed:
        X = X.transpose()
    print(X)
    print(row_i)
    print(col_i)
    print(stepcount)
    """
    X = np.array([[0, 0, 1, 0, 1],
                  [1, 1, 0, 1, 0],
                  [0, 0, 1, 0, 1]])
    f2s = Conf2DSeriation()
    f2s.fit(X)
    print(f2s.row_i)
    print(f2s.col_i)
    Y = f2s.transform(X)
    print(Y)

    r = Refiller()
    r.fit(X)
    r.transform(X)
    """
    lis = LexiIterSeriation()
    lis.fit(Y)
    print(lis.row_i)
    print(f2s.row_i[lis.row_i])
    print(f2s.row_i)
    print(f2s.col_i[lis.col_i])
    print(lis.transform(Y))
    """
    fls = FreqLexiSeriation()
    fls.fit(X)
    print(fls.row_i)
    print(fls.col_i)
    print(fls.transform(X))
    """
    Z = np.array([[1, 1, 0, 0, 0],
                  [1, 1, 1, 1, 1],
                  [0, 0, 1, 0, 0],
                  [1, 0, 0, 1, 1],
                  [1, 0, 0, 1, 1],
                  [0, 0, 0, 0, 0]])
    print(Z.shape)
    print(np.lexsort(Z)) #lexsort is buggy, https://github.com/numpy/numpy/issues/10521
    print(Z[:,np.lexsort(Z)])
    ilist = sorted(range(Z.shape[0]), key=lambda i: tuple(Z[i]), reverse=True)
    print(ilist)
    print(Z[ilist])
    """
    """
    #print(fs.fit_transform(X))
    #fs.fit(X)
    #print(fs.sortindices_)
    #print(fs.transform(X))
    """
    print("Find LexiIterSeriation steps")
    for i in range(100, 1500, 100):
        # Smallest dim determines reps? 
        # Sufficient complexity of smallest dim determines reps?
        # Logarithm of smallest dim determines reps?
        # What is max? 10 so far.
        #rnd_table = rnd.randint(2, size=(i, 1000)) 
        rnd_table = rnd.randint(2, size=(i, i)) 
        li = LexiIterSeriation()
        fls = Freq2DSeriation()
        #fls = Conf2DSeriation()
        #li.fit(rnd_table)
        li.fit(fls.fit_transform(rnd_table))
        if li.stepcount > 0:
            print(i, li.stepcount)
    
    
        
        
        