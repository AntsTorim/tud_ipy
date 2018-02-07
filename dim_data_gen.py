from random import random, gauss
import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA, FactorAnalysis


class Dimension:

    def __init__(self, test_n, subject_n, test_dist=random, subject_dist=random):
        """
        Zero argument functions test_dist and subject_dist generate test and subject strengths.
        """
        self.test_n = test_n
        self.subject_n = subject_n
        self.test_dist = test_dist
        self.subject_dist = subject_dist
        self.tests = [test_dist() for _ in range(test_n)]
        self.subjects = [subject_dist() for _ in range(subject_n)]



class ContextFactory:
    """
    Abstract base class
    """
    
    def noisy(self, value, noise):
        """
        Helper function that returns the value with added random noise
        """
        return max(0.0, min(1.0, gauss(value, value*noise)))
    
    def make_context(self, noise=0.0):
        """
        Returns a context (binary array) applying random noise 
        (recommended 0 to 1).
        """    
        raise NotImplementedError



class DimensionalContextFactory(ContextFactory):

    
    def __init__(self, *dimensions):
        """
        Sets up a factory out of the Dimensions that should have equal subject_n-s
        """
        assert len(dimensions) > 0
        self.total_test_n = sum([d.test_n for d in dimensions])
        subject_ns = [d.subject_n for d in dimensions]
        assert all([x==subject_ns[0] for x in subject_ns]), "Subject nrs vary across dimensions"
        self.subject_n = subject_ns[0]
        self.dimensions = dimensions
   
            
    def make_context(self, noise=0.0):
        """
        Returns a context (binary array) applying random noise 
        (recommended 0 to 1).
        """      
        a = np.zeros((self.subject_n, self.total_test_n), dtype=int)
        t_i = 0
        for d in self.dimensions:
            for t in d.tests:
                s_i = 0
                for s in d.subjects:
                    if self.noisy(s, noise) > t: a[s_i, t_i] = 1                        
                    s_i += 1
                t_i += 1
        return a        
 
        
    

class GeneralDimensionalContextFactory(ContextFactory):
    """
    Both objects and attributes can be a mix of dimensions. Probabilities are generated
    by multiplying row and column dimension-vector matrices.
    """
    
        
       
    def __init__(self, Gd, Md):
        """
        Gd and Md are |G|*n and n*|M| numpy arrays where rows and columns
        correspond to n dimensional vectors for these objects and attributes.
        We assume that values of Gd and Md are between 0...1.
        """
        assert Gd.shape[1] == Md.shape[0], "Gd and Md must be |G|*n and n*|M| matrices"
        self.Gd = Gd
        self.Md = Md
        self.Kp = np.minimum(Gd @ Md, 1) # Initializes the probability context Kp
        
    @staticmethod
    def quick_make(row_n, col_n, dimensions = (0.9, 0.1), variation=0.5):
        Gd = np.array([dimensions for _ in range(row_n)])
        Md = np.array([dimensions for _ in range(col_n)])
        all_noisy = np.vectorize(lambda x: ContextFactory.noisy(None, x, noise=variation))
        Gd = all_noisy(Gd)
        Md = np.transpose(all_noisy(Md))
        return GeneralDimensionalContextFactory(Gd, Md)
     
        
    
    def make_context(self, noise=0.0):
        binarize = np.vectorize(lambda x: int(self.noisy(x, noise) > 0.5))
        return binarize(self.Kp)
        
 

if __name__ == "__main__":
    Gd = np.array([[1, 0.5], [0.7, 0.7], [0.5, 0.3], [0.9, 0.0], [0.4, 0.3]])
    Md = np.array([[1, 0.0, 0.4, 0.0], [0.2, 0.7, 0.3, 1.0]])
    #gdcf = GeneralDimensionalContextFactory(Gd, Md)
    gdcf = GeneralDimensionalContextFactory.quick_make(row_n=10, col_n=5, dimensions = (0.55, 0.45))
    print(gdcf.Kp)
    print(gdcf.Gd)
    print(gdcf.Md)
    K = gdcf.make_context(noise=0.0)
    print(K)
    U, s, V = linalg.svd(K)
    print(s)
    pca = PCA(n_components='mle')
    pca.fit(K)
    print(pca.n_components_)
    """
    x = Dimension(4, 10)
    y = Dimension(3, 10)
    print(x.tests)
    print(x.subjects)
    cf = DimensionalContextFactory(x, y)
    print(cf.total_test_n)
    print(cf.dimensions)
    c_old = None
    for _ in range(3):
        c_new = cf.make_context(noise=1.0)
        print(c_new==c_old)
        print(c_new)
        c_old = c_new
    """ 
