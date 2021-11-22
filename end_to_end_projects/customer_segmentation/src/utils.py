import numpy as np
import random 

def set_seeds(seed=44):
    """ 
    reproducibility

    numpy.random 
    """
    np.random.seed(seed)
    random.seed(seed)
