import numpy as np
import random

def set_seeds(seed=44):
    """
    set seeds for reproducibility
    """
    np.random.seed(seed)
    random.seed(seed)