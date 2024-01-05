import numpy as np

def representation_shift(weights_1, weights_2) -> float:
    """
    Computes l_2 norm between representations in domain 't' and representations in domain 't+1'
    """
    return np.linalg.norm(weights_1, weights_2, 2)


def sharpness() -> float:
    """
    Computes sharpness (TODO: find library with existing computation)
    """
    pass

def num_active_relu_units() -> int:
    """
    
    """
    pass

def predictor_rank() -> int:
    """
    Computes the minimum number of principal components such that 99% of the variance is accounted for.
    """
    pass
