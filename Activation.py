# -*- coding: utf-8 -*-
"""

Activation Functions for the ANN.

"""

import numpy as np

class Activation:
    
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative
        
    def __call__(self, x):
        return self.function(x)
    
def sigma(x):
    return 1/(1+np.exp(-x))

def sigmaprime(x):
    return sigma(x)*(1-sigma(x))
    
sigmoid = Activation(sigma, sigmaprime)

def ReLU(x):
    return x * (x>0) + 1e-3 * (x*(x<0))

def ReLUprime(x):
    return 1. * (x>0) - 1e-3 * (1.*(x<0))

relu = Activation(ReLU, ReLUprime)
