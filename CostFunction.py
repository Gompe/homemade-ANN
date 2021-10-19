# -*- coding: utf-8 -*-
"""

Cost Functions for the ANN.

"""

import numpy as np

class CostFunction:
    
    def __init__(self, function, gradient):
        self.function = function
        self.gradient = gradient
        
    def __call__(self, x, y):
        return self.function(x, y)

def sqrError(x1, x2):
    errors = (x1-x2)[0]
    return np.dot(errors, errors)

def nablaSqrError(x1, x2):
    errors = x1-x2
    return 2*errors

SE = CostFunction(sqrError, nablaSqrError)

def crossEntropy(x1, x2):
    errors = -np.log(x1**x2) - np.log((1-x1)**(1 - x2))
    return np.sum(errors)
    
def nablaCrossEntropy(x1, x2):
    errors = (x1 - x2)/(x1*(1-x1))
    return errors

CE = CostFunction(crossEntropy, nablaCrossEntropy)