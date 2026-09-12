import numpy as np
import math

"""
Deprecated - this didn't end up being useful
"""

def applyFunction(params, input):
    result = float
    for term in params:
        coeff, expt = term[0], term[1]
        