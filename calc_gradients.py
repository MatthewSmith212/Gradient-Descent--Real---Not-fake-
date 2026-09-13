import numpy as np
import math

def calcGradients(lossFunc: function, params: np.ndarray, data: np.ndarray, domain: np.ndarray, h: np.float64) -> np.ndarray:
    #params is a list of terms each of the form [coeff, exp] (the exponents should just be in order from 0 to i)
    gradients = np.zeros(len(params), dtype=np.float64)
    #1e-5 is really close to 0, so this will very closely approximate the derivative at at a point
    #Different values of h produce concerningly different results. Not ideal but maybe manageable

    for i in range(len(params)):
        #since this is a partial derivative, we are treating everything else as constant, so it's just the same array
        paramsPlusH = np.array(params)
        paramsPlusH[i] += h
        loss1 = lossFunc(paramsPlusH, data, domain)
        loss2 = lossFunc(params, data, domain)
        numerator = loss1 - loss2
        denominator = h
        partialDeriv = numerator/denominator
        gradients[i] = partialDeriv
    
    return gradients

#this felt way too easy.
#Wrote that a while ago and I'm still troubleshooting