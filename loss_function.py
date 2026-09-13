import numpy as np
from numpy.polynomial import Polynomial
import math

def individualLoss(polyn : Polynomial, point: np.ndarray, domain: np.ndarray):
    """
    MinArg, the really hard thing to find, is the value of x which gives the point on the function closest to the point we're checking.
    Then the loss is simply the distance from (minArg, f(minArg)) to the point

    ~~Basic flow: create distance function -> find zeroes of distance function -> find global minimum~~

    THE DOMAIN IS LIMITED, JUST USE A BINARY SEARCH. Wait NVM
    """

    x1, y1 = point[0], point[1]
    distSquare = (y1-polyn) ** 2 #This is the y-axis component
    distSquare += (Polynomial(coef=(x1, -1), domain=domain) ** 2) #this is the x-axis component 

    distSquareDiff = distSquare.deriv()

    roots = distSquareDiff.roots()
    real_mask = np.abs(roots.imag) < 1e-6
    roots = roots[real_mask].real

    minDist = distSquare(roots[0])
    
    for i in range(1, len(roots)):
        minDist = min(minDist, distSquare(roots[i]))

    if -1e-5 < minDist < 0:
        minDist = 0
    
    return minDist #I've heard that the squared error is actually a better loss function


def lossFunction(params: np.ndarray, data: np.ndarray, domain: np.ndarray):
    #data should be a numpy array of ordered pairs, each ordered pair also being a numpy array
    #params is an array with params[i] being the coefficient to x^i
    polyn = Polynomial(coef=params, domain=domain)
    #fastIndividualLoss = np.frompyfunc(individualLoss, 2, 1) This is my white whale

    totalLoss = np.float64(0)

    #gotta get the ufuncs sorted
    for point in data:
        totalLoss += individualLoss(polyn, point, domain)
    return totalLoss / len(data) #average loss should (in theory) behave the same but it's way easier to read
    
    