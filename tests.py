#I should learn how to actually use pytest
from functions import FUNCTIONS
import numpy as np

if __name__ == "__main__":
    polynomialCoeffs = np.array([[5, 0], [3, 1], [2, 2]])
    PolynomialFunction = FUNCTIONS["Polynomial"]

    points = np.array([0, 1, 2, 4, 6])
    values = PolynomialFunction.applyFunction(polynomialCoeffs, points)
    print(values)

    invarianceArray = PolynomialFunction.constructInvarianceArray(polynomialCoeffs)

    #Array with the lambdae 
    derivFunctions = PolynomialFunction.constructDerivs(polynomialCoeffs, invarianceArray)

    #Make a parallel array to coeffs where each coeffecient gets replaced with an array 
    #of the partial derivatives of the activation function with respect to each point in points
    coeffsShape = (len(polynomialCoeffs), PolynomialFunction.getNumParameters(),)
    activationFunctionDerivs = np.zeros(
        shape=(coeffsShape[0], coeffsShape[1], len(points),), 
        dtype=np.float64
    )
    
    #Gotta go in a python loop :(
    for i in range(coeffsShape[0]):
        for j in range(coeffsShape[1]):
            activationFunctionDerivs[i, j] = derivFunctions[i, j](points)

    print(activationFunctionDerivs)


