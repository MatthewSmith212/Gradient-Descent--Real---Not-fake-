#I should learn how to actually use pytest
from functions import FUNCTIONS
import numpy as np

if __name__ == "__main__":
    polynomialCoeffs = np.array([[5, 0], [3, 1], [2, 2]])
    PolynomialFunction = FUNCTIONS["Polynomial"]

    points = np.array([0, 1, 2, 4, 6])
    value = PolynomialFunction.applyFunction(polynomialCoeffs, points)
    print(value)

    invarianceArray = PolynomialFunction.constructInvarianceArray(polynomialCoeffs)
    