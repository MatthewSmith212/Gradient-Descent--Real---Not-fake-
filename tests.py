#I should learn how to actually use pytest
from functions import constructDeriv, applyFunction
import numpy as np

if __name__ == "__main__":
    polynomialCoeffs = np.array([[5, 0], [3, 1], [2, 2]])
    value = applyFunction(polynomialCoeffs, "Polynomial", np.array([0, 1, 2, 3]))
    print(value)