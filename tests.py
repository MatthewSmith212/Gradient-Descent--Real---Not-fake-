#I should learn how to actually use pytest
from functions import constructPolynomialDerivBuilders
import numpy as np

if __name__ == "__main__":
    polynomialCoeffs = np.array([[5, 3]])
    partials = constructPolynomialDerivBuilders(polynomialCoeffs)
    print(partials)
    print(polynomialCoeffs[0][0])
    for i in range(2):
        print(partials[i]()(1))