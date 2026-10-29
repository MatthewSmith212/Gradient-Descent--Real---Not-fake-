#I should learn how to actually use pytest
from functions import constructDeriv, applyFunction
import numpy as np

if __name__ == "__main__":
    polynomialCoeffs = np.array([[5, 0], [3, 1], [2, 2]])
    #value = applyFunction(polynomialCoeffs, "Polynomial", np.array([0, 1, 2, 3]))
    #print(value) 
    polynomialDerivs = np.empty_like(polynomialCoeffs, dtype=object)
    for i in range(len(polynomialCoeffs)):
        coeffs = polynomialCoeffs[i]
        for j in range(len(coeffs)-1):
            polynomialDerivs[i][j] = constructDeriv(
                termCoeffs=coeffs,
                term=j,
                type="Polynomial"
            )
    
    