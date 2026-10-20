#I should learn how to actually use pytest
from functions import constructSinusoidalDerivBuilders
import numpy as np

if __name__ == "__main__":
    sinusoidalCoeffs = np.array([[5, 3, 2, 1]])
    partials = constructSinusoidalDerivBuilders(sinusoidalCoeffs)
    print(partials)
    print(sinusoidalCoeffs[0][0])
    for i in range(4):
        print(partials[0][i](0, sinusoidalCoeffs[0]))