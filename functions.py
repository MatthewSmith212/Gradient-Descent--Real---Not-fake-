from typing import Callable
import numpy as np

"""
To quickly calculate the loss function, the program needs a numpy vector function
Which takes an np.float64 (x-value) and returns an np.float64 (y-value)
"""
NUMERICAL_DERIVATIVE_SHAPE = Callable[[np.float64], np.float64]

def applyPolynomialFunction(coeffs: np.ndarray, x: np.float64):
    #Polynomial functions are stored as nparrays with coeff[i] being the coefficient of x^i
    result = np.float64(0.0)
    for i in range(len(coeffs)):
        result += coeffs[i] * (x ** i)
    return result

def constructPolynomialDeriv(coeffs: np.ndarray) -> list[NUMERICAL_DERIVATIVE_SHAPE]:
    """
    Partial derivative of sum over k of wkx^k
    with respect to wi
    is just x^i.
    This reutrns a list such that
    list[i] is the partial derivative with respect to wi.
    Because the derivative does not change
    When wi changes.
    Because it's just x^i.
    """
    derivs = list()
    for i in range(len(coeffs)):
        def deriv(x: np.float64) -> np.float64:
            return x ** i
        derivs.append(np.vectorize(deriv))
    return derivs

def applySinusoidalFunction(coeffs: np.ndarray, x: np.float64):
    """
    Sinusoidal functions are stored as 2d np arrays 
    with each term containing 4 floats: [a, b, c, d]
    representing asin(b(x-c)) + d
    """
    result = np.float64(0.0)
    for coeff in coeffs:
        a, b, c, d = coeff[0], coeff[1], coeff[2], coeff[3]
        result += a * np.sin(b * (x - c)) + d
    return result

def constructSinusoidalDeriv(coeffs: np.ndarray) -> list[NUMERICAL_DERIVATIVE_SHAPE]:
    """
    2D array but each parameter is a weight.
    So we need a partial gradient for each.
    Since each parameter will have a different partial derivative, we'll iterate over a row for each call of this constructor function
    To keep the mathematical working in the math functions.
    Return is a 1d list of the partial derivatives with respect to each parameter
    where derivs[4k], derivs[4k + 1], derivs[4k + 2], derivs[4k + 3]
    are the derivatives with respect to each parameter in coeffs[i].
    This does need to be recalculated every time
    """
    derivs = list()
    for i in range(len(coeffs)):
        #iterate over all the sinusoidal terms
        for j in range(len(coeffs[0])):
            def deriv()


def applyExponentialFunction(coeffs: np.ndarray, x: np.float64):
    """
    Exponential functions are stored as 2d np array
    with each term containing 4 floats: [a, b, c, d]
    representing ae^(b(x-c))+d
    """
    result = np.float64(0.0)
    for coeff in coeffs:
        a, b, c, d = coeff[0], coeff[1], coeff[2], coeff[3]
        results += a * np.e ** (b(x-c)) + d
    return result
