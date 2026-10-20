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

def constructPolynomialDeriv(coeffs: np.ndarray) -> np.ndarray:
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
    derivs = np.ndarray(shape=len(coeffs), dtype=object)
    for i in range(len(coeffs)):
        def deriv(x: np.float64) -> np.float64:
            return x ** i
        derivs[i] = deriv
        del deriv
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

def constructSinusoidalDeriv(coeffs: np.ndarray) -> np.ndarray:
    """
    2D array but each parameter is a weight.
    So we need a partial gradient for each.
    Since each parameter will have a different partial derivative, we'll iterate over a row for each call of this constructor function
    To keep the mathematical working in the math functions.
    Return is a 2d list of the partial derivatives in each term
    where derivs[4k], derivs[4k + 1], derivs[4k + 2], derivs[4k + 3]
    are the derivatives with respect to each parameter in coeffs[i].
    This does need to be recalculated every time
    """
    derivs = np.empty(shape=(len(coeffs), 4), dtype=object)
    for i in range(len(coeffs)):
        #iterate over all the sinusoidal terms, and make a list of partial derivatives with respect to each parameter of each term
       
        def deriv(x : np.float64, termCoeffs : np.ndarray) ->  np.float64: # type: ignore
            #partial derivative with respect to a is just f(x)/a
            a, b, c, d = termCoeffs[0], termCoeffs[1], termCoeffs[2], termCoeffs[3]
            return np.sin(b * (x - c))
        derivs[i][0] = deriv
        del deriv

        def deriv(x: np.float64, termCoeffs: np.ndarray) -> np.float64: # pyright: ignore[reportRedeclaration]
            #Partial derivative with respect to b
            a, b, c, d = termCoeffs[0], termCoeffs[1], termCoeffs[2], termCoeffs[3]
            return a * (np.cos(b * (x-c)) * (x-c))
        derivs[i][1] = deriv
        del deriv

        def deriv(x: np.float64, termCoeffs: np.ndarray): # pyright: ignore[reportRedeclaration]
            #Partial derivative with respect to c (most difficult)
            a, b, c, d = termCoeffs[0], termCoeffs[1], termCoeffs[2], termCoeffs[3]
            return a * (np.cos(b * (x-c)) * (-b))
        derivs[i][2] = deriv
        del deriv

        def deriv(x: np.float64, termCoeffs: np.ndarray) -> np.float64:
            return np.float64(1)
        derivs[i][3] = deriv
        del deriv
        
    return derivs



def applyExponentialFunction(coeffs: np.ndarray, x: np.float64):
    """
    Exponential functions are stored as 2d np array
    with each term containing 4 floats: [a, b, c, d]
    representing ae^(b(x-c))+d
    """
    result = np.float64(0.0)
    for coeff in coeffs:
        a, b, c, d = coeff[0], coeff[1], coeff[2], coeff[3]
        result += a * np.e ** (b(x-c)) + d
    return result
