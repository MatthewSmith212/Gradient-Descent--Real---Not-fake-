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

def constructPolynomialDerivBuilders(coeffs: np.ndarray) -> np.ndarray:
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
    derivBuilders = np.empty(shape=len(coeffs), dtype=object)
    for i in range(len(coeffs)):
        def derivBuilder() -> Callable[[np.float64], np.float64]:
            #just necessary for formatting consistency
            def deriv(x: np.float64) -> np.float64:
                return x ** i
            #Finally, vector functions
            return np.vectorize(deriv)
        derivBuilders[i] = derivBuilder
        del derivBuilder
    return derivBuilders

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

def constructSinusoidalDerivBuilders(coeffs: np.ndarray) -> np.ndarray:
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
    derivBuilders = np.empty(shape=(len(coeffs), 4), dtype=object)
    for i in range(len(coeffs)):
        #iterate over all the sinusoidal terms, and make a list of partial derivatives with respect to each parameter of each term
       
        def derivBuilder(termCoeffs : np.ndarray) ->  Callable[[np.float64], np.float64]: # pyright: ignore[reportRedeclaration]
            #partial derivative with respect to a is just f(x)/a
            #vector functions cant read out of arrays so lets define the function with them. They only change once per pass, so less depends on the efficiency with which we access these.
            a, b, c, d = termCoeffs[0], termCoeffs[1], termCoeffs[2], termCoeffs[3]
            def deriv(x: np.float64):
                return np.sin(b * (x - c))
            #Return a fast vector function
            return np.vectorize(deriv)
        derivBuilders[i][0] = derivBuilder
        del derivBuilder

        def derivBuilder(termCoeffs: np.ndarray) -> Callable[[np.float64], np.float64]: # pyright: ignore[reportRedeclaration]
            #vector functions cant read out of arrays so lets define the function with them. They only change once per pass, so less depends on the efficiency with which we access these.
            a, b, c, d = termCoeffs[0], termCoeffs[1], termCoeffs[2], termCoeffs[3]
            def deriv(x: np.float64):
                #partial derivative with respect to b
                return a * (np.cos(b * (x-c)) * (x-c))
            #Return a fast vector function
            return np.vectorize(deriv)
        derivBuilders[i][1] = derivBuilder
        del derivBuilder

        def derivBuilder(termCoeffs: np.ndarray) -> Callable[[np.float64], np.float64]: # pyright: ignore[reportRedeclaration]
            #vector functions cant read out of arrays so lets define the function with them. They only change once per pass, so less depends on the efficiency with which we access these.
            a, b, c, d = termCoeffs[0], termCoeffs[1], termCoeffs[2], termCoeffs[3]
            def deriv(x: np.float64):
                #Partial derivative with respect to c (most difficult)
                return a * (np.cos(b * (x-c)) * (-b))
            #Return a fast vector function
            return np.vectorize(deriv)
        derivBuilders[i][2] = derivBuilder
        del derivBuilder

        def derivBuilder(termCoeffs: np.ndarray) -> Callable[[np.float64], np.float64]:
            #Really just for consistency of formatting
            def deriv(x: np.float64):
                #partial derivative with respect to d is just 1
                return np.float64(1)
            return np.vectorize(deriv)
        derivBuilders[i][3] = derivBuilder
        del derivBuilder
        
    return derivBuilders



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
