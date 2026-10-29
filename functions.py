from typing import Callable
import numpy as np

"""
To quickly calculate the loss function, the program needs a numpy vector function
Which takes an np.float64 (x-value) and returns an np.float64 (y-value)
"""
NUMERICAL_DERIVATIVE_SHAPE = Callable[[np.ndarray], np.ndarray]

"""
Symbolic derivatives never change, just the coefficients 
so store the symbolics for quick access
"""

DERIV_TUPLES = dict()
BASE_FUNCS = dict()

DERIV_TUPLES["Polynomial"] = (
    lambda x, i:x**i,
    None, #i should not change
)

DERIV_TUPLES["Sinusoidal"] = (
    lambda x, a, b, c, d: np.sin(b * (x - c)), #partial derivative with respect to a is just f(x)/a
    lambda x, a, b, c, d: a * (np.cos(b * (x-c)) * (x-c)), #d/db
    lambda x, a, b, c, d: a * (np.cos(b * (x-c)) * (-b)), #d/dc
    lambda x, a, b, c, d: np.ones_like(x, dtype=np.float64), #d/dd
)

DERIV_TUPLES["Exponential"] = (
    lambda x, a, b, c: np.exp(b*x),
    lambda x, a, b, c: a * (np.exp(b * x) * x),
    lambda x, a, b, c: np.ones_like(x, dtype=np.float64),
)

DERIV_TUPLES["Logarithmic"] = (
    lambda x, a, b, c: np.log(x-b),
    lambda x, a, b, c: a * (np.log(x-b) * -1),
    lambda x, a, b, c: np.ones_like(x, dtype=np.float64),
)

BASE_FUNCS["Polynomial"] = (
    lambda x, a, i: a * (x ** i)
)

BASE_FUNCS["Sinusoidal"] = (
    lambda x, a, b, c, d: a * np.sin(b * (x - c)) + d
)

BASE_FUNCS["Exponential"] = (
    lambda x, a, b, c: a * np.exp(b*x) + c
)

BASE_FUNCS["Logarithmic"] = (
    lambda x, a, b, c: a * np.log(x-b) + c
)

def constructDeriv(termCoeffs: np.ndarray, term:int, type: str) -> NUMERICAL_DERIVATIVE_SHAPE:
    """
    

    Returns 
    """
    partial = DERIV_TUPLES[type][term]
    return lambda x: partial(x, *termCoeffs)

def constructDerivs(coeffs: np.ndarray, type:str) -> np.ndarray:
    """
    Cleaner to do all the array work here

    Need the partial derivative with respect to each parameter but, 
    in a given term, the gradient depends on the other weights
    so it needs to be recalculated in each pass.
    """
    derivs = np.empty_like(coeffs, dtype=object)
    for i in range(len(coeffs)):
        termCoeffs = coeffs[i]
        for j in range(len(termCoeffs)):
            partial = DERIV_TUPLES[type][j]
            #A function which will calculate the partial derivative with respect to a parameter at an x-value
            derivs[i][j] = lambda x: partial(x, *termCoeffs)

    return derivs

def applyFunction(coeffs: np.ndarray, type: str, x: np.ndarray) -> np.ndarray:
    """
    Apply a function (described by its coefficients) to an array of values x.
    This broadcasts across the x-values but still iterates through the coefficients
    using a slow python for-loop. 
    Gotta git gud at numpy to fix that. I think there's 3d matrices involved.


    For consistency, polynomial functions need to have indices as parameters.
    I will, however, freeze these because the optimization is more sensible
    if only the coefficients move.
    """
    termFunc = BASE_FUNCS[type]
    result = np.zeros_like(x)
    for coeff in coeffs:
        #not as fast as it could be, because the for loop runs in python
        #However, each term gets broacast over the array (I think).
        result += termFunc(x, *coeff) #pass the i for compatability with polynomial functions
    return result

#region functions

def applyPolynomialFunction(coeffs: np.ndarray, x: np.ndarray):
    #Polynomial functions are stored as nparrays with coeff[i] being the coefficient of x^i
    result = np.float64(0.0)
    for i in range(len(coeffs)):
        result += coeffs[i] * (x ** i)
    return result


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


def applyExponentialFunction(coeffs: np.ndarray, x: np.float64) -> np.float64:
    """
    Exponential functions are stored as 2d np array
    with each term containing 3 floats: [a, b, c]
    representing ae^(bx)+c.
    Really, there should just be one constant term but it would be more work to represent.

    I originally had a horizantal shift factor but it can be represented in the vertical stretch
    """
    result = np.float64(0.0)
    for coeff in coeffs:
        a, b, c = coeff[0], coeff[1], coeff[2]
        result += a * np.exp(b*x) + c
    return result



def applyLogarithmicFunction(coeffs: np.ndarray, x: np.float64) -> np.float64:
    """
    Exponential functions are stored as 2d np array
    with each term containing 3 floats: [a, b, c]
    representing aln(x-b)+c.
    Really, there should just be one constant term but it would be more work to represent.
    """
    result = np.float64(0.0)
    for coeff in coeffs:
        a, b, c = coeff[0], coeff[1], coeff[2]
        result += a * np.log(x-b) + c
    return result


#region deprecated

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
            return np.vectorize(DERIV_TUPLES["Polynomial"][0])
        derivBuilders[i] = derivBuilder
        del derivBuilder
    return derivBuilders

def constructSinusoidalDerivBuilders() -> np.ndarray:
    """
    coeff at i has 4 floats: [a, b, c, d]
    representing asin(b(x-c))+d.

    Need the partial derivative with respect to each parameter but, 
    in a given term, the gradient depends on the other weights
    so it needs to be recalculated in each pass

    Returns a 1d Array of shape (4,) to calculate a term's gradients
    with each term being a function which takes the coefficients as its only argument
    and returns a fast vector function to calculate the partial deriv for the weight's gradient
    to run before each recalculation of the gradients
    """
    derivBuilders = np.empty(shape=(4), dtype=object)
        #iterate over all the sinusoidal terms, and make a list of partial derivatives with respect to each parameter of each term
       
    def derivBuilder1(termCoeffs : np.ndarray) ->  Callable[[np.float64], np.float64]: # pyright: ignore[reportRedeclaration]
        #partial derivative with respect to a is just f(x)/a
        #vector functions cant read out of arrays so lets define the function with them. They only change once per pass, so less depends on the efficiency with which we access these.
        a, b, c, d = termCoeffs[0], termCoeffs[1], termCoeffs[2], termCoeffs[3]
        def deriv(x: np.float64):
            return np.sin(b * (x - c))
        #Return a fast vector function
        return np.vectorize(deriv)
    derivBuilders[0] = derivBuilder1

    def derivBuilder2(termCoeffs: np.ndarray) -> Callable[[np.float64], np.float64]: # pyright: ignore[reportRedeclaration]
        a, b, c, d = termCoeffs[0], termCoeffs[1], termCoeffs[2], termCoeffs[3]
        def deriv(x: np.float64):
            #partial derivative with respect to b
            return a * (np.cos(b * (x-c)) * (x-c))
        #Return a fast vector function
        return np.vectorize(deriv)
    derivBuilders[1] = derivBuilder2

    def derivBuilder3(termCoeffs: np.ndarray) -> Callable[[np.float64], np.float64]: # pyright: ignore[reportRedeclaration]
        a, b, c, d = termCoeffs[0], termCoeffs[1], termCoeffs[2], termCoeffs[3]
        def deriv(x: np.float64):
            #Partial derivative with respect to c (most difficult)
            return a * (np.cos(b * (x-c)) * (-b))
        #Return a fast vector function
        return np.vectorize(deriv)
    derivBuilders[2] = derivBuilder3

    def derivBuilder4(termCoeffs: np.ndarray) -> Callable[[np.float64], np.float64]:
        #Really just for consistency of formatting
        def deriv(x: np.float64):
            #partial derivative with respect to d is just 1
            return np.float64(1)
        return np.vectorize(deriv)
    derivBuilders[3] = derivBuilder4
        
    return derivBuilders

def constructExponentialTermDerivBuilders() -> np.ndarray:
    """
    coeff at i has 3 floats: [a, b, c]
    representing ae^(bx)+c.

    Need the partial derivative with respect to each parameter but, 
    in a given term, the gradient depends on the other weights
    so it needs to be recalculated in each pass

    Returns a 1d Array of shape (3,) to calculate a term's gradients
    with each term being a function which takes the coefficients as its only argument
    and returns a fast vector function to calculate the partial deriv for the weight's gradient
    to run before each recalculation of the gradients
    """
    derivBuilders = np.empty(shape=(3), dtype=object)
    #iterate over all the exponential terms, and make a list of partial derivatives with respect to each parameter of each term
    def derivBuilder1(termCoeffs : np.ndarray) ->  Callable[[np.float64], np.float64]: # pyright: ignore[reportRedeclaration]
        #partial derivative with respect to a is just f(x)/a
        #vector functions cant read out of arrays so lets define the function with them. They only change once per pass, so less depends on the efficiency with which we access these.
        a, b, c = termCoeffs[0], termCoeffs[1], termCoeffs[2]
        def deriv(x: np.float64):
            return np.exp(b*x)
        #Return a fast vector function
        return np.vectorize(deriv)
    derivBuilders[0] = derivBuilder1
    
    def derivBuilder2(termCoeffs: np.ndarray) -> Callable[[np.float64], np.float64]: # pyright: ignore[reportRedeclaration]
        a, b, c = termCoeffs[0], termCoeffs[1], termCoeffs[2]
        def deriv(x: np.float64):
            #partial derivative with respect to b, by chain rule
            return a * (np.exp(b * x) * x)
        #Return a fast vector function
        return np.vectorize(deriv)
    derivBuilders[1] = derivBuilder2

    def derivBuilder3(termCoeffs: np.ndarray) -> Callable[[np.float64], np.float64]:
        #Really just for consistency of formatting
        def deriv(x: np.float64):
            #partial derivative with respect to c is just 1
            return np.float64(1)
        return np.vectorize(deriv)
    derivBuilders[2] = derivBuilder3

    return derivBuilders

def constructLogarithmicDerivBuilders():
    return
