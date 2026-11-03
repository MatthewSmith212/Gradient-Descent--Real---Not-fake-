from typing import Callable
import numpy as np

"""
To quickly calculate the loss function, the program needs a numpy vector function
Which takes an np.float64 (x-value) and returns an np.float64 (y-value)
"""
NUMERICAL_DERIVATIVE_SHAPE = Callable[[np.ndarray], np.ndarray]
RETURN_ZERO = lambda x:np.zeros_like(x, dtype=np.float64)

"""
Symbolic derivatives never change, just the coefficients 
so store the symbolics for quick access
"""

class FunctionData:
    """
    The dictionaries were getting disorganized
    So here's a class to handle the functions
    """
    def __init__(self, funcType, funcNumParameters, baseFunc, coeffDerivs):
        #number of parameters per term
        self.funcNumParameters = funcNumParameters 

        #Calling function for a term
        self.baseFunc = baseFunc

        #Partial derivatives with respect to each term
        self.coeffDerivs = np.array(coeffDerivs) #more useful data type

        #Name of function's type. Convenient to have this accesible from just the object
        self.funcType = funcType

    def getNumParameters(self) -> int:
        #setters and getters
        return self.funcNumParameters

    def getFuncType(self) -> str:
        #getter, for security
        return self.funcType

    def constructDeriv(self, termCoeffs:np.ndarray, term:int) -> NUMERICAL_DERIVATIVE_SHAPE:
        """
        A function to calculate the partial derivative with respect to a parameter at an x-value
        """
        partial = self.coeffDerivs[term]
        if partial == None:
            #If its invariant then we just say its derivative is 0, so its gradient'll be 0
            #Could just do that in the function definitions but
            #this gives us the flexibility to freeze weights in the future
            return RETURN_ZERO
        else:
            return lambda x: partial(x, *termCoeffs)
    
    def constructDerivs(self, coeffs: np.ndarray, invarianceArray: np.ndarray) -> np.ndarray:
        """
        Cleaner to do all the array work in this file.
    
        Need the partial derivative with respect to each parameter but, 
        in a given term, the gradient depends on the other weights
        so it needs to be recalculated in each pass.
        """
        derivs = np.empty_like(coeffs, dtype=object)
        for i in range(len(coeffs)):
            termCoeffs = coeffs[i]
            for j in range(self.funcNumParameters):
                derivs[i][j] = self.constructDeriv(termCoeffs, j)
        
        return derivs
    
    def applyFunction(self, coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Apply a function (described by its coefficients) to an array of values x.
        This broadcasts across the x-values but still iterates through the coefficients
        using a slow python for-loop. 
        Gotta git gud at numpy to fix that. I think there's 3d matrices involved.
    
        For consistency, polynomial functions need to have indices as parameters.
        I will, however, freeze these because the optimization is more sensible
        if only the coefficients move.
        """
        result = np.zeros_like(x, dtype=np.float64)
        for coeff in coeffs:
            #not as fast as it could be, because the for loop runs in python
            #However, each term gets broacast over the array (I think).
            result += self.baseFunc(x, *coeff) #pass the i for compatability with polynomial functions
        return result

    def constructInvarianceArray(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Parameters with "None" for their partial derivative function 
        should not be adjusted during gradient descent
        
        This returns an array reflecting the invariant parameters 
        """
        
        invariants = np.array([self.coeffDerivs]) == None

        #calculate the number of variant parameters to expect
        self.numVariantParameters = self.funcNumParameters - np.sum(invariants)
        
        return np.repeat(invariants, len(coeffs), 0)


#Still useful to have the functions keyed
FUNCTIONS: dict[str, FunctionData] = dict()

FUNCTIONS["Polynomial"] = FunctionData(
    funcType="Polynomial",
    funcNumParameters=2,
    baseFunc= lambda x, a, i: a * (x ** i),
    coeffDerivs=(
        lambda x, a, i:x**i,
        None, #i should not change
    )
)

FUNCTIONS["Sinusoidal"] = FunctionData(
    funcType="Sinusoidal",
    funcNumParameters= 4,
    baseFunc= lambda x, a, b, c, d: a * np.sin(b * (x - c)) + d,
    coeffDerivs=(
        lambda x, a, b, c, d: np.sin(b * (x - c)), #partial derivative with respect to a is just f(x)/a
        lambda x, a, b, c, d: a * (np.cos(b * (x-c)) * (x-c)), #d/db
        lambda x, a, b, c, d: a * (np.cos(b * (x-c)) * (-b)), #d/dc
        lambda x, a, b, c, d: np.ones_like(x, dtype=np.float64), #d/dd
    )
)

FUNCTIONS["Exponential"] = FunctionData(
    funcType="Exponential",
    funcNumParameters= 3,
    baseFunc= lambda x, a, b, c: a * np.exp(b*x) + c,
    coeffDerivs=(
        lambda x, a, b, c: np.exp(b*x),
        lambda x, a, b, c: a * (np.exp(b * x) * x),
        lambda x, a, b, c: np.ones_like(x, dtype=np.float64),
    )
)

FUNCTIONS["Logarithmic"] = FunctionData(
    funcType="Logarithmic",
    funcNumParameters= 3,
    baseFunc= lambda x, a, b, c: a * np.log(x-b) + c,
    coeffDerivs=(
        lambda x, a, b, c: np.log(x-b),
        lambda x, a, b, c: a * (np.log(x-b) * -1),
        lambda x, a, b, c: np.ones_like(x, dtype=np.float64),
    )
)
