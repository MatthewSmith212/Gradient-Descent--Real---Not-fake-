from data_and_function.retrieve_data import readData, displayData
from loss_function import lossFunction
from calc_gradients import calcGradients
import numpy as np

"""
Based on cheating, using the actual function, the minimum achievable loss is 10790.763428
"""

def descendGradient(lossFunc: function, params: np.ndarray, data: np.ndarray, domain: np.ndarray, h: np.float64, passes: int, learningRate: float):
    for _ in range(passes):
        grads = calcGradients(lossFunc=lossFunc, params=params, data=data, domain=domain,h=h)
        for i in range(len(grads)):
            params[i] -= grads[i] * learningRate
    return params

if __name__ == "__main__":
    #showSavedData("data_and_function/testScores.pkl")
    data = np.array(readData("data_and_function/testScores.pkl"), dtype=np.float64)
    fx = np.array([20, -4, 6], dtype=np.float64) #fx[i] is the coefficient to the term x^i

    domain = np.array([min(data[:, 0]), max(data[:, 0])], dtype=np.float64)

    print("initial loss: " + str(lossFunction(fx, data, domain)))

    
    diffPrecision = 8
    passes = 100
    learningRate = 0.1

    differential = np.float64(1) ** (-diffPrecision)
    
    fx = descendGradient(lossFunc=lossFunction, params=fx, data=data, domain=domain, h=differential, passes=passes, learningRate=learningRate)
    print(fx)
    print("final loss: " + str(lossFunction(fx, data, domain)))