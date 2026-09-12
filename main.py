from data_and_function.retrieve_data import readData, displayData
from loss_function import lossFunction
from calc_gradients import calcGradients
import numpy as np
import math

"""
Based on cheating, using the actual function, the minimum achievable loss is 10790.763428
"""

def descendGradient(lossFunc: function, params: np.ndarray, data: np.ndarray, h: np.float64, passes: int, learningRate: float):
    for _ in range(passes):
        grads = calcGradients(lossFunc, params, data, h)
        for i in range(len(grads)):
            params[i][0] -= grads[i] * learningRate
    return params

if __name__ == "__main__":
    #showSavedData("data_and_function/testScores.pkl")
    data = readData("data_and_function/testScores.pkl")
    fx = np.array([[60, 0], [8, 1], [5, 2], [3, 3], [0.5, 4]])

    print("initial loss: " + str(lossFunction(fx, data)))

    
    diffPrecision = 3
    passes = 100
    learningRate = 0.0001

    differential = np.float64(1) ** (-diffPrecision)
    
    fx = descendGradient(lossFunc=lossFunction, params=fx, data=data, h=differential, passes=passes, learningRate=learningRate)
    print(fx)
    print("final loss: " + str(lossFunction(fx, data)))