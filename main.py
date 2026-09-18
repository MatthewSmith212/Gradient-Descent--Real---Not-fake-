from data_and_function.retrieve_data import readData, displayData
from loss_function import lossFunction
from calc_gradients import calcGradients
import numpy as np

"""
Based on cheating, using the actual function, the minimum achievable loss is ~0
"""

def descendGradient(lossFunc: function, params: np.ndarray, data: np.ndarray, domain: np.ndarray, h: np.float64, passes: int, learningRate: float, momentum:float):
    grads = np.zeros(len(params), dtype=np.float64)
    for _ in range(passes):
        oldGrads = grads
        grads = calcGradients(lossFunc=lossFunc, params=params, data=data, domain=domain,h=h)
        for i in range(len(grads)):
            #grads[i] += (momentum * oldGrads[i]) #turn to velocity with momentum
            params[i] -= (learningRate * grads[i])
    return params

if __name__ == "__main__":
    #showSavedData("data_and_function/testScores.pkl")
    data = np.array(readData("data_and_function/testScores.pkl"), dtype=np.float64)
    fx = np.array([110, -4], dtype=np.float64) #fx[i] is the coefficient to the term x^i

    domain = np.array([min(data[:, 0]), max(data[:, 0])], dtype=np.float64)

    print("initial loss: " + str(lossFunction(fx, data, domain)))

    diffPrecision = 8
    passes = 500
    learningRate = 0.005
    momentum = 0.9

    differential = np.float64(1) ** (-diffPrecision)
    
    fx = descendGradient(lossFunc=lossFunction, params=fx, data=data, domain=domain, h=differential, passes=passes, learningRate=learningRate, momentum=momentum) # type: ignore
    print(fx)
    print("intermediate loss: " + str(lossFunction(fx, data, domain)))

    learningRate = 0.001
    fx = descendGradient(lossFunc=lossFunction, params=fx, data=data, domain=domain, h=differential, passes=passes, learningRate=learningRate, momentum=momentum) # type: ignore #do 500 more passes at a low learning rate
    print(fx)
    print("final loss: " + str(lossFunction(fx, data, domain)))