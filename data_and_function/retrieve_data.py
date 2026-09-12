import pickle
import numpy as np
import matplotlib.pyplot as plt

def displayData(data):
    ind = np.argsort(data[:,0])
    sortedData = data[ind] 
    plt.plot(sortedData[:, 0], sortedData[:, 1])
    plt.show()

def displayDistribution(data):
    data = data[:, 0].reshape(data.shape[0])
    fig, ax = plt.subplots()
    ax.ecdf(data)
    plt.show()

def readData(path):
    data = list()
    with open(path, "rb") as file:
        unpickler = pickle.Unpickler(file)
        data = unpickler.load()
    return data
    