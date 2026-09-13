from true_function import baseFunction
from retrieve_data import displayData, displayDistribution
import pickle
import numpy as np
from numpy import random
from scipy.stats import levy



def generate_inputs(m):
    #size is outer dimension, inner dimension

    #this generate random x values according to a levy stable distribution, truncated at 40
    data = np.zeros(shape=(m, 2), dtype="f")
    inputs = levy.rvs(loc=0, scale=3, size=(m, 1))
    inputs = np.clip(inputs, a_min=0, a_max=25)
    data = np.add(data, inputs)
    return data

def noislyApplyFunction(orderedPair, noise):
    # noise is a percentage value representing maximum +- (percentage) variation from the true value of the function. 
    factor = ((random.random()- 0.5) * 2 *  noise / 100) + 1
    orderedPair[1] = baseFunction(orderedPair[0]) * factor
    return orderedPair

def efficientlyApplyFunction(data, noise):
    for i in range(len(data)):
        data[i] = noislyApplyFunction(data[i], noise)
    return data

if __name__ == "__main__":
    inputs = generate_inputs(100)
    displayDistribution(inputs)

    data = efficientlyApplyFunction(inputs, 5)
    displayData(data=data)

    #save data for posterity
    with open("Data and Real Function/testScores.pkl", "wb") as file:
        pickler = pickle.Pickler(file=file, protocol=pickle.HIGHEST_PROTOCOL)
        pickler.dump(data)
