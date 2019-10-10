import Neuron as neuron
import numpy as np

testCaseWeight = [0.5, 0.7, 1.95, 2]
testCaseInput = [[1, 2, 3, 4],
                 [4, 3, 2 , 1],
                 [-1, 2, -3, 4],
                 [0.1251235, 0.256536, 0.13123, 0.7644]]

testNeuron = neuron.Neuron()
print(testNeuron.weights)
testNeuron.weights = testCaseWeight
print(testNeuron.weights, "\n")

for i in range(len(testCaseInput)):
    # Calculates the dot using the neuron function
    testNeuron.applyDot(testCaseInput[i])
    testNeuron.getSum(False)
    neuronResult = testNeuron.currentSum

    # Calculates the dot using the numpy.dot function
    numpyResult = np.dot(np.array(testCaseInput[i]), np.array(testCaseWeight))

    print("Test Case {}:\n".format(i+1), testCaseInput[i], "x", testCaseWeight)
    print("Neuron =", neuronResult)
    testNeuron.currentSum = 0
    print("Numpy = ", numpyResult)
    print(15*"-")
