

# The test cases need to be changed for the new network class format


"""
inputlayer1 = layer.inputLayer(5, trainingInput, "Sigmoid")

layer1 = layer.neuronLayer(5, "Leaky ReLu")
layer2 = layer.neuronLayer(6, "Sigmoid")
layer3 = layer.neuronLayer(7, "ReLu")

mainlayers = [layer1, layer2, layer3]

for i in mainlayers:
    i.runLayer()
    print(i.layerOutput)
    print(15*"--")


Below is the last test case for only testing the neuron dot function
against numpy
testCaseWeight = [0.5, 0.7, 1.95, 2]
testCaseInput = [[1, 2, 3, 4],
                 [4, 3, 2 , 1],
                 [-1, 2, -3, 4],
                 [0.1251235, 0.256536, 0.13123, 0.7644]]
                 
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
"""