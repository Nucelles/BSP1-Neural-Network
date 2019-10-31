

# The test cases need to be changed for the new network class format

import NeuralNetwork as nn
"""
This is the Test Case for the first link, https://enlight.nyc/projects/neural-network/
"""

modelB = nn.Network()

modelB.add(layerToAdd=nn.inputLayer(2, "Sigmoid"))
modelB.add(layerToAdd=nn.NeuronLayer(3, "Sigmoid"))
modelB.add(layerToAdd=nn.outputLayer(1, "Sigmoid"))

presetWeights = [[], []], [[.2, .6], [.1, .8], [.3, .7]], [[.4, .5, .9]]
modelB.inputWeights(presetWeights)

print("Test Case 1")
print("Expected = 0.8579443067")
print("Network =", modelB.feedNetwork([2, 9]))
print("")


"""
This is the Test Case for the second link, https://victorzhou.com/blog/intro-to-neural-networks/
"""

modelA = nn.Network()

modelA.add(layerToAdd=nn.inputLayer(2, "Sigmoid"))
modelA.add(layerToAdd=nn.NeuronLayer(2, "Sigmoid"))
modelA.add(layerToAdd=nn.outputLayer(1, "Sigmoid"))

presetWeights = [[], []], [[0, 1], [0, 1]], [[0, 1]]
modelA.inputWeights(presetWeights)

print("Test Case 2")
print("Expected = 0.7216325609518421")
print("Network  =", modelA.feedNetwork([2, 3]))

"""
Used for debugging the layers in the model

print("\nLayers in network")
for i in model.layers:
  print("\n", i)
  print("%a is the previous layer." % i.previousLayer)
  print("%a is the following layer." % i.followingLayer)
"""



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