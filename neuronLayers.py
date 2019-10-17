import Neuron

class layer():
    def __init__(self, layers, activationFunction):
        self.layers = layers
        self.layerList = []
        self.layerOutput = []

        for i in range(self.layers):
            self.layerList.append(Neuron.Neuron(layers, activationFunction))


class neuronLayer(layer):

    def __init__(self, layers, activationFunction, previousLayer):
        self.previousLayer = previousLayer
        super().__init__(layers, activationFunction)

    def runLayer(self):
        self.layerOutput = []
        for neuron in self.layerList:
            self.layerOutput.append(neuron.runNeuron(self.previousLayer.layerOutput))


class inputLayer(layer):
    def __init__(self, inputSize, inputData, activationFunction):
        super().__init__(inputSize, activationFunction)

        for i in range(self.layers):
            self.layerList[i].output = inputData[i]
            self.layerOutput.append(inputData[i])
