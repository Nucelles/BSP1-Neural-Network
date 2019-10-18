"""
Hello World
"""

"""
Below is the defintion of the Neuron class
"""
import numpy as np
import math

class Neuron:

    def __init__(self, numberOfInputs, activationFunction):
        self.activationFunction = activationFunction
        self.weights = np.random.rand(numberOfInputs)
        self.weightsPreset = []
        self.bias = 0
        self.output = 0

    def runNeuron(self, inputs):
        """
        This function will run the other functions of the neurons, all the necessary functions to process the input.
        :return:
        """
        dotOutput = self.applyDot(inputs)
        output = self.applyActivationFunction(dotOutput)
        # print(dotOutput)
        # print(output)

        return output

    def applyDot(self, inputs):
        """
        This function use the dot function to on the input and weights.
        Input = [Inputs]
        """

        dotOutput = 0
        for currInput in range(len(inputs)):
            dotOutput += inputs[currInput] * self.weights[currInput]

        return dotOutput

    def applyActivationFunction(self, dotOutput):
        """
        This function will apply the sigmoid function to the sum of the inputs.
        Input = SumOfInputs, Output = finalOutput
        """
        if self.activationFunction == "Sigmoid":
            self.output = 1 / (1 + math.exp(-dotOutput))
        elif self.activationFunction == "ReLu":
            # ReLu activation function
            if dotOutput < 0:
                self.output = 0
            else:
                self.output = dotOutput

        elif self.activationFunction == "Leaky ReLu":
            # Leaky ReLu activation function
            if dotOutput < 0:
                self.output = 0
            else:
                self.output = 0.01*dotOutput
        else:
            print("Activation function not recognized, using default Sigmoid")
            self.output = 1 / (1 + math.exp(-dotOutput))

        return self.output

"""
Below is the definition of the layer class 
"""

class Layer:
    def __init__(self, layers, activationFunction):
        self.layers = layers
        self.neuronList = []
        self.layerOutput = []

        for i in range(self.layers):
            self.neuronList.append(Neuron(layers, activationFunction))


class NeuronLayer(Layer):

    def __init__(self, layers, activationFunction):
        self.previousLayer = None
        super().__init__(layers, activationFunction)

    def runLayer(self):
        self.layerOutput = []
        for neuron in self.neuronList:
            self.layerOutput.append(neuron.runNeuron(self.previousLayer.layerOutput))


class inputLayer(Layer):
    def __init__(self, inputSize, inputData, activationFunction):
        super().__init__(inputSize, activationFunction)

        for i in range(self.layers):
            self.neuronList[i].output = inputData[i]
            self.layerOutput.append(inputData[i])

    def runLayer(self):
        self.layerOutput = []
        for neuron in self.neuronList:
            self.layerOutput.append(neuron.applyActivationFunction())


"""
Below is the definition of the network class
"""


class Network:
    def __init__(self, layerList = None):
        self.layers = []
        self.connected = False

        if layerList != None:
            for i in layerList:
                assert issubclass(type(i), Layer)
                self.layers.append(i)

    def add(self, layerToAdd):
        """
        This function wil add a layer to the network in order. Only one layer can be added at a time.
        :param layerToAdd: This is the layer class object that will be added.
        :return:
        """
        assert issubclass(type(layerToAdd), Layer)
        self.layers.append(layerToAdd)

    def compile(self):
        """
        This function will connect each layer of the neuron and will check that the first layer is an inputLayer
        :return:
        """
        if type(self.layers[0]) == inputLayer:
            for i in range(len(self.layers), 0, -1):
                self.layers[i].previousLayer = self.layers[i-1]
        else:
            print("Error: The first layer of the network must be an input layer.")
            return

    def runNetwork(self, data):
        """
        This function will receive the training set and run the neural network.
        ;param data: This is the list of data that will be fed to the neural network.
        :return:
        """

        for layer in self.layers:
            layer.runLayer()

    def saveWeights(self):
        """
        This function will "save" the weights of each layer.
        :return:
        """
        networkWeights = []
        layerWeights = []

        for layer in self.layers:
            for neuron in layer.neuronList:
                layerWeights.append(neuron.weights)
            networkWeights.append(layerWeights)
            layerWeights = []

        return networkWeights



