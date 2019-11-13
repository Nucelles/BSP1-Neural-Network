"""
Hello World
"""

"""
NEURON CLASSES
"""
from random import random
import numpy as np
import math
from statistics import mean
import ActivationFunctions as ac


class Neuron:

    def __init__(self, numberOfInputs, activationFunction):
        self.activationFunction = activationFunction
        self.weights = [random() for i in range(numberOfInputs)]
        self.bias = 1
        self.output = 0

    def defineWeights(self, numberOfWeights):
        """
        This function defines the randomized weights for the neuron
        :param numberOfWeights:
        :return:
        """
        self.weights = np.random.rand(numberOfWeights)

    def runNeuron(self, inputs):
        """
        This function will run the other functions of the neurons, all the necessary functions to process the input.
        :param inputs: list of floats, the array of previous neuron outputs of the last layer
        :return:
        """
        dotOutput = self.applyDot(inputs)
        output = self.applyActivationFunction(dotOutput)

        return output

    def applyDot(self, inputs):
        """
        This function use the dot function to on the input and weights.
        :param inputs: list of floats, the array of previous neuron outputs of the last layer
        :return:
        """

        dotOutput = 0

        for currInput in range(len(inputs)):

            dotOutput += inputs[currInput] * self.weights[currInput]

        return dotOutput

    def applyActivationFunction(self, dotOutput):
        return self.activationFunction(dotOutput)

"""
Below is the definition of the layer class 
"""


class Layer:
    def __init__(self, layers, activationFunction):
        self.layers = layers
        self.neuronList = []
        self.layerOutput = []
        self.previousLayer = None
        self.followingLayer = None

        for i in range(self.layers):
            self.neuronList.append(Neuron(layers, activationFunction))

    def changeActivationFunction(self, activationFunction):
        """
        This function will change the activation function of the layer, and return nothing
        :param activationFunction: string, this is the code of the activation function, see code documentation
        :return:
        """
        for i in self.neuronList:
            i.activationFunction = activationFunction

class softmaxLayer(Layer):
    def __init__(self):
        super().__init__(layers=1, activationFunction="Identity")

    def runLayer(self):
        print(self.layerOutput)
        self.layerOutput = ac.softmax(self.previousLayer.layerOutput)
        print(self.layerOutput)

class NeuronLayer(Layer):

    def __init__(self, layers, activationFunction):
        super().__init__(layers, activationFunction)

    def runLayer(self):
        """
        This function will loop through all the neurons of the layer and activate the runNeuron() function.
        :return:
        """
        self.layerOutput = []
        for neuron in self.neuronList:

            self.layerOutput.append(neuron.runNeuron(self.previousLayer.layerOutput))

    def updateWeights(self, learningRate, networkOutput, expectedOutput):
        """
        This function is part of the back-propagation code. This function should run through each neuron & update their
        base weights and biases
        :param learningRate: flaot, the defined learning rate of the neural network
        :param networkOutput: float, the output of the neural network
        :param expectedOutput: float, this is the expected output of the neural network
        :return:
        """
        for n in range(len(self.neuronList)):
            for i in range(len(self.neuronList[n].weights)):
                Part1 = (-2 * (networkOutput[0] - expectedOutput))
                Part2 = (self.followingLayer.neuronList[i].weights[i] * derivitiveSigmoid(sum(self.followingLayer.layerOutput)))
                Part3 = (self.previousLayer.layerOutput[i] * derivitiveSigmoid(self.layerOutput[0]))
                self.neuronList[n].weights[i] -= learningRate * Part1 * Part2 * Part3


class inputLayer(Layer):
    def __init__(self, layerSize, activationFunction = ac.identity):
        super().__init__(layerSize, activationFunction)

    def runLayer(self, inputData):
        """
        This function will loop through all the neurons of the layer, but instead of running the runNeuron layer it will
        change the output of each neuron.(The number of neurons in the layer must match the size of the inputData array)
        :param inputData: list of floats, This is the unchanged input data that will be feed to the neuron.
        :return:
        """
        for i in range(self.layers):
            self.neuronList[i].output = inputData[i]
            self.layerOutput.append(inputData[i])

class outputLayer(Layer):
    def __init__(self, layerSize, activationFunction):
        super().__init__(layerSize, activationFunction)

    def runLayer(self):
        """
        This function will loop through all the neurons of the layer and activate the runNeuron() function.
        :return:
        """
        self.layerOutput = []
        for neuron in self.neuronList:
            self.layerOutput.append(neuron.runNeuron(self.previousLayer.layerOutput))

    def updateWeights(self, learningRate, networkOutput, expectedOutput):
        """
        This function is part of the back-propagation code. This function should run through each neuron & update their
        base weights and biases
        :param learningRate: flaot, the defined learning rate of the neural network
        :param networkOutput: float, the output of the neural network
        :param expectedOutput: float, this is the expected output of the neural network
        :return:
        """
        for neuron in self.neuronList:
            for i in range(len(neuron.weights)):
                Part1 = (-2 * (networkOutput - expectedOutput))
                Part3 = (self.previousLayer.layerOutput[i] * derivitiveSigmoid(self.layerOutput))
                neuron.weights[i] -= learningRate * Part1 * Part3





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


    def feedNetwork(self, x):
        """
        This function will feed an input through the network once, without back-propagation
        :param x: list of floats, this is the input for the network
        :return:
        """

        self.layers[0] = self.layers[0].runLayer(x)
        for layer in self.layers[1:]:
            layer.runLayer()

        return self.layers[-1].layerOutput

    def add(self, layerToAdd):
        """
        This function wil add a layer to the network in order. Only one layer can be added at a time.
        :param layerToAdd: This is the layer class object that will be added.
        :return:
        """
        assert issubclass(type(layerToAdd), Layer)
        self.layers.append(layerToAdd)
        if type(layerToAdd) != inputLayer:
            self.layers[-1].previousLayer = self.layers[-2]
            self.layers[-2].followingLayer = self.layers[-1]

            #for neuron in self.layers[-1].neuronList:
             #   neuron.defineWeights(len( self.layers[-2].neuronList))

    def runNetwork(self, inputData, learningRate, epochs):
        """
        This function will receive the training set and run the neural network.
        :param data: This is the list of data that will be fed to the neural network.
        :return:
        """
        if len(self.layers[0].neuronList) != len(inputData[0]):

            for currentEpoch in range(epochs):
                for currentX, predictedY in zip(inputData[0], inputData[1]):

                    self.layers[0] = self.layers[0].runLayer(currentX)

                    for layer in self.layers[1:]:
                        layer.runLayer()

                    networkOutput = self.layers[-1].layerOutput

                    """
                    for layer in self.layers[1:]:
                        layer.updateWeights(learningRate, networkOutput, predictedY)
                    """

                if epochs % 10 == currentEpoch:
                    print("At epoch %" % currentEpoch)
                    yPredictions = [self.feedNetwork(x) for x in inputData[0]]
                    loss = self.meanSquaredLoss(networkOutput, yPredictions)

        else:
            print("The amount of neurons in the input layer does not match the size of the x input.")

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

    def inputWeights(self, presetWeights):
        """
        This function will receive a structured array of weights which it will input into each layer of the network
        :param presetWeights: list of list of float, this is a structured array of weights that is needed adding a
        predefined weight.
        :return:
        """
        for layer in range(1, len(self.layers)):
            for neuron in range(len(self.layers[layer].neuronList)):
                self.layers[layer].neuronList[neuron].weights = presetWeights[layer-1][neuron]

    def meanSquaredLoss(self, output, actual):
        """
        This function will calculate the mean squared loss of a neuron.
        :param output: This is the output of the model
        :param actual: This is the actual intended output
        :return:
        """
        if len(output) == len(actual):
            meanList = [(output[i]-actual[i])**2 for i in range(len(output))]
            return mean(meanList)
        else:
            print("The output and actual results length do not match.")

        return



