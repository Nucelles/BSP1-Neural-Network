"""
@package docstring
This document is used for housing the class Network, which is the main controller of the layer and neuron class.
"""
from statistics import mean
from time import perf_counter
from datetime import datetime
import Layer
import pickle


class Network:
    def __init__(self, layerList = None):
        self.layers = []
        self.connected = False

        if layerList != None:
            for i in layerList:
                assert issubclass(type(i), Layer)
                self.layers.append(i)

    def compile(self):
        for layer in self.layers[1:]:
            layer.initiateWeights()

    def feedNetwork(self, x, debug):
        """
        This function will feed an input through the network once, without back-propagation
        :param x list of floats, this is the input for the network
        :return:
        """
        if debug:
            print("---------------FORWARD PROPAGATION---------------")
            print("----- Running", self.layers[0], "-----")
        self.layers[0].runLayer(x)

        for layer in self.layers[1:]:
            if debug: print("----- Running", layer, "-----")
            layer.runLayer()

        return self.layers[-1].layerOutput

    def add(self, layerToAdd):
        """
        This function wil add a layer to the network in order. Only one layer can be added at a time.
        :param layerToAdd: This is the layer class object that will be added.
        :return:
        """
        assert issubclass(type(layerToAdd), Layer.Layer)
        self.layers.append(layerToAdd)
        if type(layerToAdd) != Layer.inputLayer:
            self.layers[-1].previousLayer = self.layers[-2]
            self.layers[-2].followingLayer = self.layers[-1]


    def runNetwork(self, inputData, learningRate, epochs, debug):
        ##This function will receive the training set and run the neural network.
        #@param data This is the list of data that will be fed to the neural network.

        epochTotal = 0

        self.printNetwork(False)
        if len(self.layers[0].neuronList) == len(inputData[0][0]):

            totalTime = 0
            for currentEpoch in range(epochs):
                epochTimeIn = perf_counter()
                print("Epoch {}".format(currentEpoch))

                if debug :
                    print("\n\n","|"*35, currentEpoch, "|"*35)
                    print("---------------FORWARD PROPAGATION---------------")
                for currentInput in inputData:
                    currentX, predictedY = currentInput[0], currentInput[1]

                    if debug: print("\n----- Running", self.layers[0], "-----")
                    self.layers[0].runLayer(currentX, debug=debug)

                    for layer in self.layers[1:]:
                        if debug: print("\n----- Running", layer, "-----")
                        layer.runLayer(debug=debug)

                    networkOutput = self.layers[-1].layerOutput

                    if debug:
                        print("Network Output for Epoch {} = {}".format(currentEpoch, networkOutput))
                        print("\n---------------BACKWARDS PROPAGATION---------------")

                    self.backpropogation(learningRate, predictedY, debug)

                    self.layers[0].layerOutput = []

                epochTimeOut = perf_counter()
                epochTotal = epochTimeOut - epochTimeIn
                print("Time Taken = {0:4.3}\n".format(epochTotal))

                if currentEpoch != epochs - 1:
                    totalTime += epochTotal
                else:
                    print("Final Time (in Seconds) = {}".format(totalTime + epochTotal), "\n")

        else:
            print("The amount of neurons in the input layer does not match the size of the x input.")
            print("The length of the inputLayer is", len(self.layers[0].neuronList))
            print("The length of the x_input is", len(inputData[0]))

    def backpropogation(self, learningRate, predictedY, debug):
        rangeReverse = list(range(1, len(self.layers)))
        rangeReverse.reverse()
        for layer in rangeReverse:
            self.layers[layer].calculatePartialDerivatives(predictedY, debug)

        for layer in self.layers[1:]:
            for neuron in layer.neuronList:
                neuron.oldWeights = neuron.weights

        for layer in self.layers[1:]:
            if debug: print("\n----- Backpropagating", layer, "-----")
            layer.updateWeights(learningRate, predictedY, debug)

        return

    def saveNetwork(self, model):
        """
        This function will pickle the object, and save it to a /model folder.
        :return:
        """
        name = (datetime.now().strftime("model_%d-%m-%Y_%I-%M-%S_%p"))
        fullFileName = 'model\{}.obj'.format(name)
        object = model

        with open(fullFileName, 'wb') as objectFile:
            pickle.dump(object, objectFile)


    def printNetwork(self, debug):

        print(self)
        for l in self.layers:
            print("    ", l)
            for n in l.neuronList:
                print("        ", n)
                if debug:
                    print("             partialDerivative = {}".format(n.partialDerivative))
            print()


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



