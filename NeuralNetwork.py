"""
Hello World
"""

from random import random
import numpy as np
import math
from statistics import mean
import ActivationFunctions as ac
from time import perf_counter
from datetime import datetime
import pickle


class Neuron:

    def __init__(self, numberOfInputs, activationFunction):
        self.debug = False
        self.partialDerivative = None
        self.activationFunction = activationFunction
        self.weights = [random() for i in range(numberOfInputs)]
        self.oldWeights = self.weights
        self.bias = 0
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

        self.output = output

        return output

    def applyDot(self, inputs):
        """
        This function use the dot function to on the input and weights.
        :param inputs: list of floats, the array of previous neuron outputs of the last layer
        :return:
        """

        dotOutput = 0
        if self.debug:
            print("Inputs = {}".format(inputs))
            print("Weights = {}".format(self.weights))

        debugPrint = ["Weighted Sum = ("]
        for currInput in range(len(inputs)):
            if self.debug:
                dotOutput += inputs[currInput] * self.weights[currInput]
                debugPrint.append("{}*{} + ".format(inputs[currInput], self.weights[currInput]))
            else:

                dotOutput += inputs[currInput] * self.weights[currInput]

        debugPrint.append("{}) = {}".format(self.bias, dotOutput))
        if self.debug: print(("".join(debugPrint)))

        return dotOutput + self.bias

    def applyActivationFunction(self, dotOutput):
        if self.debug:
            act = self.activationFunction(dotOutput)
            print("Activation Function Output = {}".format(act))
            return act
        else:
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
        self.activationFunction = activationFunction

    def initiateWeights(self):
        for i in range(self.layers):
            self.neuronList.append(Neuron(self.previousLayer.layers, self.activationFunction))

    def changeActivationFunction(self, activationFunction):
        """
        This function will change the activation function of the layer, and return nothing
        :param activationFunction: string, this is the code of the activation function, see code documentation
        :return:
        """
        for i in self.neuronList:
            i.activationFunction = activationFunction

    def updateWeights(self, learningRate, expectedOutput, debug):

        for neuron in range(len(self.neuronList)): # Will loop through the range of neuronList
            currNeuron = self.neuronList[neuron] # Will assign the currentNeuron being worked on

            if debug:
                print()

            for weight in range(len(currNeuron.weights)): # Will loop through the range of weights in the currentNeuron

                d_E_d_Neuron = currNeuron.partialDerivative
                d_Neuron_d_NeuronOutput = ac.derivitiveSigmoid(currNeuron.output)
                d_NeuronOutput_d_PreviousNeuronInput = self.previousLayer.layerOutput[weight]
                d_E_d_Weight = (d_E_d_Neuron * d_Neuron_d_NeuronOutput * d_NeuronOutput_d_PreviousNeuronInput)
                weightChange = learningRate * d_E_d_Weight

                oldWeight = currNeuron.weights[weight]
                currNeuron.weights[weight] -= weightChange

                if debug:
                    print("(Neuron {}, Weight {}), Weight = {}".format(neuron, weight, oldWeight))
                    print("Weight Change = ({} * ({} * {} * {})".format(learningRate, d_E_d_Neuron, d_Neuron_d_NeuronOutput, d_NeuronOutput_d_PreviousNeuronInput))
                    print("              = ({} * {})".format(learningRate, d_E_d_Weight))
                    print("              =", weightChange)


                # Will update the weight by applying the derivative Chain rule, shown below.
                # w := w - learningRate*[d_E/d_ConnectingNeuron * d_OutputOfConnectingNeuron/d_WeightedSumOfConnectingNeuron * d_WeighedSumOfTheConnectingNeuron/d_WeightBeingUpdated]
                # w := w - learningRate * [Derivative Of Current Neuron towards Error * Derivative of the Activation Function * Output of Connecting Weight Neuron]

class softmaxLayer(Layer):
    def __init__(self, layer):
        super().__init__(layers=layer, activationFunction=ac.identity)

    def runLayer(self):
        neuronLayerOutput = []
        for neuron in self.neuronList:
            neuronLayerOutput.append(neuron.runNeuron(self.previousLayer.layerOutput))

        neuronLayerOutput = ac.softmax(neuronLayerOutput)

        for softmaxValue in neuronLayerOutput:
            self.layerOutput.append(softmaxValue)


class NeuronLayer(Layer):

    def __init__(self, layers, activationFunction):
        super().__init__(layers, activationFunction)

    def runLayer(self, debug):
        """
        This function will loop through all the neurons of the layer and activate the runNeuron() function.
        :return:
        """
        self.layerOutput = []
        for neuron in self.neuronList:
            if debug:
                print("\n", neuron)
                neuron.debug = True
            self.layerOutput.append(neuron.runNeuron(self.previousLayer.layerOutput))
            neuron.debug = False

    def calculatePartialDerivatives(self, predictedOutput, debug):
        sumOfNeuron = []
        for neuron in range(len(self.neuronList)):
            currentNeuron = self.neuronList[neuron]
            #print(currentNeuron)


            for weight in range(self.followingLayer.layers):

                d_Error_d_Neuron = self.followingLayer.neuronList[weight].partialDerivative
                d_Neuron_d_NeuronOutput = ac.derivitiveSigmoid(currentNeuron.output)
                d_NeuronOutput_d_WeightNeuron = self.followingLayer.neuronList[weight].weights[neuron]

                neuronDerivative = d_Error_d_Neuron * d_Neuron_d_NeuronOutput * d_NeuronOutput_d_WeightNeuron

                sumOfNeuron.append(neuronDerivative)
                if debug:
                    print("Current Iteration: {}".format(weight))
                    print(self.followingLayer.neuronList[weight].weights)
                    print("Neuron Iteration Partial = ({} * {} * {})".format(d_Error_d_Neuron, d_Neuron_d_NeuronOutput, d_NeuronOutput_d_WeightNeuron))
                    print("                         = ({})".format(neuronDerivative))
                    print("                         =", neuronDerivative,"\n")

            currentNeuron.partialDerivative = sum(sumOfNeuron)
            if debug:
                print("Total Sum = {}".format(sumOfNeuron))
                print("          =", sum(sumOfNeuron))
            sumOfNeuron.clear()

class inputLayer(Layer):
    def __init__(self, layerSize, activationFunction = ac.identity):
        super().__init__(layerSize, activationFunction)

        self.neuronList = [Neuron(layerSize, activationFunction) for i in range(layerSize)]

    def runLayer(self, inputData, debug):
        """
        This function will loop through all the neurons of the layer, but instead of running the runNeuron layer it will
        change the output of each neuron.(The number of neurons in the layer must match the size of the inputData array)
        :param inputData: list of floats, This is the unchanged input data that will be feed to the neuron.
        :return;
        """
        if debug:
            print(inputData)
            print(range(self.layers))

        for i in range(self.layers):
            self.neuronList[i].output = inputData[i]
            self.layerOutput.append(inputData[i])




class outputLayer(Layer):
    def __init__(self, layerSize, activationFunction):
        super().__init__(layerSize, activationFunction)

    def runLayer(self, debug):
        """
        This function will loop through all the neurons of the layer and activate the runNeuron() function.
        :return:
        """
        self.layerOutput = []
        for neuron in self.neuronList:
            if debug:
                print("\n", neuron)
                neuron.debug = True
            self.layerOutput.append(neuron.runNeuron(self.previousLayer.layerOutput))
            neuron.debug = False

    def calculatePartialDerivatives(self, predictedOutput, debug):
        for neuron in range(len(self.neuronList)):
            currentNeuron = self.neuronList[neuron]
            error = self.layerOutput[neuron] - predictedOutput[neuron]
            currentNeuron.partialDerivative = error

            if debug:
                # print(currentNeuron)
                print("Error of Output Neuron = ({} - {})".format(self.layerOutput[neuron], predictedOutput[neuron]))
                print("                = {}\n".format(error))

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

    def compile(self):
        for layer in self.layers[1:]:
            layer.initiateWeights()

    def feedNetwork(self, x, debug):
        """
        This function will feed an input through the network once, without back-propagation
        :param x: list of floats, this is the input for the network
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
        assert issubclass(type(layerToAdd), Layer)
        self.layers.append(layerToAdd)
        if type(layerToAdd) != inputLayer:
            self.layers[-1].previousLayer = self.layers[-2]
            self.layers[-2].followingLayer = self.layers[-1]


    def runNetwork(self, inputData, learningRate, epochs, debug):
        """
        This function will receive the training set and run the neural network.
        :param data: This is the list of data that will be fed to the neural network.
        :return:
        """
        epochTotal = 0
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
                    self.printNetwork()

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


    def printNetwork(self):

        print(self)
        for l in self.layers:
            print("    ", l)
            for n in l.neuronList:
                print("        ", n)
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



