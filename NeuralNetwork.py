from statistics import mean
from time import perf_counter
from datetime import datetime
from random import shuffle
import Layer
from tqdm import tqdm
import pickle


class Network:
    """This is Network class, it controls the actions of it's layer and the layer's neurons.

    :param layers: List holds the layer objects in the network
    :type layers: list
    """

    def __init__(self, layerList = None):
        """Constructor Method

        :param layerList: This is an optional list input that holds layers, for one line whole network creation
        :type layerList: list, optional
        """
        self.layers = []

        if layerList != None:
            for i in layerList:
                assert issubclass(type(i), Layer)
                self.add(i)

    def compile(self):
        """This method compiles the network, this means that it connects each layer and initializes their weights.
        """

        for n in self.layers[0].neuronList:
            n.weights = []

        for layer in self.layers[1:]:
            layer.initiateWeights()

    def feedNetwork(self, x, debug = False):
        """This function will feed an input through the network once, without back-propagation.

        :param x: list of floats, this is the input for the network
        :type x: list
        :param debug: Boolean flag for turning on debug mode
        :type debug: bool, optional

        :return: The output of the network
        :rtype: list
        """
        if debug:
            print("---------------FORWARD PROPAGATION---------------")
            print("----- Running", self.layers[0], "-----")
        self.layers[0].runLayer(x)

        for layer in self.layers[1:]:
            if debug: print("----- Running", layer, "-----")
            layer.runLayer(False)

        return self.layers[-1].layerOutput

    def add(self, layerToAdd):
        """This function wil add a layer to the network in order. Only one layer can be added at a time.

        :param layerToAdd: This is the layer class object that will be added.
        :type layerToAdd: Layer
        """
        assert issubclass(type(layerToAdd), Layer.Layer)
        self.layers.append(layerToAdd)
        if type(layerToAdd) != Layer.inputLayer:
            self.layers[-1].previousLayer = self.layers[-2]
            self.layers[-2].followingLayer = self.layers[-1]

    def runNetwork(self, inputData, learningRate, epochs, modelName = "default", debug = False):
        """This method will receive the training set and run the neural network under the given parameters.

        :param inputData: This is the list of data that will be fed to the neural network.
        :type inputData: list
        :param learningRate: Learning rate of the neural network during backpropo\agation
        :type learningRate: float
        :param epochs: The amount of times the neural network will loop over the training set
        :type epochs: int
        :param debug: Boolean flag for turning on debug mode
        :type debug: bool, optional;
        """

        #self.printNetwork()

        if len(self.layers[0].neuronList) == len(inputData[0][0]):

            epochResults = []
            totalTime = 0
            for currentEpoch in range(epochs):

                epochTimeIn = perf_counter()

                print("Epoch {}".format(currentEpoch + 1))

                if debug :
                    print("\n\n","|"*35, currentEpoch, "|"*35)
                    print("---------------FORWARD PROPAGATION---------------")

                inputNum = 1
                shuffle(inputData)
                for currentInput in tqdm(inputData, desc="Input"):
                    #print("Current Input # = {}".format(inputNum))
                    inputNum += 1
                    currentX, predictedY = currentInput[0], currentInput[1]

                    if debug:
                        print("X = ", currentX)
                        print("Y = ", predictedY)

                    if debug: print("\n----- Running", self.layers[0], "-----")
                    self.layers[0].runLayer(currentX, debug=debug)

                    for layer in self.layers[1:]:
                        if debug: print("\n----- Running", layer, "-----")
                        layer.runLayer(debug=debug)

                    networkOutput = self.layers[-1].layerOutput
                    if debug: print(networkOutput)

                    if debug:
                        print("\nNetwork Output for Epoch {} = {}".format(currentEpoch, networkOutput))
                        print("\n---------------BACKWARDS PROPAGATION---------------")



                    self.backpropogation(learningRate, predictedY, debug)

                    if debug: self.printNetwork()


                    self.layers[0].layerOutput = []

                epochTimeOut = perf_counter()
                epochTotal = epochTimeOut - epochTimeIn

                # This print needs to be done after each epoch, with the sum of the loss of each image.
                #print("Mean Squared Loss of Epoch = {}".format(self.meanSquaredLoss(networkOutput, predictedY)))
                print("Time Taken = {0:4.3}secs.\n".format(epochTotal))



                if currentEpoch != epochs - 1:
                    totalTime += epochTotal
                else:
                    self.saveNetwork(name=modelName)
                    print("Final Time (in Seconds) = {}".format(totalTime + epochTotal), "\n")

        else:
            print("The amount of neurons in the input layer does not match the size of the x input.")
            print("The length of the inputLayer is", len(self.layers[0].neuronList))
            print("The length of the x_input is", len(inputData[0][0]))

    def backpropogation(self, learningRate, predictedY, debug = False):
        """Method used during training to backpropagate through the network and update the weights of each neuron in the network.

        :param learningRate: Learning rate of the neural network during backpropo\agation
        :type learningRate: float
        :param predictedY: The expected output of the neural network
        :type predictedY: list
        :param debug: Boolean flag for turning on debug mode
        :type debug: bool, optional

        |
        """

        rangeReverse = list(range(1, len(self.layers)))
        rangeReverse.reverse()
        for layer in rangeReverse:
            self.layers[layer].calcPartialDerivatives(predictedY, debug)

        for layer in self.layers[1:]:
            for neuron in layer.neuronList:
                neuron.oldWeights = neuron.weights

        for layer in self.layers[1:]:
            if debug: print("\n----- Backpropagating", layer, "-----")
            layer.updateWeights(learningRate, debug)

    def saveNetwork(self, name = "default"):
        """This method will pickle the object, and save it to a /model folder.

        """

        if name != "default":
            fullFileName = 'model\{}.obj'.format(name)
        else:
            name = (datetime.now().strftime("model_%d-%m-%Y_%I-%M-%S_%p"))
            fullFileName = 'model\{}.obj'.format(name)

        with open(fullFileName, 'wb') as objectFile:
            pickle.dump(self, objectFile)

        print("{} Saved!".format(fullFileName))

    def printNetwork(self, debug = True):
        """This method will print the Network in a readable format, with the description of each Neuron

        :param debug: Boolean flag for turning on debug mode
        :type debug: bool, optional
        """

        print(self)
        for l in self.layers:
            print(" "*4*1, l)
            print(" " * 4*2, "Number of Neuron in Layer = {}".format(len(l.neuronList)))
            print(" " * 4*2, "Activation Function of Layer = {}".format(l.activationFunction))

            print()

    def inputWeights(self, presetWeights):
        """
        This function will receive a structured array of weights which it will input into each layer of the network

        :param presetWeights: list of list of float, this is a structured array of weights that is needed adding a predefined weight.
        :type presetWeights: list
        """
        for layer in range(1, len(self.layers)):
            for neuron in range(len(self.layers[layer].neuronList)):
                self.layers[layer].neuronList[neuron].weights = presetWeights[layer-1][neuron]

    def meanSquaredLoss(self, output, actual):
        """This function will calculate the mean squared loss of a neuron.

        :param output: This is the output of the model
        :type output: list
        :param actual: This is the actual intended output
        :type actual: list
        """

        if len(output) == len(actual):
            mseSum = sum([(output[i] - actual[i])**2 for i in range(len(output))])
            return mseSum/len(output)

            # = [0.2(output[i]-actual[i])**2 for i in range(len(output))]
            #return mean(meanList)
        else:
            return "The output and actual results length do not match.\n" \
                   "Output Length = {}, Predicted Length = {} ".format(len(output), len(actual))



