from Neuron import Neuron
import ActivationFunctions as ac


class Layer:
    """Base Layer class that is used as the superclass for each different network layer.

    :param layers: Number of layers in the network
    :type layers: int
    :param neuronList: List of all neurons in the layer
    :type neuronList: list
    :param layerOuput: The output of the layer in list form
    :type layerOutput: list
    :param previousLayer: The layer that is before the current layer
    :type previousLayer: Layer
    :param followingLayer: The layer that is after the current layer
    :type followingLayer: Layer
    :param activationFunction: The activation function of the layer (and subsequently each neuron)
    :type activationFunction: function
    """

    def __init__(self, layers, activationFunction):
        """Constructor method for Layer

        :param layers: The number of neurons that will be in the layer
        :type layers: int
        :param activationFunction: The activation function of the layer
        :type activationFunction: function
        """
        self.layers = layers
        self.neuronList = []
        self.layerOutput = []
        self.previousLayer = None
        self.followingLayer = None
        self.activationFunction = activationFunction
        self.derivActivation = ac.derivitiveSigmoid

    def runLayer(self, debug = False):
        """This function will loop through all the neurons of the layer and activate the runNeuron() function.

        :param debug: Boolean flag for turning on debug mode
        :type debug: bool, optional
        """
        self.layerOutput = []
        for neuron in self.neuronList:
            if debug:
                print("\n", neuron)
            self.layerOutput.append(neuron.runNeuron(self.previousLayer.layerOutput, debug))

    def initiateWeights(self):
        """This method initialises the weights for each neuron in the layer.
        """
        for i in range(self.layers):
            self.neuronList.append(Neuron(self.previousLayer.layers, self.activationFunction))

    def updateWeights(self, learningRate, debug = True):
        """This method loops through each neuron in the layer and updates their values.

        :param learningRate: The learning rate that will be applied to the change in weight value
        :type learningRate: float
        :param debug: Boolean flag for turning on debug mode
        :type debug: bool, optional
        """

        for neuron in range(len(self.neuronList)): # Will loop through the range of neuronList
            currNeuron = self.neuronList[neuron] # Will assign the currentNeuron being worked on

            if debug:
                print()

            for weight in range(len(currNeuron.weights)): # Will loop through the range of weights in the currentNeuron

                d_E_d_Neuron = currNeuron.partialDerivative
                d_Neuron_d_NeuronOutput = self.derivActivation(currNeuron.output)
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
    """This is the Softmax Layer class, which will apply the softmax function to the outputs of it's neurons
    """
    def __init__(self, layer):
        """Constructor method of softmaxLayer

        :param layer: The number of layers that will be in the network
        :type layer: int
        """
        super().__init__(layers=layer, activationFunction=ac.identity)

    def runLayer(self):
        """This method will run each neuron in the layer, using the runNeuron method.
        """
        neuronLayerOutput = []
        for neuron in self.neuronList:
            neuronLayerOutput.append(neuron.runNeuron(self.previousLayer.layerOutput))

        neuronLayerOutput = ac.softmax(neuronLayerOutput)

        for softmaxValue in neuronLayerOutput:
            self.layerOutput.append(softmaxValue)

class hiddenLayer(Layer):
    """This is the Neuron Layer class, which will run the neuron using the set activation function"""

    def __init__(self, layers, activationFunction):
        """Constructor method of NeuronLayer

        :param layers: The number of layers that iwll be in the network
        :type layers: int
        :param activationFunction: The activation function that will be used in the network
        :type activationFunction: function
        """
        super().__init__(layers, activationFunction)

    def calcPartialDerivatives(self, predictedOutput, debug = False):
        """This method calculates the partial derivatives for each neuron layer to be used during backpropagation

        :param predictedOutput: The predicted output of the network
        :type predictedOutput: list
        :param debug: Boolean flag for turning on debug mode
        :type debug: bool, optional
        """
        sumOfNeuron = []
        for neuron in range(len(self.neuronList)):
            currentNeuron = self.neuronList[neuron]
            if debug: print(currentNeuron)


            for weight in range(self.followingLayer.layers):

                d_Error_d_Neuron = self.followingLayer.neuronList[weight].partialDerivative
                d_Neuron_d_NeuronOutput = self.derivActivation(self.followingLayer.layerOutput[weight])
                d_NeuronOutput_d_WeightNeuron = self.followingLayer.neuronList[weight].weights[neuron]

                neuronDerivative = d_Error_d_Neuron * d_Neuron_d_NeuronOutput * d_NeuronOutput_d_WeightNeuron

                sumOfNeuron.append(neuronDerivative)
                if debug:
                    print("Current Iteration: {}".format(weight))
                    print("Following Layer Output = {}".format(self.followingLayer.layerOutput[weight]))
                    print("Neuron Iteration Partial = ({} * {} * {})".format(d_Error_d_Neuron, d_Neuron_d_NeuronOutput, d_NeuronOutput_d_WeightNeuron))
                    print("                         = ({})".format(neuronDerivative))
                    print("                         =", neuronDerivative,"\n")

            currentNeuron.partialDerivative = sum(sumOfNeuron)
            if debug:
                print("Total Sum = {}".format(sumOfNeuron))
                print("          = {}\n".format(sum(sumOfNeuron)))
            sumOfNeuron.clear()

class inputLayer(Layer):
    """This is te Input Layer class, which is used as the first layer to the network"""

    def __init__(self, layerSize, activationFunction = ac.identity):
        """Constructor method to the Input Layer

        :param layerSize: The amount of neurons in the layer
        :type layerSize: int
        :param activationFunction: The activation function that will be used in the network
        :type activationFunction: function
        """
        super().__init__(layerSize, activationFunction)

        self.neuronList = [Neuron(layerSize, activationFunction) for i in range(layerSize)]

    def runLayer(self, inputData, debug = False):
        """This function will loop through all the neurons of the layer, but instead of running the runNeuron layer it will change the output of each neuron.(The number of neurons in the layer must match the size of the inputData array)

        :param inputData: This is the unchanged input data that will be feed to the neuron.
        :type inputData: list
        :param debug: Boolean flag for turning on debug mode
        :type debug: bool, optional
        """
        self.layerOutput = []

        if debug:
            print(inputData)
            print(list(range(self.layers)))

        for i in range(self.layers):
            self.neuronList[i].output = inputData[i]
            self.layerOutput.append(inputData[i])

class outputLayer(Layer):
    """This is the Output Layer class, which is the last layer to the network"""
    def __init__(self, layerSize, activationFunction):
        """Constructor method to the outputLayer Class

        :param layerSize: The amount of neurons in the layer
        :type layerSize: int
        :param activationFunction: The activation function that will be used in the network
        :type activationFunction: function
        """
        super().__init__(layerSize, activationFunction)

    def calcPartialDerivatives(self, predictedOutput, debug = False):
        """This method calculates the error for each neuron in the layer

        :param predictedOutput: The expected output of the network
        :type predictedOutput: list
        :param debug: Boolean flag for turning on debug mode
        :type debug: bool, optional
        """

        for neuron in range(len(self.neuronList)):
            currentNeuron = self.neuronList[neuron]
            error = self.layerOutput[neuron] - predictedOutput[neuron]
            currentNeuron.partialDerivative = error

            if debug:
                # print(currentNeuron)
                print("Current Neuron -> {}".format(currentNeuron))
                print("Error of Output Neuron = ({} - {})".format(self.layerOutput[neuron], predictedOutput[neuron]))
                print("                = {}\n".format(error))

    def runLayer(self, debug = False):
        """This function will loop through all the neurons of the layer and activate the runNeuron() function.

        :param debug: Boolean flag for turning on debug mode
        :type debug: bool, optional
        """
        #print("\nOutput Layer!")
        self.layerOutput = []

        for neuron in self.neuronList:
            if debug:
                print("\n", neuron)
                print("Previous Layer Output: {}".format(self.previousLayer.layerOutput))

            self.layerOutput.append(neuron.runNeuron(self.previousLayer.layerOutput, False))
