from Neuron import Neuron
import ActivationFunctions as ac


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
