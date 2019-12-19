from random import uniform
import numpy as np

class Neuron:
    """
    This is the base class of the network layers, it is used to perform the mathematical operations on the data.

    :param debug: Bool used to check whether or not to run neuron on debug mode.
    :type debug: bool
    :param partialDerivative: Float will hold the partial derivative value of the neuron used in backpropagation
    :type partialDerivative: float
    :param activationFunction: Function name of the activation function that will be used.
    :type activationFunction: function
    :param weights: List of floats used to represent weights of the neuron. Randomly generated between 0 and 1.
    :type weights: list
    :param oldWeights: List of floats, copy of the weights attribute used for backpropagation
    :type oldWeights: list
    :param bias: Integer representing the bias value
    :type bias: int
    :param output: Float that holds the output of the neuron.
    :type output: float
    """

    def __init__(self, numberOfInputs, activationFunction):
        """
        Constructor method for the Neuron class.

        :param numberOfInputs:  Integer value representing the number of inputs the neuron will receive.
        :type numberOfInputs: int
        :param activationFunction: Function name of the activation function that will be used.
        :type activationFunction: function

        |
        """
        self.debug = False
        self.partialDerivative = None
        self.activationFunction = activationFunction
        self.weights = [uniform(-1, 1) for i in range(numberOfInputs)]
        self.oldWeights = self.weights
        self.bias = 0
        self.output = 0

    def runNeuron(self, inputs, debug = False):
        """
        Method calls each main method used for feeding data through the network.

        :param inputs: List of floats holding the output of the previous layer
        :type inputs: list

        :return: Returns the output of the neuron
        :rtype: float

        |
        """
        dotOutput = self.applyDot(inputs, debug)
        output = self.applyActivationFunction(dotOutput, debug)

        self.output = output

        return output

    def applyDot(self, inputs, debug = False):
        """Method uses the dot function with the inputs and neuron weights.

        :param inputs: list of floats, the array of previous neuron outputs of the last layer
        :type inputs: list

        :return: Returns the dot product of the weights and inputs
        :rtype: float

        |
        """

        dotOutput = 0
        if debug:
            print("Inputs = {}".format(inputs))
            print("Weights = {}".format(self.weights))

        debugPrint = ["Weighted Sum = ("]
        for currInput in range(len(inputs)):
            if debug:
                print("Iterating over input", currInput)
                singleDot = inputs[currInput] * self.weights[currInput]
                dotOutput += singleDot
                print("-- {}*{} = {}".format(inputs[currInput], self.weights[currInput], singleDot))
                if singleDot != 0:
                    debugPrint.append("{}*{} + ".format(inputs[currInput], self.weights[currInput]))
            else:
                dotOutput += inputs[currInput] * self.weights[currInput]

        debugPrint.append("{}) = {}".format(self.bias, dotOutput))
        if debug: print(("".join(debugPrint)))

        return dotOutput + self.bias

    def applyActivationFunction(self, dotOutput, debug = False):
        """Method applies the activation function to the input

        :param dotOutput: Float which is is the output of the applyDot method
        :type dotOutput: float
        :param debug: Boolean flag for turning on debug mode
        :type debug: bool, optional

        :return: Returns the input applied by the activation function.
        :rtype: float

        |
        """
        if debug:
            act = self.activationFunction(dotOutput)
            print("Activation Function Output = {}".format(act))
            return act
        else:
            return self.activationFunction(dotOutput)
