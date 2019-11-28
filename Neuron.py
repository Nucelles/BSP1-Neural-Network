from random import random

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
