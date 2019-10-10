import math
import numpy as np

class Neuron():

    def __init__(self, layerSize):
        self.weights = np.random.rand(layerSize)
        self.bias = 0

    def runNeuron(self, inputs):
        """
        This function will run the other functions of the neurons, all the necessary functions to process the input.
        :return:
        """
        dotOutput = self.applyDot(inputs)
        output = self.applyActivationFunction(dotOutput)
        #print(dotOutput)
        #print(output)

        return output

    def applyDot(self, inputs):
        """
        This function use the dot function to on the input and weights.
        Input = [Inputs]
        :return:
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
        output = 1 / (1 + math.exp(-dotOutput))

        return output

    def adjustWeights(self):
        """
        This function will compare the output of the neuron to the correct output to find the error. The weights will
        then be adjusted depending on the size of the error. (neuron's output - actual output = error)
        """

neuron = Neuron(4)
training_input = [1, 2, 3, 4]
neuron.runNeuron(training_input)


