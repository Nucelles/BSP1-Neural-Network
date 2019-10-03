import numpy as np
import math

class Neuron():

    def __init__(self):
        np.random.seed(2019)
        self.weight = 2  # np.random.rand(1)
        self.inputData = []
        self.bias = 0
        self.currentSum = 0
        self.output = 0

    def recieveInputs(self, inputs):
        """
        This function will receive the inputs of the neuron and assign their respective weights and then add to the
        object's input array.
        Input = [Inputs], Output = [Weighted Inputs]
        """
        for input in inputs:
            self.inputData.append(input*self.weight)
            print(self.inputData)

    def getSum(self, applyBias):
        """
        This function will get the sum of all the weighted inputs.
        Input = ApplyBias = Boolean, Output = SumOfInputs
        Add this as the DOT product NOT USING NUMPY
        """
        for inputs in self.inputData:
            self.currentSum += inputs

        if applyBias:
            self.currentSum
        self.inputData = []

    def applyActivationFunction(self):
        """
        This function will apply the sigmoid function to the sum of the inputs.
        Input = SumOfInputs, Output = finalOutput
        """
        output = 1 / (1 + math.exp(-self.currentSum))
        self.output = output
        return output

    def adjustWeights(self):
        """
        This function will compare the output of the neuron to the correct output to find the error. The weights will
        then be adjusted depending on the size of the error. (neuron's output - actual output = error)
        """

neuron = Neuron()
training_input = [0.1231, 0.3295, 0.12985732, 0.32875, 0.7923]
neuron.recieveInputs(training_input)
print(neuron.weight)
print(neuron.inputData)
neuron.getSum(False)
print(neuron.applyActivationFunction())

