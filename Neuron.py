import math
import numpy as np

class Neuron():

    def __init__(self, numberOfInputs, activationFunction):
        self.activationFunction = activationFunction
        self.weights = np.random.rand(numberOfInputs)
        self.weightsPreset = []
        self.bias = 0
        self.output = 0

    def runNeuron(self, inputs):
        """
        This function will run the other functions of the neurons, all the necessary functions to process the input.
        :return:
        """
        dotOutput = self.applyDot(inputs)
        output = self.applyActivationFunction(dotOutput)
        # print(dotOutput)
        # print(output)

        return output

    def applyDot(self, inputs):
        """
        This function use the dot function to on the input and weights.
        Input = [Inputs]
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
        if self.activationFunction == "Sigmoid":
            self.output = 1 / (1 + math.exp(-dotOutput))
        elif self.activationFunction == "ReLu":
            # ReLu activation function
            if dotOutput < 0:
                self.output = 0
            else:
                self.output = dotOutput

        elif self.activationFunction == "Leaky ReLu":
            # Leaky ReLu activation function
            if dotOutput < 0:
                self.output = 0
            else:
                self.output = 0.01*dotOutput
        else:
            print("Activation function not recognized, using default Sigmoid")
            self.output = 1 / (1 + math.exp(-dotOutput))

        return self.output



neuron = Neuron(4, "Sigmoid")
training_input = [1, 2, 3, 4]
neuron.runNeuron(training_input)


