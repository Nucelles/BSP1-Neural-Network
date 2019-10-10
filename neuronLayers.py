import Neuron

class neuronLayer():

    def __init__(self, layers):
        #self.connection = connection
        self.layers = layers
        self.layerList = []
        self.layerOutput = []

        for i in range(self.layers):
            self.layerList.append(Neuron.Neuron(layers))

    def runLayer(self, previousLayerOutput):
        self.layerOutput = []
        for neuron in self.layerList:
            self.layerOutput.append(neuron.runNeuron(previousLayerOutput))
            # self.layerOutput.append(neuron.runNeuron(self.connection.layerOutput))


trainingInput = [0.27, 0.3, 0.75, 0.5]

layer1 = neuronLayer(5)
layer1.runLayer(trainingInput)
print(layer1.layerOutput)
