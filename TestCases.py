

# The test cases need to be changed for the new network class format

import NeuralNetwork as nn
import ActivationFunctions as ac
import Layer as layer
import PreProcesssing as pre
import random

# Preprocess the dataset into a readable format

loc = "data/"
numberOfImages = 30000

print("Pre-processing Dataset...")
print("    Number of Images = {}".format(numberOfImages))
imageDataset = pre.loadFromIDX3(loc + "train_images.gz", 28, 28, numberOfImages)
labelDataset = pre.loadFromIDX1(loc + "train_labels.gz", numberOfImages)

dataset = [[imageDataset[i], labelDataset[i]] for i in range(numberOfImages)]
random.shuffle(dataset)
trainingDataset_Large = dataset[:4000]

print("Preprocessing Done!")


activation = ac.sigmoid


print("Creating Models...")
model = nn.Network()

model.add(layerToAdd=layer.inputLayer(784, activation))
model.add(layerToAdd=layer.neuronLayer(256, activation))
model.add(layerToAdd=layer.neuronLayer(128, activation))
model.add(layerToAdd=layer.outputLayer(10, activation))
model.compile()

model_A = model
model_B = model
model_C = model
model_D = model

model_A.runNetwork(trainingDataset_Large, 0.05, 5, False)
model_B.runNetwork(trainingDataset_Large, 0.005, 5, False)
model_C.runNetwork(trainingDataset_Large, 0.05, 10, False)
model_D.runNetwork(trainingDataset_Large, 0.005, 10, False)

model_A.saveNetwork("Model_A")
model_B.saveNetwork("Model_B")
model_C.saveNetwork("Model_C")
model_D.saveNetwork("Model_D")
print("DONE!!!!!")


"""
dataset = [[[0, 1, 1 , 0],[0, 1]], [[0, 0, 0, 0],[1, 0]], [[1, 1, 0, 0],[0, 1]], [[0, 0, 1, 1],[0, 1]], [[0, 1, 0, 1],[1, 0]], [[1, 0 , 1, 0],[0, 1]]]

model = nn.Network()

model.add(layerToAdd=layer.inputLayer(4))
model.add(layerToAdd=layer.neuronLayer(4, ac.sigmoid))
model.add(layerToAdd=layer.neuronLayer(4, ac.sigmoid))
model.add(layerToAdd=layer.outputLayer(2, ac.sigmoid))
model.compile()
model.printNetwork()

model.runNetwork(dataset, 0.05, 10, False)


#model.printNetwork()



model = unPickleModel("model/model_B.obj")

imageDataset = loadFromIDX3(loc + "train_images.gz", 28, 28, 2)
labelDataset = loadFromIDX1(loc + "train_labels.gz", 2)

model.feedNetwork(imageDataset[1])
print(labelDataset[1])
"""