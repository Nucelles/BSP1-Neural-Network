

# The test cases need to be changed for the new network class format

import NeuralNetwork as nn
import ActivationFunctions as ac
import Layer as layer
import PreProcesssing as pre
import gzip
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle


# Preprocess the dataset into a readable format

loc = "data/"
numberOfImages = 15000

print("Pre-processing Dataset...")
print("    Number of Images = {}".format(numberOfImages))
imageDataset = pre.loadFromIDX3(loc + "train_images.gz", 28, 28, numberOfImages)
labelDataset = pre.loadFromIDX1(loc + "train_labels.gz", numberOfImages)

dataset = [[imageDataset[i], labelDataset[i]] for i in range(numberOfImages)]
random.shuffle(dataset)
trainingDataset = dataset[:3000]
print(len(trainingDataset))

print("Preprocessing Done!")


"""
Shuffle Testing
showImageFrom1D(trainingDataset[0][0])
print("1st Image Pre-shuffle Label = {}".format(trainingDataset[0][1]))

random.shuffle(trainingDataset)

showImageFrom1D(trainingDataset[0][0])
print("1st Image After-shuffle Label = {}".format(trainingDataset[0][1]))
"""


learningRate = 0.05
epochs = 5
debug = False
activation = ac.sigmoid


print("Creating Model...")
model = nn.Network()

model.add(layerToAdd=layer.inputLayer(784, activation))
model.add(layerToAdd=layer.neuronLayer(256, activation))
model.add(layerToAdd=layer.neuronLayer(128, activation))
model.add(layerToAdd=layer.outputLayer(10, activation))
print("Compiling Model...")
model.compile()

print("Training Model...")
print("    Learning Rate = {}\n"
      "    Number of Epochs = {}\n"
      "    Debugging = {}".format(learningRate, epochs, debug))

model.runNetwork(trainingDataset, learningRate, epochs, debug)
print("Training Done!")


print("Model Saved!")
model.saveNetwork()


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