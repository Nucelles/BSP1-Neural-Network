

# The test cases need to be changed for the new network class format

import NeuralNetwork as nn
import ActivationFunctions as ac
import Layer as layer
import gzip
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

def unPickleModel(location):
    """Unpickles a model in obj format and returns it.

    :param location: location of the model in the directory
    :type location: str
    :return: Returns the unpickled object as a Network class object
    :rtype: Network
    """
    importModel = open(location, "rb")
    model = pickle.load(importModel)

    return model

def loadFromIDX3(location, imageHeight, imageWidth, numberOfImages):
    datasetFile = gzip.open(location, "r")
    datasetFile.read(16)
    unitLength = imageHeight*imageWidth

    data = []

    i = 0
    while i < numberOfImages:
        currentImage = datasetFile.read(unitLength)
        currentImage = np.frombuffer(currentImage, dtype=np.uint8).astype(np.float32)
        currentImage = currentImage.reshape(1, imageHeight, imageWidth, 1)
        #print(currentImage)
        image = []
        for row in currentImage:
            for value in row:
                for singleValue in value:
                    for flo in singleValue:
                        image.append(flo/255)

        data.append(image)
        i += 1

    return data

def loadFromIDX1(location, numberOfLabels):
    datasetFile = gzip.open(location, "r")
    datasetFile.read(8)

    i = 0
    labels = []
    while i < numberOfLabels:
        label = datasetFile.read(1)
        label = np.frombuffer(label, dtype=np.uint8).astype(np.int64)
        labels.append(createLabel(label[0]))
        i += 1

    return labels

def showImageFrom1D(img):
    cur = 1
    wholeImage = []
    row = []
    for i in img:

        if cur == 28:
            cur = 1
            wholeImage.append(row)
            #print(row)
            row = []
        else:
            cur += 1
            #print(i)
            row.append(i)
            #print(row)

    #print(wholeImage)

    showImageNumpy(wholeImage)



def showImageNumpy(img):
    image = np.asarray(img).squeeze()
    plt.imshow(image)
    plt.show()

def createLabel(label):
    formattedLabel = [0 for i in range(10)]
    formattedLabel[label] = 1

    return formattedLabel


# Preprocess the dataset into a readable format

loc = "data/"
numberOfImages = 1500

print("Preprocessing Dataset...")
print("    Number of Images = {}".format(numberOfImages))
imageDataset = loadFromIDX3(loc + "train_images.gz", 28, 28, numberOfImages)
labelDataset = loadFromIDX1(loc + "train_labels.gz", numberOfImages)

trainingDataset = [[imageDataset[i], labelDataset[i]] for i in range(numberOfImages)]
print("Preprocessing Done!")


"""
Shuffle Testing
showImageFrom1D(trainingDataset[0][0])
print("1st Image Pre-shuffle Label = {}".format(trainingDataset[0][1]))

random.shuffle(trainingDataset)

showImageFrom1D(trainingDataset[0][0])
print("1st Image After-shuffle Label = {}".format(trainingDataset[0][1]))
"""


learningRate = 0.0005
epochs = 10
debug = False

print("Creating Model...")
model = nn.Network()

model.add(layerToAdd=layer.inputLayer(784, ac.leakyReLu))
model.add(layerToAdd=layer.neuronLayer(512, ac.leakyReLu))
model.add(layerToAdd=layer.outputLayer(10, ac.leakyReLu))
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