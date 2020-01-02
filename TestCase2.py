import pickle
import PreProcesssing as pre
from random import shuffle

modelLocation = "model/"
dataLocation = "data/"
numberOfImages = 1000


importModel = open(modelLocation + "model_20-12-2019_10-17-09_AM.obj", "rb")
model = pickle.load(importModel)
#model.printNetwork()

imageDataset = pre.loadFromIDX3(dataLocation + "train_images.gz", 28, 28, numberOfImages)
labelDataset = pre.loadFromIDX1(dataLocation + "train_labels.gz", numberOfImages)

dataset = [[imageDataset[i], labelDataset[i]] for i in range(numberOfImages)]
shuffle(dataset)

"""
numberInEachClass = [0,0,0,0,0,0,0,0,0,0]
for i in labelDataset[:1000]:
    #print(i)
    numberInEachClass[i] += 1

print(numberInEachClass)
"""


# Feeds the first image in the shuffled dataset
pre.showImageFrom1D(dataset[0][0])
model.feedNetwork(dataset[0][0])
print(dataset[0][1])

