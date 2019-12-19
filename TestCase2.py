import pickle
from TestCases import loadFromIDX3, loadFromIDX1

modelLocation = "model/"
dataLocation = "data/"

importModel = open(modelLocation + "model_16-12-2019_12-17-29_PM.obj", "rb")
model = pickle.load(importModel)
#model.printNetwork()

imageDataset = loadFromIDX3(dataLocation + "train_images.gz", 28, 28, 10)
labelDataset = loadFromIDX1(dataLocation + "train_labels.gz", 10)

tc.showImageFrom1D(imageDataset[5])
#model.feedNetwork(imageDataset[5])
print(labelDataset[5])
