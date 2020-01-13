import NeuralNetwork as nn
import Layer as layer
import ActivationFunctions as ac
import PreProcesssing as pre

# Initialize the Network object
modelTemplate = nn.Network()

print("Preparing Model Template...")
# Add each layer of the network to the object
modelTemplate.add(layerToAdd=layer.inputLayer(784))
modelTemplate.add(layerToAdd=layer.hiddenLayer(392, ac.sigmoid))
modelTemplate.add(layerToAdd=layer.hiddenLayer(196, ac.sigmoid))
modelTemplate.add(layerToAdd=layer.outputLayer(10, ac.sigmoid))

print("Compiling Model Template...")
# Compile the object, connecting layers and initalizing weights
modelTemplate.compile()

# Create four identical models equivalent to the modelTemplate object
modelA = modelTemplate
modelB = modelTemplate
modelC = modelTemplate
modelD = modelTemplate
print("Models Prepared...")

# Preprocess the TRAINING image and label dataset
print("Preparing TRAINING Dataset...")
numberOfTrainingImages = 4000
imageDataset = pre.loadFromIDX3("data/train_images.gz", 28, 28, numberOfTrainingImages)
labelDataset = pre.loadFromIDX1("data/train_labels.gz", numberOfTrainingImages)
trainingDataset = [[imageDataset[i], labelDataset[i]] for i in range(numberOfTrainingImages)]
print("TRAINING Dataset Prepared!")

# Training each dataset
print("Training Model A...")
modelA.runNetwork(trainingDataset, 0.05, 5, modelName="Model_A")
print("Training Complete!")

print("Training Model C...")
modelB.runNetwork(trainingDataset, 0.005, 5, modelName="Model_B")
print("Training Complete!")

print("Training Model D...")
modelC.runNetwork(trainingDataset, 0.05, 10, modelName="Model_C")
print("Training Complete!")

print("Training Model D...")
modelD.runNetwork(trainingDataset, 0.005, 10, modelName="Model_D")
print("Training Complete!")

# Preprocess the TEST image and label dataset
numberOfTestImages = 10000
print("Preparing TEST Dataset...")
imageDataset = pre.loadFromIDX3("data/test_images.gz", 28, 28, numberOfTestImages)
labelDataset = pre.loadFromIDX1("data/test_labels.gz", numberOfTestImages, False)
testDataset = [[imageDataset[i], labelDataset[i]] for i in range(numberOfTestImages)]
print("TEST Dataset Prepared!")

# Location of the Saved Models
location = "model/"

# Here we unpickle the models and test them
print("MODEL A")
modelA = pre.unPickleModel(location + "Model_A.obj")
pre.testModel(testDataset, modelA)

print("MODEL B")
modelB = pre.unPickleModel(location + "Model_B.obj")
pre.testModel(testDataset, modelB)

print("MODEL C")
modelC = pre.unPickleModel(location + "Model_C.obj")
pre.testModel(testDataset, modelC)

print("MODEL D")
modelD = pre.unPickleModel(location + "Model_D.obj")
pre.testModel(testDataset, modelD)
