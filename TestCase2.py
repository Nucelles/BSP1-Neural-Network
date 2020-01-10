import pickle
import PreProcesssing as pre
from tqdm import tqdm
from random import shuffle

modelLocation = "model/"
dataLocation = "data/"
numberOfImages = 1000


importModel = open(modelLocation + "Model_A.obj", "rb")
model = pickle.load(importModel)
#model.printNetwork()

imageDataset = pre.loadFromIDX3(dataLocation + "train_images.gz", 28, 28, numberOfImages)
labelDataset = pre.loadFromIDX1(dataLocation + "train_labels.gz", numberOfImages)

dataset = [[imageDataset[i], labelDataset[i]] for i in range(numberOfImages)]
#shuffle(dataset)

"""
numberInEachClass = [0,0,0,0,0,0,0,0,0,0]
for i in labelDataset[:1000]:
    #print(i)
    numberInEachClass[i] += 1

print(numberInEachClass)
"""

numberOfImages = 500
imageDataset = pre.loadFromIDX3("data/test_images.gz", 28, 28, numberOfImages)
labelDataset = pre.loadFromIDX1("data/test_labels.gz", numberOfImages, False)
testDataset = [[imageDataset[i], labelDataset[i]] for i in range(numberOfImages)]

def testModel(testDataset, model):
    testDataLength = len(testDataset)
    digitError = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    digitSeen = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    correct = 0

    for testPair in tqdm(testDataset):

        digitSeen[testPair[1]] += 1
        prediction = model.feedNetwork(testPair[0])

        predictedDigit = prediction.index(max(prediction))
        correctDigit = testPair[1]

        if predictedDigit == correctDigit:
            correct += 1
        else:
            digitError[correctDigit] += 1

    print("Testing Completed!"
          "\nModel Results:"
          "\nAccuracy = {}%"
          "\nCorrect = {}"
          "\nIncorrect = {}"
          "\nDataset Size = {}".format(correct / testDataLength, correct, testDataLength - correct, testDataLength))
    print("Digit Error Breakdown:")
    for i in range(len(digitError)):
        print("Label {}: {}/{},".format(i, digitSeen[i], digitSeen[i]-digitError[i]), "Acc = {0:.0f}%".format((digitSeen[i]-digitError[i]) / digitSeen[i] * 100))

def importModel(location):
    importModel = open(location, "rb")
    model = pickle.load(importModel)

    return model


modelA = importModel(modelLocation + "Model_A.obj")
modelB = importModel(modelLocation + "Model_B.obj")
modelC = importModel(modelLocation + "Model_C.obj")
modelD = importModel(modelLocation + "Model_D.obj")

testModel(testDataset, modelA)
testModel(testDataset, modelB)
testModel(testDataset, modelC)
testModel(testDataset, modelD)