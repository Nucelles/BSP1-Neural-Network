import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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

def loadFromIDX3(location, imageHeight = 28, imageWidth = 28, numberOfImages = 1000):
    """
    This function unpacks the IDX MNIST image dataset and preprocesses them for training.
    :param location: location of the IDX file in the folder (zipped)
    :type location: str
    :param imageHeight: height of the image
    :type imageHeight: int
    :param imageWidth: width of the image
    :type imageWidth: int
    :param numberOfImages: number of images to unpack
    :type numberOfImages: int
    :return: Returns the an array dataset images
    :rtype: int array
    """
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

def loadFromIDX1(location, numberOfLabels, oneHotEncoded = True):
    """
    This function unpacks the IDX MNIST label dataset and preprocesses them for training.

    :param location: location of the IDX file in the folder (zipped)
    :type location: str
    :param numberOfLabels: number of labels to unpack
    :param numberOfLabels: int
    :param oneHotEncoded: True/False if the labels should be one-hot encoded
    :param oneHotEncoded: boolean
    :return: Returns the an array dataset labels
    :rtype: int array
    """
    datasetFile = gzip.open(location, "r")
    datasetFile.read(8)

    i = 0
    labels = []
    while i < numberOfLabels:
        label = datasetFile.read(1)
        label = np.frombuffer(label, dtype=np.uint8).astype(np.int64)
        if oneHotEncoded:
            formattedLabel = [0 for i in range(10)]
            formattedLabel[label[0]] = 1
            labels.append(formattedLabel)
        else:
            labels.append(label[0])
        i += 1

    return labels


def showImageNumpy(img):
    """
    Function prints a MNIST image using matplotlib
    :param img:
    :type img:
    """
    image = np.asarray(img).squeeze()
    plt.imshow(image)
    plt.show()


def testModel(testDataset, model):
    """
    Function used for testing the model, will print you accuracy and results at the end.
    :param testDataset: An array contain an annotated list of
    :type testDataset: int array
    :param model: A network object that will be tested
    :type model: Network
    """
    print("\nTesting Started...")
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
          "\n\nModel Results:"
          "\nAccuracy = {}%"
          "\nCorrect = {}"
          "\nIncorrect = {}"
          "\nDataset Size = {}".format(correct / testDataLength * 100, correct, testDataLength - correct, testDataLength))

    print("\nDigit Error Breakdown:")
    for i in range(len(digitError)):
        labelSeen, labelCorrect = digitSeen[i], digitSeen[i] - digitError[i]
        acc = (labelCorrect / labelSeen) * 100
        print("Label {} = {}/{}".format(i, labelCorrect, labelSeen), ", Acc = {}%\n".format(acc))
    print(digitError)

