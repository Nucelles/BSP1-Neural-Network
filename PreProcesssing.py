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
        #labels.append(createLabel(label[0]))

        labels.append(label[0])
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