"""
This file is built to hold the activation functions that will be used in the code.
"""

import math

def identity(x):
    """
    This function returns the idenity of x, which is x
    :param x: weighted sum of a neuron
    :type x: float

    :return: The output of the activation function
    :rtype: float

    |
    """
    return x

def sigmoid(x):
    """
    This function returns the sigmoid of x
    :param x: float, weighted sum of a neuron
    :type x: float

    :return: The output of the activation function
    :rtype: float

    |
    """
    return 1 / (1 + math.exp(-x))


def binary(x):
    """
    This function performs binary function, returns 0 or 1
    :param x: weighted sum of a neuron
    :type x: float

    :return: The output of the activation function
    :rtype: float

    |
    """
    if x < 0:
        return 0
    else:
        return 1


def tanH(x):
    """
    This function returns the tanH activation function on x
    :param x: weighted sum of a neuron
    :type x: float

    :return: The output of the activation function
    :rtype: float

    |
    """
    return (math.exp(x) - math.exp(-x)) / math.exp(x)


def reLu(x):
    """
    This function returns the reLu actvation function on x
    :param x: weighted sum of a neuron
    :type x: float

    :return: The output of the activation function
    :rtype: float

    |
    """
    if x < 0:
        return 0
    else:
        return x

def leakyReLu(x, negativeSlope = 0.01):
    """
    This function performs leaky ReLu (dying ReLu) on x
    :param x: weighted sum of a neuron
    :type x: float
    :param negativeSlope: the negative slope that will be applied to x if below 0
    :type negativeSlope: float

    :return: The output of the activation function
    :rtype: float

    |
    """
    if x < 0:
        return x*negativeSlope
    else:
        return x

def softmax(x):
    """
    This function performs the softmax function on an array of inputs, returning an array of
    the probability of each score.
    :param x: A float list of the outputs of the softmax layer's neurons
    :type x: list

    :return: The output of the activation function
    :rtype: float

    |
    """

    return [p/sum(x) for p in x]


def derivitiveSigmoid(x):
    """
    This function returns the value inputed into the derivative sigmoid function.
    :param x: value that will passed nto the function
    :type x: float

    :return: The output of the derivative of the activation function
    :rtype: float

    |
    """

    return x * (1 - x)


def derivativeReLU(x):

    if x < 0:
        return 0
    elif x > 0:
        return 1
    else:
        return 0

def derivativeLeakyReLU(x, negativeSlope = 0.01):

    if x > 0:
        return 1
    else:
        return negativeSlope
