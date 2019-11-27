"""
This file is built to hold the activation functions that will be used in the code.
"""

import math

def identity(x):
    """
    This function returns the idenity of x, which is x
    :param x: float, weighted sum of a neuron
    :return:
    """
    return x

def sigmoid(x):
    """
    This function returns the sigmoid of x
    :param x: float, weighted sum of a neuron
    :return:
    """
    return 1 / (1 + math.exp(-x))


def binary(x):
    """
    This function performs binary function, returns 0 or 1
    :param x: float, weighted sum of a neuron
    :return:
    """
    if x < 0:
        return 0
    else:
        return 1


def tanH(x):
    """
    This function returns the tanH activation function on x
    :param x: float, weighted sum of a neuron
    :return:
    """
    return (math.exp(x) - math.exp(-x)) / math.exp(x)


def reLu(x):
    """
    This function returns the reLu actvation function on x
    :param x: float, weighted sum of a neuron
    :return:
    """
    if x < 0:
        return 0
    else:
        return x

def leakyReLu(x, negativeSlope = 0.01):
    """
    This function performs leaky ReLu (dying ReLu) on x
    :param x: float, weighted sum of a neuron
    :param negativeSlope: float, the negative slope that will be applied to x if below 0
    :return:
    """
    if x < 0:
        return x*negativeSlope
    else:
        return x

def softmax(x):
    """
    This function performs the softmax function on an array of inputs, returning an array of
    the probability of each score.
    :param x: [floats], the list of incoming neurons
    :return:
    """

    return [p/sum(x) for p in x]


def derivitiveSigmoid(x):
    """
    This function returns the value inputed into the derivative sigmoid function.
    :param x: float, value that will passed nto the function
    :return:
    """

    return x * (1 - x)