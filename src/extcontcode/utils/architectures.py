from torch import nn as nn


def gen_fcnet(neurons, act=nn.ELU()):
    """
    Generate fully connected neural nets given the specified layer widths
    """
    layers = []
    for i in range(len(neurons) - 1):
        layers.append(nn.Linear(neurons[i], neurons[i + 1]))
        layers.append(act)
    return nn.Sequential(*layers[:-1])
