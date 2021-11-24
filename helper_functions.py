#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 18:11:01 2021

@author: ronan

Some basic functions lifted from Tobias' github. May be able to replace with
some numpy functions.
"""
import qutip as qt
import numpy as np
from functools import partial, reduce
import operator
import matplotlib.pyplot as plt


def prod(factors):
    return reduce(operator.mul, factors, 1)


def flatten(l):
    return [item for sublist in l for item in sublist]


#tensors operators together
def genFockOp(op, position, size, levels=2, opdim=0):
    opList = [qt.qeye(levels) for x in range(size - opdim)]
    opList[position] = op
    return qt.tensor(opList)


def natural_num_check(N, param_name):
    if type(N) is not int:
        raise Exception(f"{param_name} must be an integer.")
    if N < 0:
        raise Exception(f"{param_name} must be greater than 0.")
    return N


def pretty_subplot(axis, x_label, y_label, title, fontsize):  #formatting graphs
    axis.set_xlabel(x_label, fontsize=fontsize)
    axis.set_title(title, fontsize=fontsize + 2)
    axis.set_ylabel(y_label, fontsize=fontsize)
    axis.tick_params(labelsize=fontsize)
    axis.set_facecolor("#fffcf5")
    #axis.legend(fontsize=fontsize-2)

def pretty_graph(x_label, y_label, title, fontsize):  #formatting graphs
    figure = plt.gcf()
    figure.set_size_inches(18, 10)
    plt.xlabel(x_label, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    plt.gca().set_facecolor("#fffcf5")

def extend(nested_list):
    sizes = []
    for l in nested_list:
        sizes.append(len(l))
    pad_to_length = max(sizes)
    for l in nested_list:
        pad_with = l[-1]
        while len(l) < pad_to_length:
            l.append(pad_with)
    return np.array(nested_list)
        

