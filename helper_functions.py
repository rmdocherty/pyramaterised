#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 18:11:01 2021

@author: ronan

Some basic functions lifted from Tobias' github. May be able to replace with
some numpy functions.
"""
import qutip as qt
from functools import partial, reduce
import operator


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