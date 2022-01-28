#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 13:10:03 2022

@author: ronan
"""

import PQC_lib as pqc
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy

from measurement import Measurements
import circuit_structures as cs
from helper_functions import pretty_subplot, extend

clifford_angles = np.arange(0, 4) * (np.pi / 2)

N = 4
L = 3
layers = cs.generic_HE(L, N)
circuit = pqc.PQC(N)
for l in layers[1:]:
    circuit.add_layer(l)

n_params = circuit.n_params
initial_clifford_angles = [random.choice(clifford_angles) for i in range(n_params)]

circuit.set_params(initial_clifford_angles)


