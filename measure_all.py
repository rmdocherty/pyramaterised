#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 22:39:16 2021

@author: ronan
"""

import PQC_lib as pqc
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy

from measurement import Measurements
from math import isclose
from circuit_structures import *
from helper_functions import pretty_graph

#%%

N_REPEATS = 10

CIRCUIT = "TFIM" #options are HE, NPQC, TFIM, TFIM_modified, XXZ
N_QUBITS = 4
N_LAYERS = 4
N_SAMPLES = 100
HAMILTONIAN = "ZZ" #options are ZZ, TFIM

TRAIN = True
START = "random" #options are random, fixed
TRAIN_METHOD = "BFGS"
TRAIN_EPSILON = 1e-6
LEARNING_RATE = 0.001

SAVE = True
PLOT = True

g, h = 0, 0

#%%


def generate_circuit(circuit_type, N, p):
    circuit = pqc.PQC(N)

    if circuit_type == "NPQC":
        layers, theta_ref = NPQC_layers(p, N)
    elif circuit_type == "TFIM":
        layers = gen_TFIM_layers(p, N)
    elif circuit_type == "TFIM_modified":
        layers = gen_modified_TFIM_layers(p, N)
    elif circuit_type == "XXZ":
        layers = gen_XXZ_layers(p, N)

    for l in layers:
        circuit.add_layer(l)
    
    if HAMILTONIAN == "TFIM":
        ham = TFIM_hamiltonian(N, g, h)
        circuit.set_H(ham)

    return circuit

def measure_everything():
    circuit_expr = 0
    circuit_entanglement = 0
    circuit_magic = 0
    training_magics = []
    training_entanglement = []
    training_costs = []

    for i in range(N_REPEATS):
        circuit = generate_circuit(CIRCUIT, N_QUBITS, N_LAYERS)
        