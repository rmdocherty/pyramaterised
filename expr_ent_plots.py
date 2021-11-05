#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 15:17:19 2021

@author: ronan
"""

#%% IMPORTS
from measurement import Measurements
from helper_functions import pretty_graph
import matplotlib.pyplot as plt
import qutip as qt
import numpy as np
import random
import PQC_lib as pqc

random.seed(1) #for reproducibility
SAMPLE_N = 124 #this produces 10011 unique state pairs with itertools.combination
LOAD = False
SAVE = False
VERBOSE = True

#%% CIRCUITS

c1_l = [pqc.R_x(0, 4), pqc.R_x(1, 4), pqc.R_x(2, 4), pqc.R_x(3, 4),
         pqc.R_y(0, 4), pqc.R_y(1, 4), pqc.R_y(2, 4), pqc.R_y(3, 4)]
c2_l = [pqc.R_x(0, 4), pqc.R_x(1, 4), pqc.R_x(2, 4), pqc.R_x(3, 4),
         pqc.R_z(0, 4), pqc.R_z(1, 4), pqc.R_z(2, 4), pqc.R_z(3, 4),
         pqc.CHAIN(pqc.CNOT, 4)]
c9_l = [pqc.H(0, 4), pqc.H(1, 4), pqc.H(2, 4), pqc.H(3,4), 
         pqc.CHAIN(pqc.CPHASE, 4),
         pqc.R_x(0, 4), pqc.R_x(1, 4), pqc.R_x(2, 4), pqc.R_x(3, 4)]
c11_l = [pqc.R_y(0, 4), pqc.R_y(1, 4), pqc.R_y(2, 4), pqc.R_y(3, 4),
         pqc.R_z(0, 4), pqc.R_z(1, 4), pqc.R_z(2, 4), pqc.R_z(3, 4),
                 pqc.CNOT([1, 0], 4), pqc.CNOT([3, 2], 4),
                     pqc.R_y(1, 4), pqc.R_y(2, 4),
                     pqc.R_z(1, 4), pqc.R_z(2, 4),
                         pqc.CNOT([2, 1], 4)]
c12_l = [pqc.R_y(0, 4), pqc.R_y(1, 4), pqc.R_y(2, 4), pqc.R_y(3, 4),
         pqc.R_z(0, 4), pqc.R_z(1, 4), pqc.R_z(2, 4), pqc.R_z(3, 4),
                 pqc.CPHASE([1, 0], 4), pqc.CPHASE([3, 2], 4),
                     pqc.R_y(1, 4), pqc.R_y(2, 4),
                     pqc.R_z(1, 4), pqc.R_z(2, 4),
                         pqc.CPHASE([2, 1], 4)]

circuit_layers = [c1_l, c2_l, c9_l, c11_l, c12_l]
circuit_names = ["1", "2", "9", "11", "12"]
#%% DATA COLLECTION

if LOAD is False:
    circuit_data = np.zeros(shape=(len(circuit_layers), 5, 3))
    for count, layers in enumerate(circuit_layers):
        for L in range(1, 6):
            print(f"Computing layer {L} of Circuit {circuit_names[count]}")
            circuit = pqc.PQC(4, L)
            circuit.set_gates(layers)
            m = Measurements(circuit)
            data = m.efficient_measurements(SAMPLE_N)
            expr = data['Expr']
            ent, std = data['Ent'][0], data['Ent'][1]
            circuit_data[count, L - 1, 0] = expr
            circuit_data[count, L - 1, 1] = ent
            circuit_data[count, L - 1, 2] = std
            if VERBOSE is True:
                print(f"Expr is {expr}, Ent is {ent} +/- {std} \n")
else:
    circuit_data = np.load("expr_ent_test_plots")

if SAVE is True:
    np.save("expr_ent_test_plots", circuit_data)

#%% PLOTTING
layer_icons = ['o', 's', '*', 'd', 'p']
colors = ["red", "blue", "black", "green", "purple"]
plt.figure("Expressibility of circuits")
for circuit in range(len(circuit_layers)):
    for L in range(5):
        plt.plot(circuit_names[circuit], circuit_data[circuit, L, 0],
                 label=f"L={L+1}", marker=layer_icons[L], ms=13, color=colors[L])
plt.yscale("log")
#plt.ylim(0.0015, 0.9)
#plt.legend(fontsize=18)
pretty_graph("Circuit ID", "Expr, D_KL", "Expressibility vs Circuit", 20)
plt.grid()

plt.figure("Entanglement")
for circuit in range(len(circuit_layers)):
    for L in range(5):
        plt.plot(circuit_names[circuit], circuit_data[circuit, L, 1],
                 label=f"L={L+1}", marker=layer_icons[L], ms=13, color=colors[L])
pretty_graph("Circuit ID", "Entaglement, Q", "Entanglement vs Circuit", 20)
#plt.legend(fontsize=18)
plt.grid()

