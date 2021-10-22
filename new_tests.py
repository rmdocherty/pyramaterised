#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 13:40:05 2021

@author: ronan
"""

from measurement import Measurements
from helper_functions import pretty_subplot
from math import isclose
import matplotlib.pyplot as plt
import qutip as qt
import numpy as np
import random
import PQC_lib as pqc

random.seed(1) #for reproducibility

#%% ===========BASIC CIRCUIT TESTS===========
test_H = pqc.PQC(1,1)
layer = [pqc.H(0, 1)]
test_H.set_gates(layer)
out = test_H.gen_quantum_state()

x_component = np.real(out[1][0][0])
y_component = np.real(out[1][0][0])
over_sqrt2 = 1 / np.sqrt(2)
assert (isclose(x_component, over_sqrt2))
print(f"Output of 1 Hadamard circuit on |0> basis is ({x_component}, {y_component}) and expected output is {over_sqrt2}")

test_2H = pqc.PQC(1,2)
layer = [pqc.H(0, 1)]
test_2H.set_gates(layer)
out = test_2H.gen_quantum_state()
assert (out == qt.basis(2,0))
print("Action of 2 Hadamards on |0> is |0> again")
#%% ===========EXPR AND ENT TESTS===========
idle_circuit = pqc.PQC(1, 1)
layer = [pqc.PRot(0, 1)]
idle_circuit.set_gates(layer)
idle_circuit_m = Measurements(idle_circuit)
idle_circuit_m.expressibility(5000, graphs=True)

#%%
circuit_A = pqc.PQC(1, 1)
layer = [pqc.H(0, 1), pqc.R_x(0, 1)]
circuit_A.set_initialiser(pqc.PRot)
circuit_A.set_gates(layer)
circuit_A_expr = Measurements(circuit_A)
circuit_A_expr.expressibility(5000, graphs=True)
#%%
original_circuit = pqc.PQC(4,3)
original_circuit.set_initialiser(pqc.H)
rotate_layer = []
options = [pqc.R_x, pqc.R_y, pqc.R_z]
for i in range(4):
    R = random.choice(options)
    rotate_layer.append(R(i, 4))
print(rotate_layer)
layer = rotate_layer + [pqc.Chain(pqc.CNOT, 4)]
original_circuit.set_gates(layer)
original_circuit.gen_quantum_state()
print(original_circuit)
e = original_circuit.energy()
print(e)

#%%
circuit_9 = pqc.PQC(4, 5)
layer = [pqc.H(0, 4), pqc.H(1, 4), pqc.H(2, 4), pqc.H(3,4), 
         pqc.CPHASE([0,1], 4), pqc.CPHASE([1,2], 4), pqc.CPHASE([2,3], 4),
         pqc.R_x(0, 4), pqc.R_x(1, 4), pqc.R_x(2, 4), pqc.R_x(3, 4)]
#circuit_A.set_initialiser(pqc.PRot)
circuit_9.set_gates(layer)
#circuit_9.gen_quantum_state()
circuit_9_m = Measurements(circuit_9)
#%%
circuit_9_m.expressibility(5000, graphs=True)
#%%
c9_ent = circuit_9_m.entanglement(5000, graphs=True)
mean, std = np.mean(c9_ent), np.std(c9_ent)
print(f"Circuit 9 entanglement is {mean} +/- {std}")

#%%
circuit_1 = pqc.PQC(4, 1)
layer = [pqc.R_x(0, 4), pqc.R_x(1, 4), pqc.R_x(2, 4), pqc.R_x(3, 4),
         pqc.R_y(0, 4), pqc.R_y(1, 4), pqc.R_y(2, 4), pqc.R_y(3, 4)]
circuit_1.set_gates(layer)

circuit_1_m = Measurements(circuit_1)
#%%
circuit_1_m.expressibility(5000, graphs=True)
#%%
c1_ent = circuit_1_m.entanglement(5000, graphs=True)
mean, std = np.mean(c1_ent), np.std(c1_ent)
print(f"Circuit 1 entanglement is {mean} +/- {std}")

#%%
circuit_2 = pqc.PQC(4, 1)
layer = [pqc.R_x(0, 4), pqc.R_x(1, 4), pqc.R_x(2, 4), pqc.R_x(3, 4),
         pqc.R_z(0, 4), pqc.R_z(1, 4), pqc.R_z(2, 4), pqc.R_z(3, 4),
         pqc.Chain(pqc.CNOT, 4)]
circuit_2.set_gates(layer)

circuit_2_m = Measurements(circuit_2)
#%%
circuit_2_m.expressibility(5000, graphs=True)
#%%
c2_ent = circuit_2_m.entanglement(10000, graphs=True)
mean, std = np.mean(c2_ent), np.std(c2_ent)
print(f"Circuit 2 entanglement is {mean} +/- {std}")

#%%

circuit_11 = pqc.PQC(4, 1)
layer = [pqc.R_y(0, 4), pqc.R_y(1, 4), pqc.R_y(2, 4), pqc.R_y(3, 4),
         pqc.R_z(0, 4), pqc.R_z(1, 4), pqc.R_z(2, 4), pqc.R_z(3, 4),
                 pqc.CNOT([1, 0], 4), pqc.CNOT([3, 2], 4),
                     pqc.R_y(1, 4), pqc.R_y(2, 4),
                     pqc.R_z(1, 4), pqc.R_z(2, 4),
                         pqc.CNOT([2, 1], 4)]
circuit_11.set_gates(layer)
circuit_11_m = Measurements(circuit_11)
#%%
circuit_11_m.expressibility(5000, graphs=True)
#%%
c11_ent = circuit_11_m.entanglement(5000, graphs=True)
mean, std = np.mean(c11_ent), np.std(c11_ent)
print(f"Circuit 11 entanglement is {mean} +/- {std}")
