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

#%% HADAMARDS
"""Single Hadamard gate acting on |0> basis state should just be (sqrt(2), sqrt(2))"""
test_H = pqc.PQC(1, 1)
layer = [pqc.H(0, 1)]
test_H.set_gates(layer)
out = test_H.gen_quantum_state()

x_component = np.real(out[1][0][0])
y_component = np.real(out[0][0][0])
over_sqrt2 = 1 / np.sqrt(2)
assert isclose(x_component, over_sqrt2) #isclose() checks if 2 numbers within an epsilon of each other
print(f"Output of 1 Hadamard circuit on |0> basis is ({x_component}, {y_component}) and expected output is {over_sqrt2}")


"""Apply 2 Hadamard gates using the layer structure to test if it works."""
test_2H = pqc.PQC(1, 2)
layer = [pqc.H(0, 1)]
test_2H.set_gates(layer)
out = test_2H.gen_quantum_state()
assert out == qt.basis(2, 0) #assert throws exception if conditional not true
print("Action of 2 Hadamards on |0> is |0> again")

#%% ENTANGLING



#%% ===========EXPR AND ENT TESTS===========
idle_circuit = pqc.PQC(1, 1)
layer = [pqc.PRot(0, 1)]
idle_circuit.set_gates(layer)
idle_circuit_m = Measurements(idle_circuit)
e = idle_circuit_m.expressibility(5000, graphs=True)
print(f"Idle circuit expr is {e}, should be around 4.317")

#%%
circuit_A = pqc.PQC(1, 1)
layer = [pqc.H(0, 1), pqc.R_x(0, 1)]
circuit_A.set_initialiser(pqc.PRot)
circuit_A.set_gates(layer)
circuit_A_expr = Measurements(circuit_A)
a_e = circuit_A_expr.expressibility(100, graphs=True)
print(circuit_A)
#%%
circuit_9 = pqc.PQC(4, 1)
layer = [pqc.H(0, 4), pqc.H(1, 4), pqc.H(2, 4), pqc.H(3,4), 
         pqc.CPHASE([3,2], 4), pqc.CPHASE([2,1], 4), pqc.CPHASE([1,0], 4),
         pqc.R_x(0, 4), pqc.R_x(1, 4), pqc.R_x(2, 4), pqc.R_x(3, 4)]
#circuit_A.set_initialiser(pqc.PRot)
circuit_9.set_gates(layer)
#circuit_9.gen_quantum_state()
circuit_9_m = Measurements(circuit_9)
#%% 1min05
e = circuit_9_m.expressibility(5000, graphs=True)
print(e)
#%%
c9_ent = circuit_9_m.entanglement(5000, graphs=True)
mean, std = np.mean(c9_ent), np.std(c9_ent)
print(f"Circuit 9 entanglement is {mean} +/- {std}")

#%%
LAYERS = 3
circuit_1 = pqc.PQC(4, LAYERS)
layer = [pqc.R_x(0, 4), pqc.R_x(1, 4), pqc.R_x(2, 4), pqc.R_x(3, 4),
         pqc.R_y(0, 4), pqc.R_y(1, 4), pqc.R_y(2, 4), pqc.R_y(3, 4)]
circuit_1.set_gates(layer)

circuit_1_m = Measurements(circuit_1)
#%%
c1_expr = circuit_1_m.expressibility(5000, graphs=True)
print(f"Circuit 1 expressibility at L={LAYERS} is {c1_expr}")
#%%
c1_ent = circuit_1_m.entanglement(5000, graphs=True)
mean, std = np.mean(c1_ent), np.std(c1_ent)
print(f"Circuit 1 entanglement is {mean} +/- {std}")

#%%
circuit_2 = pqc.PQC(4, 1)
layer = [pqc.R_x(0, 4), pqc.R_x(1, 4), pqc.R_x(2, 4), pqc.R_x(3, 4),
         pqc.R_z(0, 4), pqc.R_z(1, 4), pqc.R_z(2, 4), pqc.R_z(3, 4),
         pqc.CNOT([3, 2], 4), pqc.CNOT([2, 1], 4), pqc.CNOT([1, 0], 4)]
circuit_2.set_gates(layer)

circuit_2_m = Measurements(circuit_2)
#%%
circuit_2_m.expressibility(5000, graphs=True)
#%%
c2_ent = circuit_2_m.entanglement(5000, graphs=True)
mean, std = np.mean(c2_ent), np.std(c2_ent)
print(f"Circuit 2 entanglement is {mean} +/- {std}")

c2_mw = circuit_2_m.MeyerWallach(1000)
print(f"Circuit 2 MW is {c2_mw} ")

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
#%%
circuit_11.initialise()
deriv = circuit_11.take_derivative(1)
print(deriv * qt.tensor([qt.basis(2, 0) for i in range(circuit_11._n_qubits)]))
#%%
qg_circuit = pqc.PQC(4, 1)
layer1 = [pqc.R_z(0, 4), pqc.R_x(1, 4), pqc.R_y(2, 4), pqc.R_z(3, 4), pqc.CHAIN(pqc.CNOT, 4)]
layer2 = [pqc.R_x(0, 4), pqc.R_x(1, 4), pqc.R_x(2, 4), pqc.R_y(3, 4), pqc.CHAIN(pqc.CNOT, 4)]
layer3 = [pqc.R_z(0, 4), pqc.R_x(1, 4), pqc.R_y(2, 4), pqc.R_y(3, 4), pqc.CHAIN(pqc.CNOT, 4)]
layer = layer1 + layer2 + layer3
qg_circuit.set_initialiser(pqc.sqrtH) #needs to be a sqrtH initialiser!!
qg_circuit.set_gates(layer)

qg_circuit._quantum_state = qt.Qobj(qg_circuit.initialise(random=False, angles=[3.21587011, 5.97193953, 0.90578156, 5.96054027,
       1.9592948 , 2.65983852, 5.20060878, 2.571074,
       3.45319898, 0.17315902, 4.73446249, 3.38125416]))
print(qg_circuit)
energy = qg_circuit.energy()
print(f"Energy is {energy}") #should ouput 0.46135870050914374
qg_m = Measurements(qg_circuit)
efd = qg_m.get_effective_quantum_dimension(10**-12)
print(efd) #should output 12