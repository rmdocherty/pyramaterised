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
test_H = pqc.PQC(1)
layer = [pqc.H(0, 1)]
test_H.add_layer(layer)
out = test_H.gen_quantum_state()

x_component = np.real(out[1][0][0])
y_component = np.real(out[0][0][0])
over_sqrt2 = 1 / np.sqrt(2)
assert isclose(x_component, over_sqrt2) #isclose() checks if 2 numbers within an epsilon of each other
print(f"Output of 1 Hadamard circuit on |0> basis is ({x_component}, {y_component}) and expected output is {over_sqrt2}")


"""Apply 2 Hadamard gates using the layer structure to test if it works."""
test_2H = pqc.PQC(1)
layer = [pqc.H(0, 1)]
test_2H.add_layer(layer, n=2)
out = test_2H.gen_quantum_state()
assert out == qt.basis(2, 0) #assert throws exception if conditional not true
print("Action of 2 Hadamards on |0> is |0> again")

#%% ENTANGLING


#%% ===========EXPR AND ENT TESTS===========
"""Tests based on Expr baselines from Fig 1 arXiv:1905.10876v1"""
idle_circuit = pqc.PQC(1)
layer = [pqc.PRot(0, 1)]
idle_circuit.add_layer(layer)
idle_circuit_m = Measurements(idle_circuit)
e = idle_circuit_m.expressibility(5000, graphs=True)
print(f"Idle circuit expr is {e}, should be around 4.317")

#%%
circuit_A = pqc.PQC(1)
layer = [pqc.H(0, 1), pqc.R_x(0, 1)]
circuit_A.add_layer(layer)
circuit_A_expr = Measurements(circuit_A)
a_e = circuit_A_expr.expressibility(100, graphs=True)
print(f"Circuit_A expr is {a_e}, should be around 0.2")
#%%
N = 4
circuit_9 = pqc.PQC(4)
layer = [pqc.H(0, 4), pqc.H(1, 4), pqc.H(2, 4), pqc.H(3,4), 
         pqc.CHAIN(pqc.CPHASE, 4),
         pqc.R_x(0, 4), pqc.R_x(1, 4), pqc.R_x(2, 4), pqc.R_x(3, 4)]
#circuit_A.set_initialiser(pqc.PRot)
circuit_9.add_layer(layer, N)
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
circuit_1 = pqc.PQC(4)
layer = [pqc.R_x(0, 4), pqc.R_x(1, 4), pqc.R_x(2, 4), pqc.R_x(3, 4),
         pqc.R_y(0, 4), pqc.R_y(1, 4), pqc.R_y(2, 4), pqc.R_y(3, 4)]
circuit_1.add_layer(layer, LAYERS)

circuit_1_m = Measurements(circuit_1)
#%%
c1_expr = circuit_1_m.expressibility(5000, graphs=True)
print(f"Circuit 1 expressibility at L={LAYERS} is {c1_expr}")
#%%
c1_ent = circuit_1_m.entanglement(5000, graphs=True)
mean, std = np.mean(c1_ent), np.std(c1_ent)
print(f"Circuit 1 entanglement is {mean} +/- {std}")

#%%
N = 4
circuit_2 = pqc.PQC(N)
layer = [pqc.R_x(0, N), pqc.R_x(1, N), pqc.R_x(2, N), pqc.R_x(3, N),
         pqc.R_z(0, N), pqc.R_z(1, N), pqc.R_z(2, N), pqc.R_z(3, N),
         pqc.CNOT([3, 2], N), pqc.CNOT([2, 1], N), pqc.CNOT([1, 0], N)]
circuit_2.add_layer(layer, n=3)

circuit_2_m = Measurements(circuit_2)
#%%
circuit_2_m.get_effective_quantum_dimension(10)
#%%
circuit_2_m.expressibility(5000, graphs=True)
#%%
c2_ent = circuit_2_m.entanglement(100, graphs=True)
mean, std = np.mean(c2_ent), np.std(c2_ent)
print(f"Circuit 2 entanglement is {mean} +/- {std}")

c2_mw = circuit_2_m.meyer_wallach(100)
print(f"Circuit 2 MW is {c2_mw} ")

#%%

circuit_11 = pqc.PQC(4)
layer = [pqc.R_y(0, 4), pqc.R_y(1, 4), pqc.R_y(2, 4), pqc.R_y(3, 4),
         pqc.R_z(0, 4), pqc.R_z(1, 4), pqc.R_z(2, 4), pqc.R_z(3, 4),
                 pqc.CNOT([1, 0], 4), pqc.CNOT([3, 2], 4),
                     pqc.R_y(1, 4), pqc.R_y(2, 4),
                     pqc.R_z(1, 4), pqc.R_z(2, 4),
                         pqc.CNOT([2, 1], 4)]
circuit_11.add_layer(layer, 3)
circuit_11_m = Measurements(circuit_11)
#%%
out = circuit_11_m.efficient_measurements(104)
print(out['Expr'])

#%%
"""Tests based on arXiv:2102.01659v1 github"""
qg_circuit = pqc.PQC(4)
init_layer = [pqc.sqrtH(i, 4) for i in range(4)]
layer1 = [pqc.R_z(0, 4), pqc.R_x(1, 4), pqc.R_y(2, 4), pqc.R_z(3, 4), pqc.CHAIN(pqc.CNOT, 4)]
layer2 = [pqc.R_x(0, 4), pqc.R_x(1, 4), pqc.R_x(2, 4), pqc.R_y(3, 4), pqc.CHAIN(pqc.CNOT, 4)]
layer3 = [pqc.R_z(0, 4), pqc.R_x(1, 4), pqc.R_y(2, 4), pqc.R_y(3, 4), pqc.CHAIN(pqc.CNOT, 4)]
#layer = layer1 + layer2 + layer3
#qg_circuit.set_initialiser(pqc.sqrtH) #needs to be a sqrtH initialiser!!
#qg_circuit.set_gates(layer)

qg_circuit.add_layer(init_layer)
qg_circuit.add_layer(layer1)
qg_circuit.add_layer(layer2)
qg_circuit.add_layer(layer3)
#qg_circuit.add_layer(layer1, 100)

qg_circuit.gen_quantum_state()

qg_circuit._quantum_state = qt.Qobj(qg_circuit.run(
        angles=[
        3.21587011, 5.97193953, 0.90578156, 5.96054027,
        1.9592948 , 2.65983852, 5.20060878, 2.571074,
        3.45319898, 0.17315902, 4.73446249, 3.38125416]))
print(qg_circuit)
energy = qg_circuit.energy()
#%%
print(f"Energy is {energy}, should be 0.46135870050914374")
qg_m = Measurements(qg_circuit)
efd = qg_m.get_effective_quantum_dimension(10**-12)
print(f"Effective quantum dimension is {efd}, should be 12")
new_measure = qg_m.new_measure()
print(f"New measure is {new_measure}")
out = qg_m.efficient_measurements(500)
entropy = out['Magic']
print(f"Magic is {entropy[0]} +/- {entropy[1]}")
#%%
[print(g._theta) for g in qg_circuit.gates if g._is_param]
#print(qg_circuit.get_params())
#%%
minimun = qg_m.train(epsilon=0.0001, rate=0.0001)
print(f"Minimum ground state energy is {minimun}")
#%%
class Bell:
    def __init__(self):
        self._n_qubits = 2
        self._quantum_state = qt.states.bell_state('11')
    def gen_quantum_state(self):
        return qt.states.bell_state('11')

bell_m = Measurements(Bell())
e = bell_m.entropy_of_magic()
print(f"Reyni Entropy of Magic is {e}, should be 0 for stabiliser state")

#%%


def gen_shift_list(p, N):
    A = [i for i in range(N // 2)]
    s = 1
    shift_list = np.zeros(2**(N//2), dtype=np.int32) #we have at most 2^(N/2) layers
    count = 1
    while A != []:
        r = A.pop(0) #get first elem out of A
        shift_list[s - 1] = r #a_s
        qs = [i for i in range(1, s)] #count up from 1 to s-1
        for q in qs:
            shift_list[s + q - 1] = shift_list[q - 1] #a_s+q = a_q
        s = 2 * s
    return shift_list
        

def NPQC_layers(p, N):
    initial_layer = [pqc.R_y(i, N) for i in range(N)] + [pqc.R_z(i, N) for i in range(N)]
    angles = [np.pi/2 for i in range(N)] + [0 for i in range(N)]
    layers = [initial_layer]
    shift_list = gen_shift_list(p, N)
    for i in range(0, p-1):
        p_layer = []
        a_l = shift_list[i]
        fixed_rots, cphases = [], []
        #U_ent layer - not paramterised and shouldn't be counted as such!
        for k in range(1, 1 + N // 2):
            q_on = 2 * k - 2
            rotation = pqc.fixed_R_y(q_on, N)
            fixed_rots.append(rotation)
            U_ent = pqc.CPHASE([q_on, ((q_on + 1) + 2 * a_l) % N], N)
            cphases.append(U_ent)
        p_layer = fixed_rots + cphases #need fixed r_y to come before c_phase

        #rotation layer - r_y then r_z on each kth qubit
        for k in range(1, N // 2 + 1):
            q_on = 2 * k - 2
            p_layer = p_layer + [pqc.R_y(q_on, N), pqc.R_z(q_on, N)]
            angles.append(np.pi/2)
            angles.append(0)
        layers.append(p_layer)
    return layers, angles

def check_iden(A):
    M = len(A)
    diag_entries = []
    non_zero_off_diags = []
    for i, row in enumerate(A):
        for j, column in enumerate(row):
            if i == j:
                diag_entries.append(column)
            elif column != 0 and i != j:
                non_zero_off_diags.append(column)
    print(f"Diagonals are: {diag_entries}")
    if len(non_zero_off_diags) == 0:
        print("No nonzero off diagonals!")
    elif len(non_zero_off_diags) > 0:
        print("There are nonzero off diagonals!")
        print(non_zero_off_diags)


N = 8
P = 6 #works iff P < N - is this expected behaviour?
print(N, P)
layers, theta_ref = NPQC_layers(P, N)
NPQC = pqc.PQC(N)
for l in layers:
    NPQC.add_layer(l)
NPQC._quantum_state = qt.Qobj(NPQC.run(angles=theta_ref))
print(NPQC, theta_ref)

NPQC_m = Measurements(NPQC)
QFI = np.array(NPQC_m._get_QFI())
masked = np.where(QFI < 10**-12, 0, QFI)
print(f"QFI of NPQC with N={N}, layers={P} is: \n {masked}")
check_iden(masked)
#print(QFI)