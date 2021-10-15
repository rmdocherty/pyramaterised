#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 18:12:45 2021

@author: ronan



Does our QC code match up to the results from Tobias' code?
"""
from QC import QuantumCircuit
from measurement import Measurements
from helper_functions import pretty_subplot
import matplotlib.pyplot as plt
import qutip as qt
import numpy as np
import random
from PQC import *

random.seed(1) #for reproducibility

#%% LOOK AT ENTANGLEMENT TOPOLOGIES
q = QuantumCircuit(4, 1, "chain", "cnot")
print("4 qubit chain topology connected indices")
print(q._gen_entanglement_indices())

print("5 qubit chain topology connected indices")
q2 = QuantumCircuit(5, 3, "chain", "cnot")
print(q2._gen_entanglement_indices())

#%% TEST PQC CODE BY COMPARING TO REFERENCE VALUE
q.run() #should ouput 0.46135870050914374

M = Measurements(q)
EFD = M.get_effective_quantum_dimension(10**-12)
print(f"Effective quantum dimension is {EFD}")

#%% TEST EXPR BY LOOKING AT EXPR OF SINGLE QUBIT SYSTEM - page 5


class SingleQubit():
    def __init__(self):
        self._initial_state = qt.basis(2, 0)
        self._n_qubits = 1


class IdleCircuit(SingleQubit):
    def gen_quantum_state(self, energy_out=False):
        quantum_state = qt.qeye(2) * self._initial_state
        self._quantum_state = qt.Qobj(quantum_state)
        return 0


class CircuitA(SingleQubit):
    def gen_quantum_state(self, energy_out=False):
        hadamard = qt.qip.operations.ry(np.pi / 4)
        angle = random.random() * 2 * np.pi
        R_z = qt.qip.operations.rz(angle)
        quantum_state = R_z * hadamard * self._initial_state
        self._quantum_state = qt.Qobj(quantum_state)
        return 0


class CircuitB(SingleQubit):
    def gen_quantum_state(self, energy_out=False):
        hadamard = qt.qip.operations.ry(np.pi / 4)
        angle1 = random.random() * 2 * np.pi
        R_z = qt.qip.operations.rz(angle1)
        angle2 = random.random() * 2 * np.pi #this isn;t right, needs tobe accodring to haar / arcsin
        R_x = qt.qip.operations.rx(angle2)
        quantum_state = R_x * R_z * hadamard * self._initial_state
        self._quantum_state = qt.Qobj(quantum_state)
        return 0


class Unitary(SingleQubit):
    def gen_quantum_state(self, energy_out=False):
        self._quantum_state = qt.random_objects.rand_unitary(2) * self._initial_state
        return 0


class BellState():
    def __init__(self):
        self._initial_state = qt.bell_state("00")
        self._n_qubits = 2

    def gen_quantum_state(self, energy_out=False):
        self._quantum_state = self._initial_state
        return 0


class Fock():
    def __init__(self):
        self._initial_state = qt.basis([2, 2])
        self._n_qubits = 2

    def gen_quantum_state(self, energy_out=False):
        self._quantum_state = self._initial_state
        return 0

#%%
idle = IdleCircuit()
idle_circuit_expr = Measurements(idle)
idle_circuit_expr.expressibility(1000, graphs=True) #shoud be around 4.30
#%%
A = CircuitA()
circuit_A_expr = Measurements(A)
circuit_A_expr.expressibility(1000, graphs=True) #shoud be around 0.22
#%%
B = CircuitB()
circuit_B_expr = Measurements(B)
circuit_B_expr.expressibility(1000, graphs=True) #shoud be around 0.02 - isn't atm, order of magnitude too large
#not really seeing ringing at F=0 that A and B have - is this cause of error?
#%%
U = Unitary()
unitary_expr = Measurements(U)
unitary_expr.expressibility(1000, graphs=True) #shoud be around 0.007 - now works!
#%%
M.expressibility(5000, graphs=True)
#%% GET DATA OF EXPR VS DEPTH FOR DIFF ENTANGLERS
entangler_expr = []
for entangler in ["cnot", "cphase", "iswap"]:
    exprs = []
    layers = range(1, 11)
    for depth in layers:
        print(depth)
        q = QuantumCircuit(4, depth, "chain", entangler)
        M = Measurements(q)
        expr = M.expressibility(5000)
        exprs.append(expr)
    entangler_expr.append(exprs)

#%% PLOT GRAPH OF EXPR VS DEPTH FOR DIFF ENTANGLERS
markers = [".", "x", "^"]
for i, entangler in enumerate(["cnot", "cphase", "iswap"]):
    plt.gca().semilogy(layers, entangler_expr[i], ms=15, lw=4, ls="--", marker=markers[i], label=entangler)
pretty_subplot(plt.gca(), "Circuit Depth", "Expr, $D_{KL}$", \
               "Expressibility vs Depth for 4 qubit chain topology", 20)
plt.gca().set_ylim(0, 1)
plt.gca().legend(fontsize=18)

#%%
bell_state = BellState()
ent_M = Measurements(bell_state)
entanglement = ent_M.entanglement(1)
print(entanglement)
#%%
unentangled = Fock()
uent_M = Measurements(unentangled)
unentanglement = uent_M.entanglement(1)
print(unentanglement)
#%%
test_ent = M.entanglement(5000, graphs=(True))
mean, std = np.mean(test_ent), np.std(test_ent)
print(test_ent)
print(f"Mean is {mean} +/- {std} ")
#%%
test_magic = ent_M.entropy_of_magic()
print(test_magic)

#%%
#%%
"""====================NEW PQC TESTS=================="""
#%%
circuit_A = PQC(1, 1)
layer = [H(0, np.pi/4), R_x(0, 0)]
circuit_A.set_gates(layer)
circuit_A_expr = Measurements(circuit_A)
circuit_A_expr.expressibility(1000, graphs=True)