#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 13:40:05 2021

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

random.seed(1) #for reproducibility

#%% =============================BASIC CIRCUIT TESTS=============================

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


#%% =============================EXPR AND ENT TESTS=============================
"""Tests based on Expr baselines from Fig 1 arXiv:1905.10876v1"""
# idle_circuit = pqc.PQC(1)
# layer = [pqc.PRot(0, 1)]
# idle_circuit.add_layer(layer)
# idle_circuit_m = Measurements(idle_circuit)
# e = idle_circuit_m.efficient_measurements(100, expr=True, ent=False, eom=False) #N=100 -> 4950 state pairs
# print(f"Idle circuit expr is {e['Expr']}, should be around 4.317")

#%%
circuit_A = pqc.PQC(1)
layer = [pqc.H(0, 1), pqc.R_x(0, 1)]
circuit_A.add_layer(layer)
circuit_A_expr = Measurements(circuit_A)
a_e = circuit_A_expr.efficient_measurements(100, expr=True, ent=False, eom=False)
print(f"Circuit_A expr is {a_e['Expr']}, should be around 0.2")
#%%
"""Tests based on Expr and Ent values from Figs 3 and 4 of arXiv:1905.10876v1 """
L = 4
circuit_9 = pqc.PQC(4)
layer = [pqc.H(0, 4), pqc.H(1, 4), pqc.H(2, 4), pqc.H(3,4), 
         pqc.CHAIN(pqc.CPHASE, 4),
         pqc.R_x(0, 4), pqc.R_x(1, 4), pqc.R_x(2, 4), pqc.R_x(3, 4)]
#circuit_A.set_initialiser(pqc.PRot)
circuit_9.add_layer(layer, L)
#circuit_9.gen_quantum_state()
circuit_9_m = Measurements(circuit_9)
#%% 1min05
e = circuit_9_m.efficient_measurements(100, expr=True, ent=True, eom=False)
print(f"Circuit 9 expr for {L} layers is {e['Expr']} ")

mean, std = e['Ent']
print(f"Circuit 9 entanglement is {mean} +/- {std}")

#%%
L = 3
circuit_1 = pqc.PQC(4)
layer = [pqc.R_x(0, 4), pqc.R_x(1, 4), pqc.R_x(2, 4), pqc.R_x(3, 4),
         pqc.R_y(0, 4), pqc.R_y(1, 4), pqc.R_y(2, 4), pqc.R_y(3, 4)]
circuit_1.add_layer(layer, L)

circuit_1_m = Measurements(circuit_1)
#%%
c1_out = circuit_1_m.efficient_measurements(100, expr=True, ent=True, eom=False)
print(f"Circuit 1 expressibility at L={L} is {c1_out['Expr']}")

mean, std = c1_out['Ent']
print(f"Circuit 1 entanglement is {mean} +/- {std}")

#%%
N = 4
L = 3
circuit_2 = pqc.PQC(N)
layer = [pqc.R_x(0, N), pqc.R_x(1, N), pqc.R_x(2, N), pqc.R_x(3, N),
         pqc.R_z(0, N), pqc.R_z(1, N), pqc.R_z(2, N), pqc.R_z(3, N),
         pqc.CNOT([3, 2], N), pqc.CNOT([2, 1], N), pqc.CNOT([1, 0], N)]
circuit_2.add_layer(layer, n=L)

circuit_2._quantum_state = circuit_2.run()
circuit_2_m = Measurements(circuit_2)

c2_efd = circuit_2_m.get_effective_quantum_dimension(10**-12)
print(f"Circuit 2 effective quantum dimensions is {c2_efd}")

c2_out = circuit_2_m.efficient_measurements(100, expr=True, ent=True, eom=False)
print(f"Circuit 2 expr for {L} layers is {c2_out['Expr']} ")

"""Check if Q value measurements is same as Meyer Wallach method"""
mean, std = c2_out['Ent']
print(f"Circuit 2 entanglement is {mean} +/- {std}")

skip = True
if skip is False:
    c2_mw = circuit_2_m.meyer_wallach(100)
#print(f"Circuit 2 MW is {c2_mw} ")

#%%
L = 3
circuit_11 = pqc.PQC(4)
layer = [pqc.R_y(0, 4), pqc.R_y(1, 4), pqc.R_y(2, 4), pqc.R_y(3, 4),
         pqc.R_z(0, 4), pqc.R_z(1, 4), pqc.R_z(2, 4), pqc.R_z(3, 4),
                 pqc.CNOT([1, 0], 4), pqc.CNOT([3, 2], 4),
                     pqc.R_y(1, 4), pqc.R_y(2, 4),
                     pqc.R_z(1, 4), pqc.R_z(2, 4),
                         pqc.CNOT([2, 1], 4)]
circuit_11.add_layer(layer, L)
circuit_11_m = Measurements(circuit_11)

c11_out = circuit_11_m.efficient_measurements(104)
print(f"Circuit 11 expr for {L} layers is  {c11_out['Expr']}")

#%% =============================QUANTUM GEOMETRY CIRCUIT TESTS=============================
"""Tests based on default circuit in arXiv:2102.01659v1 github"""
qg_circuit = pqc.PQC(4)
init_layer = [pqc.fixed_R_y(i, 4, np.pi / 4) for i in range(4)]
layer1 = [pqc.R_z(0, 4), pqc.R_x(1, 4), pqc.R_y(2, 4), pqc.R_z(3, 4), pqc.CHAIN(pqc.CNOT, 4)]
layer2 = [pqc.R_x(0, 4), pqc.R_x(1, 4), pqc.R_x(2, 4), pqc.R_y(3, 4), pqc.CHAIN(pqc.CNOT, 4)]
layer3 = [pqc.R_z(0, 4), pqc.R_x(1, 4), pqc.R_y(2, 4), pqc.R_y(3, 4), pqc.CHAIN(pqc.CNOT, 4)]


qg_circuit.add_layer(init_layer)
qg_circuit.add_layer(layer1)
qg_circuit.add_layer(layer2)
qg_circuit.add_layer(layer3, n=1)

qg_circuit.gen_quantum_state()
default_angles = [
        3.21587011, 5.97193953, 0.90578156, 5.96054027,
        1.9592948 , 2.65983852, 5.20060878, 2.571074,
        3.45319898, 0.17315902, 4.73446249, 3.38125416]

# qg_circuit._quantum_state = qt.Qobj(qg_circuit.run(
#         angles=[
#         3.21587011, 5.97193953, 0.90578156, 5.96054027,
#         1.9592948 , 2.65983852, 5.20060878, 2.571074,
#         3.45319898, 0.17315902, 4.73446249, 3.38125416]))
qg_circuit._quantum_state = qt.Qobj(qg_circuit.run())
print(qg_circuit)

energy = qg_circuit.cost(default_angles)

print(f"Energy is {energy}, should be 0.46135870050914374")
qg_m = Measurements(qg_circuit)
#%%
efd = qg_m.get_effective_quantum_dimension(10**-12)
print(f"Effective quantum dimension is {efd}, should be 12")
new_measure = qg_m.new_measure()
print(f"New measure is {new_measure}")
out = qg_m.efficient_measurements(500)
entropy = out['Magic']
print(f"Magic is {entropy[0]} +/- {entropy[1]}")

#%% 
"""Tested with the fixed angles as initial params and took 3539 iteraions = 1m42s
 to 1e-6 accuracy and cost function is 0.34583907096349875"""
out = qg_m.train(rate=0.001, method="BFGS", angles=[])
print(out)
"""Lowest we've recorded is -0.69589... but there is massive variance"""
#%% =============================ENTROPY OF MAGIC TESTS=============================


class Bell:
    def __init__(self):
        self._n_qubits = 2
        self._quantum_state = qt.states.bell_state('11')

    def gen_quantum_state(self):
        return qt.states.bell_state('11')

class One:
    def __init__(self):
        self._n_qubits = 1
        self._quantum_state = qt.basis(1, 1)

    def gen_quantum_state(self):
        return qt.basis(1, 1)

bell_m = Measurements(Bell())
e = bell_m.entropy_of_magic()
g = bell_m.GKP_magic()
print(f"Reyni Entropy of Magic is {e}, should be 0 for stabiliser state")
print(f"GKP Magic is {g}, should be 0 for stabiliser state")


"""
Generate max_N * max_P circuits comprised entriely of random clifford gates
from the group {H, S, CNOT} with n qubits, p layers and n operations per layer.
Then calculate the entropy of magic of these circuits - should be 0 or close
to it for all p, n values.
"""

max_N = 6
max_P = 12

entropies = []
GKPs = []
for n in range(2, max_N):
    for p in range(1, max_P):
        layers = gen_clifford_circuit(p, n)
        clifford_circuit = pqc.PQC(n)
        for l in layers:
            clifford_circuit.add_layer(l)
        clifford_circuit._quantum_state = qt.Qobj(clifford_circuit.run())
        c_c_m = Measurements(clifford_circuit)
        e = c_c_m.entropy_of_magic()
        entropies.append(e)
        gkp = c_c_m.GKP_magic()
        GKPs.append(gkp)

print(f"Entropies of magic are {entropies}, should be roughly 0") #values are ~0 for all so further proff code is working.

#%% Just see if it's generating random clifford circuits
N, P = 4, 5

for i in range(10):
    layers = gen_clifford_circuit(P, N)
    clifford_circuit = pqc.PQC(N)
    for l in layers:
        clifford_circuit.add_layer(l)
    print(clifford_circuit)

#%% Test by looking at magic of Haar states (should be high)

"""
Accoring to https://arxiv.org/pdf/2106.12587.pdf , the expected reyni entropy
of magic for haar random states is log(3 + 2^N) - log(4). Seems to match up
well, though note some measured EoMs are ever so slightly less than the analytic
value, which shouldn't be the case as these are lower bounds. Will chalk this
up to computational error at this point.
"""

N = 4


class Haar():
    def __init__(self, N):
        self._n_qubits = N
        circuit_state = qt.tensor([qt.basis(2, 0) for i in range(self._n_qubits)])
        haar = qt.random_objects.rand_unitary_haar(2**N, [[2 for i in range(N)], [2 for i in range(N)]])
        self._quantum_state = haar * circuit_state

    def gen_quantum_state(self):
        N = self._n_qubits
        circuit_state = qt.tensor([qt.basis(2, 0) for i in range(self._n_qubits)])
        haar = qt.random_objects.rand_unitary_haar(2**N, [[2 for i in range(N)], [2 for i in range(N)]])
        self._quantum_state = haar * circuit_state
        return self._quantum_state


for i in range(2, 8):
    haar = Haar(i)
    haar_m = Measurements(haar)
    magic = haar_m.efficient_measurements(100, expr=False, ent=False, eom=True)
    m, m_std = magic['Magic']
    upper_bound = np.log((2**i) + 1) - np.log(2)
    #print(f"Reyni entropy of magic for Haar unitary of {i} qubits is {m} +/- {m_std}, fraction of max magic is {m / upper_bound} and difference is {(upper_bound - m)}")
    print(f"{i} qubit Haar state EoM is {m / upper_bound}")
    avg_magic = np.log(3 + 2**i) - np.log(4)
    frac_avg = avg_magic / upper_bound
    print(f"Expected fractional magic is {frac_avg}")

#%% =============================NPQC TESTS=============================
"""
Testing using an NPQC as defined in https://arxiv.org/pdf/2107.14063.pdf
Defining property of NPQC is that QFI(theta_r) = Identity, where
theta_r_i =  0 for R_x gates, pi/2 for R_y gates. This speeds up training when
theta_r used as initial parameter vector. To check our QFI code is working,
generate each NPQC with N qubits and P <= 2**(N/2) layers and check it's QFI
is the identity when initialised with theta_r - this means every off diagonal
element of the QFI should be 0, which is checked for in check_iden().
"""


def check_iden(A):
    diag_entries = []
    non_zero_off_diags = []
    for i, row in enumerate(A):
        for j, column in enumerate(row):
            if i == j:
                diag_entries.append(column)
            elif column != 0 and i != j:
                non_zero_off_diags.append(column)
    if len(non_zero_off_diags) == 0:
        print("No nonzero off diagonals!")
    elif len(non_zero_off_diags) > 0:
        print("There are nonzero off diagonals!")
        print(non_zero_off_diags)


for N in range(4, 10, 2): #step=2 for even N
    for P in range(1, 2**(N // 2) + 1): #works iff P < 2^(N/2) so agrees with paper.
        print(f"NPQC with {N} qubits and {P} layers")
        layers, theta_ref = NPQC_layers(P, N)
        NPQC = pqc.PQC(N)
        for l in layers:
            NPQC.add_layer(l)
        #need to set |psi> before making QFI measurements
        NPQC._quantum_state = qt.Qobj(NPQC.run(angles=theta_ref))
        NPQC_m = Measurements(NPQC)
        QFI = np.array(NPQC_m._get_QFI())
        masked = np.where(QFI < 10**-12, 0, QFI)
        check_iden(masked)
        print("\n")
#%%
N = 4
P = 4
layers, theta_ref = NPQC_layers(P, N)
NPQC = pqc.PQC(N)
for l in layers:
    NPQC.add_layer(l)

NPQC_m = Measurements(NPQC)
#%%
out = NPQC_m.train(rate=0.01, epsilon=1e-3)


#%%

"""
Add layers to a PQC until the rank of the QFI saturates - this indicates
overparameterisation and should lead to a favorable cost function landscape and
easy training.
"""

test = pqc.PQC(4)
test_layer = [pqc.H(0, 4), pqc.H(1, 4), pqc.H(2, 4), pqc.H(3,4), 
                          pqc.CHAIN(pqc.CPHASE, 4),
              pqc.R_x(0, 4), pqc.R_x(1, 4), pqc.R_x(2, 4), pqc.R_x(3, 4)]
test.add_layer(test_layer, n=1)


op = find_overparam_point(test, [0])
print(f"Test circuit overparameterised after {op} layesr added")
#%%
test_m = Measurements(test)
ener, traj, *others = test_m.train(trajectory=True)


iterations = range(len(traj))
plt.plot(iterations, traj, lw=4)
pretty_graph("Iterations", "Cost function", "Cost function vs iterations for overparameterised PQC", 20)

#%%

"""
Test TFIM overparameterisation values for different N.
"""

N, p = 4, 4
g = 0.5

TFIM = pqc.PQC(N)
#need to use |+> as initial state for TFIM model
plus_state = (1/np.sqrt(2)) * (qt.basis(2,0) + qt.basis(2,1))
TFIM.set_initial_state(plus_state)

hamiltonian = TFIM_hamiltonian(N, g=g)
groundstate = hamiltonian.groundstate()[1]
TFIM.set_H(hamiltonian)

TFIM_layers = gen_TFIM_layers(p, N)
for l in TFIM_layers:
    TFIM.add_layer(l)

#TFIM.flip_deriv()
#%%
random.seed(2)
random_angles = [random.random()*2*np.pi for i in range(2*p)]
clifford_angles = [0 for i in range(2*p)]

TFIM_m = Measurements(TFIM)
out = TFIM_m.train(method='BFGS', rate=0.001, epsilon=1e-6, angles=random_angles)
BFGS_min = out[0]

out = TFIM_m.train(method='gradient', rate=0.001, epsilon=1e-6, angles=random_angles, magic=True, trajectory=True)
grad_min = out[0]
print(f"BFGS (scipy minimizer) TFIM min = {BFGS_min}, gradient min = {grad_min}")
#%%
overlap = TFIM._fidelity(groundstate)
print(f"Fidelity of TFIM circuit state after training with groundstate is {overlap} for g={g}")
#%%
plt.figure("magic")
N = 4
max_magic = np.log((2**N) + 1) - np.log(2)
magics = np.array(out[2]) / max_magic
iterations = range(len(magics))
plt.plot(iterations, magics, lw=4, color="green")
pretty_graph("Training Iteration", "Fractional Reyni Entropy of Magic", f"Magic during {N} qubit {p} layer TFIM training initialised with random angles", 20)

traj = np.array(out[1])
plt.figure("trajedy")
plt.plot(iterations, out[1], color="orange", lw=4, label="Gradient descent")
plt.hlines(BFGS_min, 0, iterations[-1], lw=2, ls='dotted', label="BFGS min", color='red')
pretty_graph("Iterations", "Cost function", f"Cost function vs iterations for {N} qubit {p} layer TFIM training initialised with random angles", 20)

plt.legend(fontsize=18)
