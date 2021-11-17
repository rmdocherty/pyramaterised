#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 12:15:15 2021

@author: ronan
"""
import PQC_lib as pqc
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import random
from measurement import Measurements
from circuit_structures import gen_clifford_circuit, NPQC_layers, find_overparam_point
from helper_functions import pretty_graph

#%%
p = 10
N = 4
all_entropies = []

N_repeats = 100

for n in range(N_repeats):
    clifford_layers = gen_clifford_circuit(p, N)
    entropies = []
    if n % 100 == 0:
        print(f"{n} circuits computed")
    for i in range(p // 2): #want to insert in middle of circuit
        insertion_point = i + p // 4 #insert from the quarter upwards
        clifford_layers[insertion_point] = clifford_layers[insertion_point] + [pqc.T(i, N) for i in range(N)]
        insert_circuit = pqc.PQC(N)
        for l in clifford_layers:
            insert_circuit.add_layer(l)
        insert_circuit._quantum_state = qt.Qobj(insert_circuit.run())
        i_c_m = Measurements(insert_circuit)
        e = i_c_m.efficient_measurements(1, expr=False, ent=False, eom=True)
        entropies.append(e['Magic'][0])

    entropies = np.array(entropies)
    max_magic = np.log((2**N) + 1) - np.log(2)
    entropies = entropies / max_magic
    all_entropies.append(entropies)

#%%
stderr = True
magics = np.mean(all_entropies, axis=0)
if stderr:
    magic_stds = np.std(all_entropies, axis=0) / np.sqrt(len(all_entropies))
    label_str = f"Standard error of the mean over {len(all_entropies)} repeats"
else:
    magic_stds = np.std(all_entropies, axis=0)
    label_str = f"Standard deviation over {len(all_entropies)} repeats"
    
def plot_magic(N, p, magics, magic_stds, color, std_color, label_str, title_str):
    n_t_gates = np.array([(i + 1) for i in range(p // 2)])
    plt.errorbar(n_t_gates, magics, yerr=magic_stds, ls="", marker=".", lw=2, ms=0, color=std_color)
    plt.plot(n_t_gates, magics, ls="-", marker=".", lw=4, ms=15, color=color, label=label_str)
    pretty_graph("Number of T-Gates layers", "Fractional Reyni Entropy of Magic", title_str, 20)
    plt.legend(fontsize=18)


default_title = f"T-Gate injection on {N}-qubit, {p}-layer Clifford Circuit"
plot_magic(N, p, magics, magic_stds, "cornflowerblue", "darkblue", label_str, default_title)

#%%
files = [f"magic_t_gate_1000r_{n}q_40l.npy" for n in range(3, 6)] #"magic_t_gate_1000r_4q_4l.npy", "magic_t_gate_1000r_5q_40l.npy"
stds = [f[:5] + "_std" + f[5:] for f in files]
magic_paths = ["data/" + f for f in files]
std_paths = ["data/" + f for f in stds]
magic_colors = ["lightgreen", "cornflowerblue", "orange"]
std_colors = ["darkgreen", "darkblue", "darkorange"]

title_str = "T-Gate injection on N-qubit, 40-layer Clifford Circuit"
p = 40
for i in range(len(files)):
    N = i + 3
    label = f"{N} qubits"
    magic = np.load(magic_paths[i])
    magic_std = np.load(std_paths[i])
    plot_magic(N, p, magic, magic_std, magic_colors[i], std_colors[i], label, title_str)

#%%

p = 60
N = 4
N_gates = 6

entropies = []

clifford_layers = gen_clifford_circuit(p, N)
for g in range(N_gates):
    insertion_point = random.randint(p//4, (3 * p) //4)
    q_on = random.randint(0, N - 1)
    clifford_layers[insertion_point] = clifford_layers[insertion_point] + [pqc.T(q_on, N)]
    insert_circuit = pqc.PQC(N)
    for l in clifford_layers:
        insert_circuit.add_layer(l)
    insert_circuit._quantum_state = qt.Qobj(insert_circuit.run())
    i_c_m = Measurements(insert_circuit)
    e = i_c_m.efficient_measurements(1, expr=False, ent=False, eom=True)
    entropies.append(e['Magic'][0])

print(entropies)

#%%
P, N = 6, 6

layers, theta_ref = NPQC_layers(P, N)
train_NPQC = pqc.PQC(N)
for l in layers:
    train_NPQC.add_layer(l)

train_NPQC_m = Measurements(train_NPQC)
ener, magics = train_NPQC_m.train(method="QNG", magic=True, angles=theta_ref) #theta_ref
#%% Really interesting results - when initialised with theta_ref, training always seems to take 1762 iterations and gives really nice gaussian
#magics = np.load('data/magics_npqc_training_4l_4q.npy')
max_magic = np.log((2**N) + 1) - np.log(2)
magics = np.array(magics) / max_magic
iterations = range(len(magics))
plt.plot(iterations, magics, lw=4)
pretty_graph("Training Iteration", "Fractional Reyni Entropy of Magic", "Magic during NPQC training initialised with reference param", 20)