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
import circuit_structures as cs
from helper_functions import pretty_subplot, extend
from os import mkdir

#%%

N_REPEATS = 10

CIRCUIT = "TFIM" #options are HE, NPQC, TFIM, TFIM_modified, XXZ
N_QUBITS = 4
N_LAYERS = 4
N_SAMPLES = 100
HAMILTONIAN = "TFIM" #options are ZZ, TFIM

TRAIN = True
START = "random" #options are random, fixed
TRAIN_METHOD = "BFGS"
TRAIN_EPSILON = 1e-6
LEARNING_RATE = 0.001

SAVE = True
PLOT = True

g, h = 1, 0

#%%


def generate_circuit(circuit_type, N, p, hamiltonian):
    circuit = pqc.PQC(N)

    if circuit_type == "NPQC":
        layers, theta_ref = cs.NPQC_layers(p, N)
    elif circuit_type == "TFIM":
        layers = cs.gen_TFIM_layers(p, N)
    elif circuit_type == "TFIM_modified":
        layers = cs.gen_modified_TFIM_layers(p, N)
    elif circuit_type == "XXZ":
        layers = cs.gen_XXZ_layers(p, N)

    for l in layers:
        circuit.add_layer(l)

    if hamiltonian == "TFIM":
        ham = cs.TFIM_hamiltonian(N, g=g, h=h)
        circuit.set_H(ham)

    return circuit


def measure_everything(circuit_type, n_qubits, n_layers, n_repeats, n_samples, \
                       hamiltonian='ZZ', train=True, start='random', start_angles=[], \
                           train_method='gradient', epsilon=1e-6, rate=0.001, save=True, plot=True, target_state=-1):
    random.seed(2)
    circuit_expr = 0
    circuit_entanglement = 0
    circuit_entanglement_std = 0
    circuit_magic = 0
    circuit_magic_std = 0
    training_magics = []
    training_entanglement = []
    training_costs = []
    training_fidelities = []

    max_magic = max_magic = np.log((2**n_qubits) + 1) - np.log(2)

    circuit = generate_circuit(circuit_type, n_qubits, n_layers, hamiltonian)
    circuit_m = Measurements(circuit)

    circuit_data = circuit_m.efficient_measurements(n_samples)
    circuit_expr = circuit_data['Expr']
    circuit_entanglement, circuit_entanglement_std = circuit_data['Ent']
    circuit_magic, circuit_magic_std = circuit_data['Magic']

    circuit_data_dict = {"Expr": circuit_expr, "Ent": circuit_entanglement, "Reyni": circuit_magic / max_magic, "GKP": -1, "Fidelity": -1}
    circuit_std_dict = {"Expr": -1, "Ent": circuit_entanglement_std, "Reyni": circuit_magic_std / max_magic, "GKP": -1, "Fidelity": -1}

    if train is True:
        for i in range(n_repeats):
            print(f"On repeat {i}")
            circuit = generate_circuit(circuit_type, n_qubits, n_layers, hamiltonian)
            circuit_m = Measurements(circuit)
            if start == "random":
                init_angles = [random.random() * 2 * np.pi for i in range(circuit.n_params)]
            elif start == "clifford":
                clifford_angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
                init_angles = [random.choice(clifford_angles) for i in range(circuit.n_params)]
            else:
                init_angles = start_angles

            training_data = circuit_m.train(epsilon=epsilon, rate=rate, method=train_method, angles=init_angles, trajectory=True, magic=True, ent=True)

            value, cost, magic, ent = training_data
            print(f"Training finished on iteration {len(cost)} with cost function = {value}")

            if target_state != -1:
                fidelity = circuit._fidelity(target_state)
                training_fidelities.append(fidelity)
            else:
                training_fidelities.append(-1)

            training_costs.append(cost)
            training_magics.append(magic)
            training_entanglement.append(ent)
            

        training_costs = extend(training_costs)
        training_magics = extend(training_magics)
        training_entanglement = extend(training_entanglement)
        training_data = {"costs": training_costs, "magics": training_magics, "ents": training_entanglement}
        mean_magic, std_magic = np.mean(training_magics[:, -1]) / max_magic, np.std(training_magics[:, -1]) / max_magic
        mean_ent, std_ent = np.mean(training_entanglement[:, -1]), np.std(training_entanglement[:, -1])
        mean_fidelity, std_fidelities = np.mean(training_fidelities), np.std(training_fidelities)
        training_final_data = {"Expr": -1, "Ent": mean_ent, "Reyni": mean_magic, "GKP": -1, "Fidelity": mean_fidelity}
        training_final_std = {"Expr": -1, "Ent": std_ent, "Reyni": std_magic, "GKP": -1, "Fidelity": mean_fidelity}

    if save is True:
        file_name = f"{circuit_type}_{hamiltonian}_{n_qubits}q_{n_layers}l_{n_repeats}r_{train_method}"
        directory = "data/" + file_name
        mkdir(directory)
        with open(directory + "/" + file_name + "properties.txt", 'w+') as file:
            file.write(f"{n_qubits} qubit, {n_layers} layer {circuit_type} PQC using {hamiltonian} hamiltonian, trained using {train_method} and averaged over {n_repeats} repeats. \n")
            file.write(f"Following properties calculated using {n_samples} state samples: \n")
            file.write(f"Circuit expressibility is {circuit_expr}\n")
            file.write(f"Circuit Entanglement is {circuit_entanglement} +/- {circuit_entanglement_std}\n")
            file.write(f"Circuit Reyni Entropy of Magic is {circuit_magic} +/- {circuit_magic_std}\n")
            if circuit_type == "HE":
                file.write(circuit.__repr__())
        if train is True:
            for key, value in training_data.items():
                np.save(directory + "/" + file_name + key, value)

    if plot is True:
        iters = range(len(training_costs[0]))
        
        frac_mag = training_data["magics"] / max_magic
        fig, axes = plt.subplots(2, 2, sharex=True)
        fig.suptitle(f"{n_qubits} qubit, {n_layers} layer {circuit_type} with {hamiltonian} hamiltonian, optimized with {train_method}", fontsize=20)
        for i in training_data["costs"]:
            axes[0, 0].plot(iters, i, color='orange', lw=1)
        pretty_subplot(axes[0, 0], "Iterations", "Cost function", "", fontsize=18)
        for i in frac_mag:
            axes[0, 1].plot(iters, i, color='green', lw=1)
        pretty_subplot(axes[0, 1], "Iterations", "Fractional Reyni Magic", "", fontsize=18)
        for i in training_data["ents"]:
            axes[1, 0].plot(iters, i, color='blue', lw=1)
        pretty_subplot(axes[1, 0], "Iterations", "Training Entanglement", "", fontsize=18)
        axes[1, 1].axis('off')
        
        data = [circuit_data_dict, training_final_data]
        data_std = [circuit_std_dict, training_final_std]

        cell_text = []
        for key, row in circuit_data_dict.items():
            current_row = []
            for column in range(2):
                if data[column][key] == -1 :
                    value = "-"
                else:
                    value = f"{data[column][key]:.3f}"
                if data_std[column][key] == -1:
                    std = ""
                else:
                    std = f" ± {data_std[column][key]:.2f}"
                text = value + std
                current_row.append(text)
            cell_text.append(current_row)
        table = axes[1, 1].table(cellText=cell_text, rowLabels=list(data[0].keys()),\
                                 colLabels=["Initial", "Final"], loc='center', \
                                     fontsize=20, cellLoc='center', rowLoc='center')

    return training_data, circuit_data


#%%
measure_everything(CIRCUIT, 4, 4, 10, 100, HAMILTONIAN, train_method='gradient', save=False)
