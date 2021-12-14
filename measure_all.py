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
import pickle

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
    training_final_states = []

    max_magic = max_magic = np.log((2**n_qubits) + 1) - np.log(2)
    circuit_metadata = {"Type": circuit_type, "N_qubits": n_qubits, "H": hamiltonian, 
                        "angles": start, "train": train, "train_method": train_method,
                        "epsilon": epsilon, "rate": rate, "n_repeats": n_repeats, 
                        "n_samples": n_samples}
    circuit = generate_circuit(circuit_type, n_qubits, n_layers, hamiltonian)
    circuit_m = Measurements(circuit)

    circuit_data = circuit_m.efficient_measurements(n_samples)
    circuit_expr = circuit_data['Expr']
    circuit_entanglement, circuit_entanglement_std = circuit_data['Ent']
    circuit_magic, circuit_magic_std = circuit_data['Magic']
    circuit_GKP, circuit_GKP_std = circuit_data['GKP']
    circuit_QFI = circuit_m._get_QFI()
    circuit_eigvals, _ = circuit_m.get_eigenvalues(circuit_QFI)
    circuit_capped_eigvals = circuit_m.new_measure(circuit_QFI)
    circuit_gradients = []

    for i in range(n_samples):
        if start == "random":
            init_angles = [random.random() * 2 * np.pi for i in range(circuit.n_params)]
        elif start == "clifford":
            clifford_angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
            init_angles = [random.choice(clifford_angles) for i in range(circuit.n_params)]
        gradients = circuit_m.get_gradient_vector(init_angles)
        for g in gradients:
            circuit_gradients.append(g)
    circuit_var_grad = np.var(circuit_gradients)

    circuit_data_dict = {"Expr": circuit_expr, "Ent": circuit_entanglement, "Ent_std": circuit_entanglement_std, 
                         "Reyni": circuit_magic / max_magic, "Reyni_std": circuit_magic_std / max_magic, 
                         "GKP": circuit_GKP, "GKP_std": circuit_GKP_std, "Var_grad": circuit_var_grad, "QFIM_e-vals": circuit_eigvals, 
                         "Capped_e-vals": circuit_capped_eigvals}
    #circuit_std_dict = {"Ent": circuit_entanglement_std, "Reyni": circuit_magic_std / max_magic, "GKP": -1}

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
            circuit_final_state = np.abs(circuit._quantum_state.data.toarray())
            training_final_states.append(circuit_final_state)

        training_costs = extend(training_costs)
        training_magics = extend(training_magics)
        training_entanglement = extend(training_entanglement)
        training_data = {"costs": training_costs, "magics": training_magics, "ents": training_entanglement}
        mean_magic, std_magic = np.mean(training_magics[:, -1]) / max_magic, np.std(training_magics[:, -1]) / max_magic
        mean_ent, std_ent = np.mean(training_entanglement[:, -1]), np.std(training_entanglement[:, -1])
        mean_fidelity, std_fidelities = np.mean(training_fidelities), np.std(training_fidelities)
        training_final_data = {"Cost_arrs": training_costs, "Ent": mean_ent, "Ent_std": std_ent, "Ent_arrs": training_entanglement , "Reyni": mean_magic, "Reyni_std": std_magic, "Reyni_arrs": training_magics, "GKP": -1, "GKP_std": -1, "Final_states": training_final_states}

    out_dict = {"Metatdata": circuit_metadata, "Circuit_data": circuit_data_dict, "Training_data": training_final_data}

    if save is True:
        file_name = f"{circuit_type}_{hamiltonian}_{n_qubits}q_{n_layers}l_{n_repeats}r_{train_method}"
        with open(f'{file_name}.pickle', 'wb') as handle:
            pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return out_dict


#%%
a = measure_everything("TFIM", 4, 4, 2, 100, HAMILTONIAN, start='random', train_method='BFGS', save=True)
print(a)