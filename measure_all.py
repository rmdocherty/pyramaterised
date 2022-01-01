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
    elif circuit_type == "Circuit_1":
        layers = cs.circuit_1(p, N)
    elif circuit_type == "Circuit_2":
        layers = cs.circuit_2(p, N)
    elif circuit_type == "Circuit_9":
        layers = cs.circuit_9(p, N)
    elif circuit_type == "qg_circuit":
        layers = cs.qg_circuit(p, N)
    elif circuit_type == "generic_HE":
        layers = cs.generic_HE(p, N)

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
    circuit_magic = 0
    circuit_eigvals = []
    circuit_capped_eigvals = []
    
    training_final_data = {}
    training_magics = []
    training_entanglement = []
    training_costs = []
    training_fidelities = []
    training_final_states = []
    training_gkps = []

    max_magic = max_magic = np.log((2**n_qubits) + 1) - np.log(2)
    circuit_metadata = {"Type": circuit_type, "N_qubits": n_qubits, "N_layers": n_layers,
                        "H": hamiltonian, "angles": start, "train": train,
                        "train_method": train_method, "epsilon": epsilon, "rate": rate,
                        "n_repeats": n_repeats, "n_samples": n_samples}
    circuit = generate_circuit(circuit_type, n_qubits, n_layers, hamiltonian)
    circuit_m = Measurements(circuit)
    if start == "clifford":
        circuit_data = circuit_m.efficient_measurements(n_samples, full_data=True, angles='clifford')
    else:
        circuit_data = circuit_m.efficient_measurements(n_samples, full_data=True)
    circuit_expr = circuit_data['Expr']
    circuit_entanglement = circuit_data['Ent']
    circuit_magic = circuit_data['Magic']
    circuit_GKP = circuit_data['GKP']
    circuit_gradients = []

    for i in range(n_samples // 20):
        circuit.set_H('ZZ')
        if start == "random":
            init_angles = [random.random() * 2 * np.pi for i in range(circuit.n_params)]
        elif start == "clifford":
            clifford_angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
            init_angles = [random.choice(clifford_angles) for i in range(circuit.n_params)]
        gradients = circuit_m.get_gradient_vector(init_angles)
        for g in gradients:
            circuit_gradients.append(g)
        circuit_QFI = circuit_m._get_QFI(grad_list=circuit_m.gradient_list) #should put this in the repeats and save whole array
        eigvals, _ = circuit_m.get_eigenvalues(circuit_QFI)
        circuit_eigvals.append(eigvals)
        circuit_capped_eigvals.append(circuit_m.new_measure(circuit_QFI))
    circuit_mean_grad = np.mean(circuit_gradients)
    circuit_var_grad = np.var(circuit_gradients)

    circuit.set_H(hamiltonian)

    circuit_data_dict = {"Expr": circuit_expr, "Ent": circuit_entanglement, 
                         "Reyni": circuit_magic, "GKP": circuit_GKP,
                         "Gradients": circuit_gradients, "QFIM_e-vals": circuit_eigvals, 
                         "Capped_e-vals": circuit_capped_eigvals}

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

            value, cost, magic, ent, gkp = training_data
            print(f"Training finished on iteration {len(cost)} with cost function = {value}")

            if target_state != -1:
                fidelity = circuit._fidelity(target_state)
                training_fidelities.append(fidelity)
            else:
                training_fidelities.append(-1)

            training_costs.append(cost)
            training_magics.append(magic)
            training_entanglement.append(ent)
            training_gkps.append(gkp)
            circuit_final_state = np.abs(circuit._quantum_state.data.toarray())
            training_final_states.append(circuit_final_state)

        training_costs = extend(training_costs)
        training_magics = extend(training_magics)
        training_entanglement = extend(training_entanglement)
        training_gkps = extend(training_gkps)

        training_final_data = {"Cost": training_costs, "Ent": training_entanglement, "Reyni": training_magics, "GKP": training_gkps, "Final_states": training_final_states}

    out_dict = {"Metatdata": circuit_metadata, "Circuit_data": circuit_data_dict, "Training_data": training_final_data}

    if save is True:
        if train is True:
            file_name = f"{circuit_type}_{hamiltonian}_{n_qubits}q_{n_layers}l_{n_repeats}r_{start}_{train_method}"
        else:
            file_name = f"{circuit_type}_{hamiltonian}_{n_qubits}q_{n_layers}l_{n_repeats}r_{start}"
        with open(f'data/{file_name}.pickle', 'wb') as handle:
            pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Completed circuit {circuit_type}_{hamiltonian}_{n_qubits}q_{n_layers}l_{n_repeats}")
    return 0 #out_dict


