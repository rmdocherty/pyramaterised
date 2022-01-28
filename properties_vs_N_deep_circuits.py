#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 12:59:26 2022

@author: ronan
"""

import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import qutip as qt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from helper_functions import pretty_graph
from measurement import Measurements
from PQC_lib import PQC
import numpy as np
from plot import quantity_vs_N, plot_quantity_vs_N

limited_circuit_data = {"XXZ": [], "TFIM": [], "generic_HE": [], "fsim": [], "fermionic": [], "qg_circuit": [], "y_CPHASE": [], "zfsim": []} #, "Circuit_9": []


for circuit_type in limited_circuit_data.keys():
    for n_qubits in range(2, 10):
        qubit_data = []
        if n_qubits % 2 == 1 and circuit_type in ["XXZ", "fermionic", "TFIM"]:
            pass
        else:
            if circuit_type != "y_CPHASE" and n_qubits == 8: #circuit_type in ["fsim", "generic_HE", "qg_circuit", "XXZ"] and
                file_name = f"data/capacity_3/{circuit_type}_ZZ_{n_qubits}q_{4*n_qubits}l_0r_random.pickle"
            elif circuit_type in ["fsim", "generic_HE", "qg_circuit", "zfsim"] and n_qubits == 9:
                file_name = f"data/capacity_3/{circuit_type}_ZZ_{n_qubits}q_{7*n_qubits}l_0r_random.pickle"
            else:
                file_name = f"data/capacity_3/{circuit_type}_ZZ_{n_qubits}q_{3*n_qubits}l_0r_random.pickle"
            with open(file_name, 'rb') as file:
                qubit_data.append(pickle.load(file))
        limited_circuit_data[circuit_type].append(qubit_data)

plot_quantity_vs_N("QFIM_e-vals", 1, check="last", data_array=limited_circuit_data)