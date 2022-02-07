#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 13:49:13 2022

@author: ronan
"""
import pickle
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

from t_plot import plot1D
from measurement import Measurements
from PQC_lib import PQC


#%%
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
    
    def cost(self):
        return None

load = True
haars = []
if load is False:
    for i in range(1,10):
        haar = Haar(i)
        haar_m = Measurements(haar)
        magic = haar_m.efficient_measurements(100, expr=False, ent=False, eom=False, GKP=True)
        haars.append(magic['GKP'])
    np.save("data/haar_values.npy", np.array(haars))
else:
    haars = np.load("data/haar_values1.npy")

#%%

#circuit_data = {"TFIM": [], "XXZ": [], "generic_HE": [], "fsim": [], "fermionic": [], "qg_circuit": [], "NPQC": [], "y_CPHASE": [], "zfsim": []} #, "Circuit_9": []


def load_file(circuit, n_q, n_l):
    file_name_1 = f"data/combined/{circuit}_ZZ_{n_q}q_{n_l}l_0r_30g_random.pickle"
    file_name_2 = f"data/combined/{circuit}_ZZ_{n_q}q_{n_l}l_0r_0g_random.pickle"
    try:
        file = open(file_name_1, 'rb')
    except FileNotFoundError:
        try:
            file = open(file_name_2, 'rb')
        except FileNotFoundError:
            return []
    temp_data = pickle.load(file)
    file.close()
    return temp_data


def load_data(circuits_to_load, qubit_range=(2, 13), depth_range=(1, 14)):
    circuit_data = {key: [] for key in circuits_to_load}
    for c in circuits_to_load:
        for n_qubits in range(qubit_range[0], qubit_range[1]):
            qubit_data = []
            for n_layers in range(depth_range[0], depth_range[1]):
                if c in ["XXZ", "TFIM", "fsim", "zfsim"] and n_qubits % 2 == 1:
                    file_data = []
                else:
                    file_data = load_file(c, n_qubits, n_layers)
                qubit_data.append(file_data)
            circuit_data[c].append(qubit_data)
    return circuit_data


def get_values_from_circuit_data(data_dict, quantity, index_n=0):
    N = index_n + 2

    if quantity in ["QFIM_e-vals", "avg_e-vals", "std_e-vals"]:
        val = data_dict["QFIM_e-vals"]
    else:
        val = data_dict[quantity]

    if quantity == "Expr":
        if N < 7:
            dummy = Measurements(PQC(N), load=False)
            avg = dummy._expr(val, 2**N)
            std = 0
        else:
            avg, std = 0, 0
    elif quantity == "Gradients":
        grad_list = np.array(val).flatten()
        avg = np.var(grad_list)
        std = 0
    elif quantity == "Reyni":
        haar_m = np.log(3 + 2**N) - np.log(4)
        avg = np.mean(val) / haar_m
        std = np.std(val) / haar_m
    elif quantity == "GKP":
        haar_m = haars[index_n][0]
        avg = np.mean(val) / haar_m
        std = np.std(val) / haar_m
    elif quantity == "QFIM_e-vals":
        non_zeros = []
        for i in val:
            n_non_zero = len([j for j in i if np.abs(j) > 10**-12])
            non_zeros.append(n_non_zero)
        avg = np.mean(non_zeros)
        std = 0
    else:
        avg = np.mean(val)
        std = np.std(val)
    return (avg, std)


def get_data_array_from_dict(circuit_dict, quantity, fixed_N=0, fixed_d=0):
    x_arr, y_arr, err_arr = [], [], []
    for circuit, data_dict_list in circuit_dict.items():
        circuit_x, circuit_y, circuit_err = [], [], []
        if fixed_N >= 2:
            index_N = fixed_N - 2
            for depth, data_dict in enumerate(data_dict_list[index_N]):
                try:
                    out_data = get_values_from_circuit_data(data_dict["Circuit_data"], quantity, index_n=index_N)
                    circuit_x.append(depth + 1)
                    circuit_y.append(out_data[0])
                    circuit_err.append(out_data[1])
                except TypeError:
                    #print(circuit, index_N, depth)
                    pass

            x_arr.append(circuit_x)
            y_arr.append(circuit_y)
            err_arr.append(circuit_err)

        elif fixed_d != 0:
            index_d = fixed_d - 1 if fixed_d != -1 else -1
            for N, data_dict in enumerate(data_dict_list):
                try:
                    out_data = get_values_from_circuit_data(data_dict[index_d]["Circuit_data"], quantity, index_n=N)
                    circuit_x.append(N + 2)
                    circuit_y.append(out_data[0])
                    circuit_err.append(out_data[1])
                except TypeError:
                    pass

            x_arr.append(circuit_x)
            y_arr.append(circuit_y)
            err_arr.append(circuit_err)
    return (x_arr, y_arr, err_arr)


def map_style_dict_to_list(loaded_circuits, style_dict):
    style_list = []
    for c in loaded_circuits:
        style_list.append(style_dict[c])
    return style_list


markers = {"TFIM": 'p', "XXZ": 'x', "generic_HE": 'h', "fsim": 'o', "fermionic": 's', "Circuit_9": '$9$', "qg_circuit": '^', "y_CPHASE": '$Y$', "zfsim": '$Z$', "NPQC": 'v'}
colours = {"TFIM": "#5ADBFF", "XXZ": "#235789", "generic_HE": "#C1292E", "fsim": "#F1D302", "fermionic": "#E9BCB7", "qg_circuit": "#139A43", "Circuit_9": "#FF8552", "y_CPHASE": "orange", "NPQC": "#395756", "zfsim": "#A67DB8"}
label_dict = {"Expr": "Expr, $D_{KL}$", "Ent": "Ent, <Q>", "Reyni": "Relative Reyni Magic", "GKP": "Relative GKP Magic", "Capped_e-vals": "Capped Eigenvalues", "Gradients": "Var{Grad}", "QFIM_e-vals": "$G_{c}$"}
title_dict = {"Expr": "Expressibility", "Ent": "Entanglement", "Reyni": "Reyni Entropy of Magic", "GKP": "GKP Magic ", "Capped_e-vals": "Capped Eigenvalues", "Gradients": "Var{Grad}", "QFIM_e-vals": "Effective quant"}


to_load = ["TFIM", "XXZ", "generic_HE", "qg_circuit", "fsim", "fermionic", "NPQC", "y_CPHASE", "zfsim"]
# marker_list = map_style_dict_to_list(to_load, markers)
# colour_list = map_style_dict_to_list(to_load, colours)


def plot_data(circuit_dict, quantity, fixed_N=0, fixed_depth=0, logx=False, 
              logy=False, save=False, add_legend=True):
    marker_list = map_style_dict_to_list(circuit_dict.keys(), markers)
    colour_list = map_style_dict_to_list(circuit_dict.keys(), colours)
    if save is True:
        logx_str = "logx" if logx else "" 
        logy_str = "logy" if logy else ""
        save_path = "plots"
        dataname = (f"{quantity}_{fixed_N}q_{fixed_depth}d_{logx_str}{logy_str}")
    else:
        save_path = ""
        dataname=""
    if fixed_N > 0:
        data = get_data_array_from_dict(circuit_dict, quantity, fixed_N=fixed_N)
        x_axis_str = "Depth"
    else:
        data = get_data_array_from_dict(circuit_dict, quantity, fixed_d=fixed_depth)
        x_axis_str = "N"
    if add_legend == True:
        legend = circuit_dict.keys()
    else:
        legend = []
    plot1D(data[1], x=data[0], xlabelstring=x_axis_str, ylabelstring=label_dict[quantity], legend=legend, 
           customMarkerStyle=marker_list, customlinewidth=[4 for i in range(len(to_load))], 
           customplot1DLinestyle=["-" for i in range(len(to_load))], customColorList=colour_list,
           logy=logy, logx=logx, saveto=save_path, dataname=dataname)

#%%
to_load = ["TFIM", "generic_HE", "qg_circuit", "fsim", "fermionic", "y_CPHASE", "zfsim"]
cd = load_data(to_load, qubit_range=(2, 5))

plot_data(cd, "Expr", fixed_N=4, logy=True, save=True, add_legend=False)
plot_data(cd, "Reyni", fixed_N=4, save=True, add_legend=False)
plot_data(cd, "GKP", fixed_N=4, save=True, add_legend=False)

plot_data(cd, "Ent", fixed_N=4, save=True, add_legend=True)

plot_data(cd, "Gradients", fixed_N=4, logy=True, save=True, add_legend=False)

#%%
cd = load_data(to_load, qubit_range=(2, 7))
plot_data(cd, "Expr", fixed_N=4, logy=True)



#%%
to_load = ["TFIM", "generic_HE", "qg_circuit", "fsim", "fermionic", "NPQC", "y_CPHASE", "zfsim"]
cd = load_data(to_load, qubit_range=(2, 9))
plot_data(cd, "Reyni", fixed_depth=-1 , save=True, add_legend=False)
plot_data(cd, "GKP", fixed_depth=-1, save=True, add_legend=False)

plot_data(cd, "Ent", fixed_depth=-1, save=True, add_legend=False)

cd = load_data(to_load, qubit_range=(2, 7))
plot_data(cd, "Expr", fixed_depth=-1, logy=True, save=True, add_legend=False)

cd = load_data(to_load, qubit_range=(2, 13))
plot_data(cd, "Gradients", fixed_depth=-1, logy=True, save=True, add_legend=False)
#%%
expo_scale = ["XXZ", "generic_HE", "qg_circuit", "fsim", "zfsim"]
non_expo_scale = ["y_CPHASE", "TFIM", "fermionic"]

for count, circuits in enumerate([expo_scale, non_expo_scale]):
    temp_cd = load_data(circuits, qubit_range=(2, 7))
    temp_data = get_data_array_from_dict(temp_cd, "QFIM_e-vals", fixed_N=0, fixed_d=-1)
    for i, c in enumerate(circuits):
        x = temp_data[0][i]
        logx = np.log(np.array(x))
        y = temp_data[1][i]
        logy = np.log(np.array(y))
        if count == 0:
            log_coeff = np.polyfit(x, logy, deg=1)
        else:
            log_coeff = np.polyfit(logx, logy, deg=1)
        print(f"{c} linear coeffs: {log_coeff[0]:.3f}, {log_coeff[1]:.3f}")
    
    if count == 0:
        plot_data(temp_cd, "QFIM_e-vals", fixed_depth=-1, logy=True, save=True, add_legend=True)
    else:
        plot_data(temp_cd, "QFIM_e-vals", fixed_depth=-1, logy=True, logx=True, save=True, add_legend=True)
