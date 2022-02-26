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
import scipy as sp

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


def resample(F_samples, n_iters, n_resamples, n, dummy):
    exprs = []
    for i in range(n_iters):
        new_samples = np.random.choice(F_samples, size=n_resamples, replace=False)
        #print(new_samples.shape)
        expr = dummy._expr(list(new_samples), n)
        #print(expr)
        exprs.append(expr)
    return np.std(exprs)
    

#%%

#circuit_data = {"TFIM": [], "XXZ": [], "generic_HE": [], "fsim": [], "fermionic": [], "qg_circuit": [], "NPQC": [], "y_CPHASE": [], "zfsim": []} #, "Circuit_9": []


def load_file(circuit, n_q, n_l, loc="combined"):
    if circuit == "clifford":
        start = "clifford"
    else:
        start = "random"
    file_name_1 = f"data/{loc}/{circuit}_ZZ_{n_q}q_{n_l}l_0r_30g_{start}.pickle"
    file_name_2 = f"data/{loc}/{circuit}_ZZ_{n_q}q_{n_l}l_0r_0g_{start}.pickle"
    file_name_3 = f"data/{loc}/{circuit}_ZZ_{n_q}q_{n_l}l_0r_5g_{start}.pickle"
    file_name_4 = f"data/{loc}/{circuit}_ZZ_{n_q}q_{n_l}l_0r_4g_{start}.pickle"
    file_name_5 = f"data/{loc}/{circuit}_ZZ_{n_q}q_{n_l}l_0r_10g_{start}.pickle"
    try:
        file = open(file_name_4, 'rb')
    except FileNotFoundError:
        try:
            file = open(file_name_3, 'rb')
        except FileNotFoundError:
            try:
                file = open(file_name_2, 'rb')
            except FileNotFoundError:
                try:
                    file = open(file_name_1, 'rb')
                except FileNotFoundError:
                    try:
                        file = open(file_name_5, 'rb')
                    except FileNotFoundError:
                        print(f"No data found for {circuit} {n_q}q {n_l}l")
                        return []
    temp_data = pickle.load(file)
    file.close()
    return temp_data


def load_data(circuits_to_load, qubit_range=(2, 13), depth_range=(1, 14), loc="combined"):
    circuit_data = {key: [] for key in circuits_to_load}
    for c in circuits_to_load:
        for n_qubits in range(qubit_range[0], qubit_range[1]):
            qubit_data = []
            for n_layers in range(depth_range[0], depth_range[1]):
                if c in ["XXZ", "TFIM", "fsim", "zfsim", "fermionic"] and n_qubits % 2 == 1:
                    file_data = []
                elif n_qubits == 8 and c == "XXZ" and depth_range == (30,31):
                    file_data = load_file(c, n_qubits, 160, loc=loc)
                elif n_qubits == 8 and c == "clifford" and depth_range == (30,31):
                    file_data = load_file(c, n_qubits, 50, loc=loc)
                elif n_qubits == 8 and depth_range == (30,31):
                    file_data = load_file(c, n_qubits, 30, loc=loc) #set to 50 for expr, 30 for everyhting else
                elif n_qubits > 8 and depth_range == (30,31) and c in ["fsim", "zfsim", "XXZ"]:
                    file_data = load_file(c, n_qubits, 220, loc=loc)
                elif n_qubits > 8 and depth_range == (30,31):
                    file_data = load_file(c, n_qubits, 120, loc=loc)
                else:
                    file_data = load_file(c, n_qubits, n_layers, loc=loc)
                qubit_data.append(file_data)
            circuit_data[c].append(qubit_data)
    return circuit_data


def get_values_from_circuit_data(data_dict, quantity, index_n=0):
    N = index_n + 2

    if quantity in ["QFIM_e-vals", "avg_e-vals", "std_e-vals"]:
        val = data_dict["QFIM_e-vals"]
    elif quantity == "F":
        val = data_dict["Expr"]
    else:
        val = data_dict[quantity]

    if quantity == "Expr":
        if N < 10:
            dummy = Measurements(PQC(N), load=False)
            avg = dummy._expr(val, 2**N)# dummy._expr(val, 2**N)
            std = resample(val, 20, 10000, 2**N, dummy)
        else:
            avg, std = 0, 0
    elif quantity == "Gradients":
        grad_list = np.array(val).flatten()
        avg = np.var(grad_list)
        std = 0
    elif quantity == "Reyni":
        haar_m = np.log(3 + 2**N) - np.log(4)
        avg = np.mean(val) / haar_m
        std = (np.std(val) / haar_m)
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
        std = np.std(non_zeros)
    elif quantity == "F":
        avg = val
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
                    if circuit in ["XXZ", "fermionic"] and N < 1 and quantity == "Gradients":
                        pass
                    else:
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


markers = {"TFIM": 'p', "XXZ": 'x', "generic_HE": 'h', "fsim": 'o', "fermionic": 's', "Circuit_9": '$9$', "qg_circuit": '^', "y_CPHASE": '1', "zfsim": '*', "NPQC": 'v', "clifford": "8"}
colours = {"TFIM": "#5ADBFF", "XXZ": "#235789", "generic_HE": "#C1292E", "fsim": "#F1D302", "fermionic": "#E9BCB7", "qg_circuit": "#139A43", "Circuit_9": "#FF8552", "y_CPHASE": "orange", "NPQC": "#395756", "zfsim": "#A67DB8", "clifford": "#854b00"}
label_dict = {"Expr": "Expr, $D_{KL}$", "Ent": "Entanglement", "Reyni": "$\\mathcal{M}_{2} / \\mathcal{M}_{\mathrm{Haar}}$", "GKP": "$\\mathcal{G} \\mathrm{ / N} $", "Capped_e-vals": "Capped Eigenvalues", "Gradients": "Var{Grad}", "QFIM_e-vals": "$G_{c}$"}
title_dict = {"Expr": "Expressibility", "Ent": "Entanglement", "Reyni": "Reyni Entropy of Magic", "GKP": "GKP Magic ", "Capped_e-vals": "Capped Eigenvalues", "Gradients": "Var{Grad}", "QFIM_e-vals": "Effective quant"}


to_load = ["TFIM", "XXZ", "generic_HE", "qg_circuit", "fsim", "fermionic", "NPQC", "y_CPHASE", "zfsim"]
# marker_list = map_style_dict_to_list(to_load, markers)
# colour_list = map_style_dict_to_list(to_load, colours)


def plot_data(circuit_dict, quantity, fixed_N=0, fixed_depth=0, logx=False, 
              logy=False, save=False, add_legend=True, fontsize=18, figcols=2, errors=False, error_op=0.3):
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
    if errors == True:
        e_bars_y = []
        fill_between = []
        for c in range(len(circuit_dict.keys())):
            x_points = []
            y1_points = []
            y2_points = []
            ebars = [[], []]
            for i in range(0, len(data[2][c])):
                #print(i, len(data[2]), c, len(data[2][c]))
                x_points.append(data[0][c][i])
                y = data[1][c][i]
                std = data[2][c][i]
                y1 = y + std
                if y1 > 1 and quantity not in ["Expr", "Renyi", "QFIM_e-vals"]:
                    y1 = 1
                    y_err_top = 1 - y
                else:
                    y_err_top = std
                y2 = y - std
                if y2 < 0:
                    y2 = 0
                    y_err_bot = y
                else:
                    y_err_bot = std
                y1_points.append(y1)
                y2_points.append(y2)
                ebars[1].append(y_err_top)
                ebars[0].append(y_err_bot)
            cfill = [x_points, y1_points, y2_points, [], error_op] #was 0.1
            ebars = np.array(ebars)
            print(ebars.shape)
            e_bars_y.append(ebars)
            fill_between.append(cfill)
    else:
        fill_between = []
        e_bars_y = []
    plot1D(data[1], x=data[0], xlabelstring=x_axis_str, ylabelstring=label_dict[quantity], legend=legend, 
           customMarkerStyle=marker_list, customlinewidth=[4 for i in range(len(to_load))], 
           customplot1DLinestyle=["-" for i in range(len(to_load))], customColorList=colour_list,
           logy=logy, logx=logx, saveto=save_path, dataname=dataname, fontsize=fontsize, legendcol=figcols,
            custom_error_y=e_bars_y, errorcapsize=None, fillbetween=fill_between) #fillbetween=fill_between

#%%

if __name__ == "__main__":
    to_load = ["generic_HE", "qg_circuit",  "fermionic", "y_CPHASE", "clifford", "TFIM", "XXZ", "fsim","zfsim",  ]
    cd = load_data(to_load, qubit_range=(2, 10), loc="deep_circuits")
    
    #plot_data(cd, "Expr", fixed_N=4, logy=True, save=False, add_legend=False, errors=True)
    #%%
    plot_data(cd, "Reyni", fixed_N=4, save=False, add_legend=False, errors=True)
    plot_data(cd, "GKP", fixed_N=4, save=False, add_legend=False)
    
    #%%
    plot_data(cd, "Ent", fixed_N=4, save=True, add_legend=True, fontsize=16, figcols=2)
    
    #%%
    cd = load_data(to_load, qubit_range=(2, 11))
    plot_data(cd, "Gradients", fixed_N=6, logy=True, save=True, add_legend=False)
    
    #%%
    cd = load_data(to_load, qubit_range=(2, 13))
    #%%
    plot_data(cd, "QFIM_e-vals", fixed_N=10, logy=True, save=True, add_legend=False)
    
    
    #%%
    cd = load_data(to_load, qubit_range=(2, 7))
    plot_data(cd, "Expr", fixed_N=4, logy=True)
    
    
    
    #%%
    to_load = ["TFIM", "XXZ", "generic_HE", "qg_circuit", "fsim", "fermionic", "y_CPHASE", "zfsim"]
    cd = load_data(to_load, qubit_range=(2, 9))
    plot_data(cd, "Reyni", fixed_depth=1 , save=False, add_legend=False)
    plot_data(cd, "GKP", fixed_depth=1, save=False, add_legend=False)
    
    #%%
    #cd = load_data(to_load, qubit_range=(2, 11))
    plot_data(cd, "Ent", fixed_depth=-1, save=True, add_legend=False)
    
    cd = load_data(to_load, qubit_range=(2, 7))
    plot_data(cd, "Expr", fixed_depth=-1, logy=True, save=True, add_legend=False)
    
    cd = load_data(to_load, qubit_range=(2, 13))
    plot_data(cd, "Gradients", fixed_depth=-1, logy=True, save=True, add_legend=False)
    #%%
    expo_scale = ["XXZ", "generic_HE", "qg_circuit", "fsim", "zfsim"]
    non_expo_scale = ["y_CPHASE", "TFIM", "fermionic"]
    
    for count, circuits in enumerate([non_expo_scale, expo_scale]):
        temp_cd = load_data(circuits, qubit_range=(2, 11))
        temp_data = get_data_array_from_dict(temp_cd, "QFIM_e-vals", fixed_N=0, fixed_d=-1)
        for i, c in enumerate(circuits):
            x = temp_data[0][i]
            logx = np.log(np.array(x))
            y = temp_data[1][i]
            logy = np.log(np.array(y))
            if count == 1:
                log_coeff = np.polyfit(x, logy, deg=1)
            else:
                log_coeff = np.polyfit(logx, logy, deg=1)
            print(f"{c} linear coeffs: {log_coeff[0]:.3f}, {log_coeff[1]:.3f}")
        
        if count == 1:
            plot_data(temp_cd, "QFIM_e-vals", fixed_depth=-1, logy=True, save=True, add_legend=True)
        else:
            plot_data(temp_cd, "QFIM_e-vals", fixed_depth=-1, logy=True, logx=True, save=True, add_legend=True)
    
    #%%
    to_load = ["XXZ", "generic_HE", "qg_circuit", "fsim", "zfsim", "TFIM", "fermionic", "y_CPHASE"]
    cd = load_data(to_load, qubit_range=(2, 11), depth_range=(30, 31), loc="deep_circuits")
    plot_data(cd, "QFIM_e-vals", fixed_depth=-1 , save=True, add_legend=False, logx=True, logy=True, fontsize=12)
    #%%
    plot_data(cd, "QFIM_e-vals", fixed_depth=-1 , save=True, add_legend=False, logy=True, logx=True, fontsize=12, figcols=2)
    new_legend = ["TFIM$\propto N$", "fermionic$\propto N^{2}$", "y_CPHASE$\propto N^{2} - N$", "Others$\propto e^{N}$"]
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=colours["TFIM"], marker=markers["TFIM"], lw=4),
                Line2D([0], [0], color=colours["fermionic"], marker=markers["fermionic"], lw=4),
                Line2D([0], [0], color=colours["y_CPHASE"], marker=markers["y_CPHASE"],lw=4),
                Line2D([0], [0], color="white", marker=None,lw=4),]

    plt.gca().legend(custom_lines, new_legend, fontsize=16)
    #print(plt.gca().get_legend_handles_labels())
    #plt.gca().legend(labels=new_legend, handles=["TFIM", "fermionic", "y_CPHASE"], fontsize=14, ncol=2)
    
    #%%
    to_load = ["XXZ", "generic_HE", "qg_circuit", "fsim", "zfsim"]
    cd = load_data(to_load, qubit_range=(2, 11), depth_range=(30, 31), loc="deep_circuits")
    plot_data(cd, "QFIM_e-vals", fixed_depth=-1 , save=True, add_legend=True, logy=True, fontsize=12)
    #%%
    to_load = ["TFIM", "fermionic", "y_CPHASE"]
    cd = load_data(to_load, qubit_range=(2, 11), depth_range=(30, 31), loc="deep_circuits")
    plot_data(cd, "QFIM_e-vals", fixed_depth=-1 , save=True, add_legend=True, logy=True, logx=True, fontsize=12, figcols=1)
    current_handles, current_labels = plt.gca().get_legend_handles_labels()
    new_legend = ["TFIM$\propto N$", "fermionic$\propto N^{2}$", "y_CPHASE$\propto N^{2} - N$"]
    plt.gca().legend(labels=new_legend, fontsize=16)
    
    #%%
    to_load = ["TFIM", "XXZ", "generic_HE", "qg_circuit", "fsim", "fermionic", "y_CPHASE", "zfsim"]
    cd = load_data(to_load, qubit_range=(2, 11), depth_range=(30, 31), loc="deep_circuits")
    plot_data(cd, "Gradients", fixed_depth=-1 , save=False, add_legend=False, logy=True, fontsize=12)

    #plot_data(cd, "Ent", fixed_depth=-1  , save=True, add_legend=True, fontsize=12)
    #%%
    to_load = ["clifford", "TFIM", "XXZ", "generic_HE", "qg_circuit", "fsim", "fermionic", "y_CPHASE", "zfsim", ]
    cd = load_data(to_load, qubit_range=(2, 9), depth_range=(30, 31), loc="deep_circuits")
    plot_data(cd, "Expr", fixed_depth=-1 , save=True, add_legend=False, fontsize=12, logy=True, errors=True)
    #%%
    to_load = ["clifford", "TFIM", "XXZ", "generic_HE", "qg_circuit", "fsim", "fermionic", "y_CPHASE", "zfsim", ]
    cd = load_data(to_load, qubit_range=(2, 9), depth_range=(30, 31), loc="deep_circuits")
    plot_data(cd, "Reyni", fixed_depth=-1  , save=True, add_legend=False, fontsize=12, errors=True, error_op=0.1)
    plot_data(cd, "Ent", fixed_depth=-1  , save=True, add_legend=False, fontsize=12, errors=True)
    plot_data(cd, "GKP", fixed_depth=-1 , save=True, add_legend=True, fontsize=12)
    
    #%%
    plot_data(cd, "QFIM_e-vals", fixed_depth=-1  , save=True, add_legend=True, fontsize=12, logy=True, logx=True)
    #%%
    cd = load_data(to_load, qubit_range=(2, 7), depth_range=(30, 31), loc="deep_circuits")
    plot_data(cd, "Expr", fixed_depth=-1 , save=True, add_legend=True, fontsize=12, logy=True)
    
    #%%
    to_load = ["TFIM", "XXZ", "generic_HE", "qg_circuit", "fsim", "fermionic", "y_CPHASE", "zfsim"]
    cd = load_data(to_load, qubit_range=(2, 11), depth_range=(1, 17), loc="deep_circuits")
    #%%
    plot_data(cd, "Reyni", fixed_N=4 , save=True, add_legend=True, fontsize=12)
    plot_data(cd, "GKP", fixed_N=4 , save=True, add_legend=True, fontsize=12)
    #plot_data(cd, "Ent", fixed_N=4 , save=True, add_legend=True, fontsize=12)
    #plot_data(cd, "Expr", fixed_N=4 , save=True, add_legend=True, fontsize=12, logy=True)
    
    #plot_data(cd, "GKP", fixed_depth=-1, save=True, add_legend=False)
    #%%
    plot_data(cd, "Gradients", fixed_N=4 , save=True, add_legend=True, fontsize=12, logy=True)