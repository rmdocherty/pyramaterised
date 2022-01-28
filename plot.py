#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 13:49:13 2022

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
    for i in range(2,10):
        haar = Haar(i)
        haar_m = Measurements(haar)
        magic = haar_m.efficient_measurements(100, expr=False, ent=False, eom=False, GKP=True)
        haars.append(magic['GKP'])
    np.save("data/haar_values.npy", np.array(haars))
else:
    haars = np.load("data/haar_values.npy")

#%%

circuit_data = {"TFIM": [], "XXZ": [], "generic_HE": [], "fsim": [], "fermionic": [], "qg_circuit": [], "zfsim": [], "NPQC":[]} #, "Circuit_9": []

for circuit_type in circuit_data.keys():
    for n_qubits in range(2, 10):
        qubit_data = []
        if n_qubits % 2 == 1 and circuit_type in ["XXZ", "fermionic", "TFIM"]:
            pass
        else:
            for n_layers in range(1, 14):
                file_name = f"data/capacity_3/{circuit_type}_ZZ_{n_qubits}q_{n_layers}l_0r_random.pickle"
                if circuit_type == "NPQC" and n_layers >= (2**(n_qubits // 2)):
                    pass
                else:
                    with open(file_name, 'rb') as file:
                        temp_data = pickle.load(file)
                        qubit_data.append(temp_data)
        circuit_data[circuit_type].append(qubit_data)


#%%
markers = {"TFIM": 'p', "XXZ": 'x', "generic_HE": 'h', "fsim": 'o', "fermionic": 's', "Circuit_9": '$9$', "qg_circuit": '^', "y_CPHASE": '$Y$', "zfsim": '$Z$', "NPQC": 'v'}
colours = {"TFIM": "#5ADBFF", "XXZ": "#235789", "generic_HE": "#C1292E", "fsim": "#F1D302", "fermionic": "#E9BCB7", "qg_circuit": "#139A43", "Circuit_9": "#FF8552", "y_CPHASE": "orange", "NPQC": "#395756", "zfsim": "#A67DB8"}
label_dict = {"Expr": "Expressibility, $D_{KL}$", "Ent": "Entanglement, <Q>", "Reyni": "Reyni Magic relative to Haar random magic", "GKP": "GKP Magic relative to Haar random magic", "Capped_e-vals": "Capped Eigenvalues", "Gradients": "Var{Grad}", "QFIM_e-vals": "Rank"}
title_dict = {"Expr": "Expressibility", "Ent": "Entanglement", "Reyni": "Reyni Entropy of Magic", "GKP": "GKP Magic ", "Capped_e-vals": "Capped Eigenvalues", "Gradients": "Var{Grad}", "QFIM_e-vals": "Rank"}

def quantity_vs_depth(circuit, quantity, N, count_params=False):
    index_n = N - 2
    depths = []
    count = 1
    param_count = 0
    if count_params == True:
        param_count = len(circuit_data[circuit][index_n][0]["Circuit_data"]["Gradients"][0])
    values = []
    stds = []
    for data_dict in circuit_data[circuit][index_n]:
        val = data_dict["Circuit_data"][quantity]
        
        
        if quantity == "Expr":
            dummy = Measurements(PQC(N))
            avg = dummy._expr(val, 2**N)
            std = 0
        elif quantity == "Gradients":
            grad_list = np.array(val).flatten()
            avg = np.var(grad_list)
            std = 0
        elif quantity == "Reyni":
            haar_m = avg_magic = np.log(3 + 2**N) - np.log(4)
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
        
        values.append(avg)
        stds.append(std)

        if count_params is True:
            param_count += len(data_dict["Circuit_data"]["Gradients"][0])
            depths.append(param_count)
        else:
            depths.append(count)
            count += 1

    return (depths, values, stds)


def quantity_vs_N(circuit, quantity, depth, check="last", data_array=circuit_data):
    if check == "fixed":
        index_d = depth - 1
    elif check == "last":
        index_d = -1
    ns = []
    count = 2
    values = []
    stds = []
    for qubit_depth_list in data_array[circuit]:
        if len(qubit_depth_list) > 0:
            if check == "same":
                index_d = count - 2
            data_dict = qubit_depth_list[index_d] #index_d
            val = data_dict["Circuit_data"][quantity]
            if quantity == "Expr" and count < 7:
                dummy = Measurements(PQC(count))
                avg = dummy._expr(val, 2**count)
                std = 0
            elif quantity == "Gradients":
                grad_list = np.array(val).flatten()
                avg = np.var(grad_list)
                std = 0
            elif quantity == "Reyni":
                haar_m = avg_magic = np.log(3 + 2**count) - np.log(4)
                #max_magic = np.log((2**count) + 1) - np.log(2)
                avg = np.mean(val) / haar_m #max_magic
                std = np.std(val) / haar_m #max_magic
            elif quantity == "GKP":
                haar_m = haars[count - 2][0]
                #max_magic = np.log((2**count) + 1) - np.log(2)
                avg = np.mean(val) / haar_m #max_magic
                std = np.std(val) / haar_m #max_magic
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
            
            
            values.append(avg)
            ns.append(count)
            stds.append(std)
            count += 1
        else:
            count += 1
            std = 0
    return (ns, values, stds)


def plot_quantity_vs_depth(quantity, N, count_params=False):
    for keys, items in circuit_data.items():
        depth, values, std = quantity_vs_depth(keys, quantity, N, count_params)
        plt.errorbar(depth, values, yerr=0, color=colours[keys], marker=markers[keys], ms=15, lw=4, label=keys)
        if quantity in ["Expr", "Gradients"]:
            plt.gca().set_yscale("log")
        # elif quantity in ["GKP", "Reyni"]:
        #     plt.ylim((-0.1, 1.1))
        #     plt.xlim((0.5, 13.5))
        quantity_name = label_dict[quantity]
        
        if count_params is True:
            pretty_graph("Number of parameters", quantity_name, f"{title_dict[quantity]} vs number of parameters for {N} qubit PQCs", 20)
            #plt.gca().set_facecolor("#ffefde")
        else:
            pretty_graph("Depth", quantity_name, f"{title_dict[quantity]} vs depth for {N} qubit PQCs", 20)
    plt.legend(fontsize=18)


def plot_quantity_vs_N(quantity, depth, check="last", data_array=circuit_data):
    for keys, items in data_array.items():
        print(keys)
        ns, values, std = quantity_vs_N(keys, quantity, depth, check, data_array)
        #ns = 3 * np.array(ns)
        if quantity == "Gradients" and keys == "XXZ":
            ns = ns[1:]
            values = values[1:]
            std = std[1:]
        elif quantity == "Gradients" and keys == "fermionic":
            ns = ns[1:]
            values = values[1:]
            std = std[1:]
        plt.errorbar(ns, values, yerr=std, color=colours[keys], marker=markers[keys], ms=15, lw=4, label=keys)
        if quantity in ["Expr", "Gradients"]:
            plt.gca().set_yscale("log")
        elif quantity in ["QFIM_e-vals"]:
            plt.gca().set_yscale("log")
            plt.gca().set_xscale("log")
        # elif quantity in ["GKP", "Reyni"]:
        #     plt.ylim((-0.1, 1.1))
        #     plt.xlim((1.5, 9.5))
        
        quantity_name = label_dict[quantity]
        pretty_graph("N", quantity_name, f"{title_dict[quantity]} vs qubit number for l=3N layer PQCs", 20)
        #plt.gca().set_facecolor("#ecdfed")
    plt.legend(fontsize=18)


def plot_quantity_vs_depth_vs_N(circuit, quantity):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title(f"{circuit} {quantity} vs depth and qubit number", fontsize=20)
    all_data = []
    if quantity == "Expr":
        offset = -1
        ax.set_zscale("log")
    elif quantity == "Reyni":
        offset = 1
    else:
        offset = 2
        
    if circuit in ["fermionic", "fsim"]:
        qubit_range = np.arange(2, len(circuit_data[circuit]) + offset, step=2)
    else:
        qubit_range = np.arange(2, len(circuit_data[circuit]) + offset)
    for i in qubit_range:
        out = quantity_vs_depth(circuit, quantity, i)
        all_data.append(out[1])
    depth_range = np.arange(1,14)

    X, Y = np.meshgrid(depth_range, qubit_range)
    Z = np.array(all_data)

    surf = ax.plot_surface(X, Y, Z, color=colours[circuit],
                       linewidth=0, antialiased=True)
    ax.set_xlabel("Depth", fontsize=16)
    ax.set_ylabel("N", fontsize=16)
    ax.set_zlabel(label_dict[quantity], fontsize=16)
    ax.tick_params(labelsize=16)
    plt.show()
        

def two_quantity_vs_colour_map_plot(q1, q2, q3, N, ds):
    plt.title(f"{q1} vs {q2} vs {q3}")
    cmap = cm.get_cmap('viridis')
    index_d = ds - 1
    for circuit, qubit_depth_list in circuit_data.items():
        q1_val = quantity_vs_depth(circuit, q1, N)[1][index_d]
        q2_val = quantity_vs_depth(circuit, q2, N)[1][index_d]
        q3_val = quantity_vs_depth(circuit, q3, N)[1][index_d]
        plt.plot(q1_val, q2_val, marker=markers[circuit], color=cmap(q3_val), ms=25, label=circuit)
    pretty_graph(label_dict[q1], label_dict[q2], f"Descriptor landscape for {N}-qubit {d}-layer PQCs", 20) #{label_dict[q1]} vs {label_dict[q2]} for {N} qubit, {d} layer PQCs
    norm = colors.Normalize(vmin=0, vmax=1)
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
    cbar.set_label(title_dict[q3], fontsize=18)
    cbar.ax.yaxis.set_tick_params(labelsize=18)
    plt.legend(fontsize=18)
    plt.xlim(-0.1, 1.1)
    plt.ylim(10**-4, 10**1)
    if q1 == "Expr":
        plt.gca().set_xscale('log')
    elif q2 == "Expr":
        plt.gca().set_yscale('log')
    plt.grid()
    plt.savefig(f"{q1}_{q2}_{q3}_{N}q_{ds}l.png")

#%%

if __name__ == "__main__":
    for N in [6]:
        for quantity in ["QFIM_e-vals"]: #"Expr", "Ent", "Reyni", "GKP", "Capped_e-vals",
            plt.figure(f"{quantity} vs depth {N}" )
            plot_quantity_vs_depth(quantity, N)
            
    #%%
    for N in [4]:
        for quantity in ["Ent", "Reyni", "GKP", "Capped_e-vals", "Gradients"]:
            plt.figure(f"{quantity} vs depth {N}" )
            plot_quantity_vs_depth(quantity, N)
    
    
    #%%
    for N in [4]:
        for quantity in ["Expr", "Ent", "Reyni", "GKP", "Capped_e-vals", "Gradients"]:
            plt.figure(f"{quantity} vs depth {N}" )
            plot_quantity_vs_depth(quantity, N, count_params=True)
    
    
    #%%
    for d in [4]:
        for quantity in ["QFIM_e-vals"]:
            plt.figure(f"{quantity} vs N, depth" )
            plot_quantity_vs_N(quantity, d)
    
    #%%
    plot_quantity_vs_depth_vs_N("generic_HE", "QFIM_e-vals")
    
    #%%
    for d in range(1, 14):
        plt.figure(f"Descriptor landscape {d}")
        two_quantity_vs_colour_map_plot("Ent", "Expr", "Reyni", 6, d)