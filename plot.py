#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 13:49:13 2022

@author: ronan
"""
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
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

haars = []
for i in range(2,10):
    haar = Haar(i)
    haar_m = Measurements(haar)
    magic = haar_m.efficient_measurements(100, expr=False, ent=False, eom=False, GKP=True)
    haars.append(magic['GKP'])

#%%

circuit_data = {"TFIM": [], "XXZ": [], "generic_HE": [], "fsim": [], "fermionic": [], "qg_circuit": []} #, "Circuit_9": []

for circuit_type in circuit_data.keys():
    for n_qubits in range(2, 10):
        qubit_data = []
        if n_qubits % 2 == 1 and circuit_type in ["fsim", "fermionic"]:
            pass
        else:
            for n_layers in range(1, 14):
                file_name = f"data/capacity_3/{circuit_type}_ZZ_{n_qubits}q_{n_layers}l_0r_random.pickle"
                with open(file_name, 'rb') as file:
                    temp_data = pickle.load(file)
                    qubit_data.append(temp_data)
        circuit_data[circuit_type].append(qubit_data)


#%%
markers = {"TFIM": 'p', "XXZ": 'x', "generic_HE": 'h', "fsim": 'o', "fermionic": 's', "Circuit_9": '$9$', "qg_circuit": '^'}
colours = {"TFIM": "#5ADBFF", "XXZ": "#235789", "generic_HE": "#C1292E", "fsim": "#F1D302", "fermionic": "#E9BCB7", "qg_circuit": "#139A43", "Circuit_9": "#FF8552"}
label_dict = {"Expr": "Expressibility", "Ent": "Entanglement", "Reyni": "Reyni Magic relative to Haar random magic", "GKP": "GKP Magic relative to Haar random magic", "Capped_e-vals": "Capped Eigenvalues", "Gradients": "Var{Grad}"}

def quantity_vs_depth(circuit, quantity, N):
    index_n = N - 2
    depths = []
    count = 1
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
            #max_magic = np.log((2**N) + 1) - np.log(2)
            avg = np.mean(val) / haar_m # max_magic
            std = np.std(val) / haar_m #max_magic
        elif quantity == "GKP":
            haar_m = haars[index_n][0]
            #max_magic = np.log((2**count) + 1) - np.log(2)
            avg = np.mean(val) / haar_m #max_magic
            std = np.std(val) / haar_m #max_magic
        else:
            avg = np.mean(val)
            std = np.std(val)
        
        values.append(avg)
        depths.append(count)
        stds.append(std)
        count += 1
    return (depths, values, stds)


def quantity_vs_N(circuit, quantity, depth):
    index_d = depth - 1
    ns = []
    count = 2
    values = []
    stds = []
    for qubit_depth_list in circuit_data[circuit]:
        if len(qubit_depth_list) > 0:
            index_d = count - 2
            data_dict = qubit_depth_list[-1] #index_d
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


def plot_quantity_vs_depth(quantity, N):
    for keys, items in circuit_data.items():
        depth, values, std = quantity_vs_depth(keys, quantity, N)
        plt.errorbar(depth, values, yerr=std, color=colours[keys], marker=markers[keys], ms=15, lw=4, label=keys)
        if quantity in ["Expr", "Gradients"]:
            plt.gca().set_yscale("log")
        elif quantity in ["GKP", "Reyni"]:
            plt.ylim((-0.1, 1.1))
            plt.xlim((0.5, 13.5))
        quantity_name = label_dict[quantity]
        pretty_graph("Depth", quantity_name, f"{quantity_name} vs depth for {N} qubit PQCs", 20)
    plt.legend(fontsize=18)


def plot_quantity_vs_N(quantity, depth):
    for keys, items in circuit_data.items():
        #if keys in ["fermionic", "fsim"]:
        #    return 0
        ns, values, std = quantity_vs_N(keys, quantity, depth)
        plt.errorbar(ns, values, yerr=std, color=colours[keys], marker=markers[keys], ms=15, lw=4, label=keys)
        if quantity in ["Expr", "Gradients"]:
            plt.gca().set_yscale("log")
        elif quantity in ["GKP", "Reyni"]:
            plt.ylim((-0.1, 1.1))
            plt.xlim((1.5, 9.5))
        quantity_name = label_dict[quantity]
        pretty_graph("N", quantity_name, f"{quantity_name} vs qubit number for l=13 layer PQCs", 20)
    plt.legend(fontsize=18)


def plot_quantity_vs_depth_vs_N(circuit, quantity):
    #plt.figure(f"{circuit} {quantity} vs depth and qubit number")
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
    ax.set_facecolor("#fffcf5")
    ax.tick_params(labelsize=16)
    plt.show()
        
    

#%%
for N in [4]:
    plt.figure(f"Expr vs depth {N}" )
    plot_quantity_vs_depth("Expr", N)
    
    plt.figure(f"Ent vs depth {N}" )
    plot_quantity_vs_depth("Ent", N)
    
    plt.figure(f"Reyni vs depth {N}")
    plot_quantity_vs_depth("Reyni", N)
    
    plt.figure(f"GKP vs depth {N}")
    plot_quantity_vs_depth("GKP", N)
    
    plt.figure(f"Capped_e-vals vs depth {N}")
    plot_quantity_vs_depth("Capped_e-vals", N)
    
    plt.figure(f"Var[grad] vs depth {N}")
    plot_quantity_vs_depth("Gradients", N)

#%%
for d in [4]:
    plt.figure(f"Expr vs N, depth" )
    plot_quantity_vs_N("Expr", d)
    
    plt.figure(f"Ent vs N, depth" )
    plot_quantity_vs_N("Ent", d)
    
    plt.figure(f"Reyni vs N, depth ")
    plot_quantity_vs_N("Reyni", d)
    
    plt.figure(f"GKP vs N, depth ")
    plot_quantity_vs_N("GKP", d)
    
    plt.figure(f"Capped_e-vals vs N, depth")
    plot_quantity_vs_N("Capped_e-vals", d)
    
    plt.figure(f"Var[grad] vs N, depth")
    plot_quantity_vs_N("Gradients", d)

#%%
plot_quantity_vs_depth_vs_N("fermionic", "Reyni")