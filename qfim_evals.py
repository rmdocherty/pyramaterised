#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 23:08:02 2022

@author: ronan
"""

import scipy
import numpy as np
import qutip as qt
import random
import measure_all as measure
import PQC_lib as pqc
import plot as pt
from t_plot import plot1D
import matplotlib.pyplot as plt
from measurement import Measurements


N = 2

to_load = [ "clifford", "TFIM", "XXZ", "generic_HE", "qg_circuit", "fsim", "fermionic", "y_CPHASE", "zfsim", "fixed_fsim", "double_y_CPHASE"]
cd = pt.load_data(to_load, qubit_range=(2, 11), depth_range=(30, 31), loc="deep_circuits")

#%% QFIM EVAL MEASUREMENTS
qfim_dict = {"clifford": {}, "TFIM": {}, "XXZ": {}, "generic_HE": {}, "qg_circuit": {},
                 "fsim": {}, "fermionic": {}, "y_CPHASE": {}, "zfsim" : {}, "fixed_fsim": {}, "double_y_CPHASE": {}}
n_params_dict = {"clifford": {}, "TFIM": {}, "XXZ": {}, "generic_HE": {}, "qg_circuit": {},
                 "fsim": {}, "fermionic": {}, "y_CPHASE": {}, "zfsim" : {}, "fixed_fsim": {}, "double_y_CPHASE": {}}
n_repeats_dict = {"clifford": {}, "TFIM": {}, "XXZ": {}, "generic_HE": {}, "qg_circuit": {},
                 "fsim": {}, "fermionic": {}, "y_CPHASE": {}, "zfsim" : {}, "fixed_fsim": {}, "double_y_CPHASE": {}}
for N in range(2, 11):
    if N == 7:
        pass
    else:
        qfim_arr = pt.get_data_array_from_dict(cd, "QFIM_e-vals_dist", fixed_N=N)[1]
    for count, e in enumerate(qfim_arr):
        c = to_load[count]
        if N == 7:
            qfim_dict[c][str(N)] = []
        elif e == []:
            qfim_dict[c][str(N)] = []
        else:
            qfim_dict[c][str(N)] = []
            n_repeats_dict[c][str(N)] = len(e[0])
            n_params_dict[c][str(N)] = len(e[0][0])
            for i in e[0]:
                qfim_dict[c][str(N)] = qfim_dict[c][str(N)] + list(i)
            
#%% HISTOGRAM
def histo(data, bins=40, filt=2**-15, reverse=False):
    data=np.array(data)
    if reverse == False:
        data = data[data > filt]
    else:
        data = data[data < filt]
    prob, edges = np.histogram(data, bins=40)#, range=(0, 1) #used to be 1, could be np.amax(F_samples)
    prob = prob / sum(prob) #normalise by sum of prob or length?
    #this F assumes bins go from 0 to 1. Calculate midpoints of bins from np.hist
    F = np.array([(edges[i - 1] + edges[i]) / 2 for i in range(1, len(edges))])
    return prob, F

def log_histo(data, bins=50):
    vals = np.log(data)
    vals = vals[vals > -17]
    mean_val = np.mean(np.exp(vals))
    prob, edges = np.histogram(vals, bins=bins)#, range=(0, 1) #used to be 1, could be np.amax(F_samples)
    prob = prob / sum(prob) #normalise by sum of prob or length?
    #this F assumes bins go from 0 to 1. Calculate midpoints of bins from np.hist
    F = np.array([(edges[i - 1] + edges[i]) / 2 for i in range(1, len(edges))])
    return prob, F, mean_val
#%%

for c, qfim_arr in qfim_dict.items():
    if c in ['clifford', 'fixed_fsim', 'zfsim']:
        pass
    else:
       # print(c, qfim_arr.keys())
        data = qfim_arr["8"]
        prob, edges, _ = log_histo(data)
        plt.plot(edges, prob, color=pt.colours[c])
        
#%% PLOT EVAL DIST FOR EACH CIRCUIT FOR DIFFERENT N
fig, axs = plt.subplots(ncols=4, nrows=1, sharex=True)
fig2, axs2 = plt.subplots(ncols=4, nrows=1, sharex=True)

for N in range(2, 10, 2):
    #fig.text(0.06, 0.5, 'Probability', va='center', rotation='vertical', fontsize=16)
    #fig.text(0.5, 0.04, 'log $\\lambda$', ha='center', fontsize=16)
    for count, c in enumerate(["generic_HE", "qg_circuit", "fsim", "zfsim", "y_CPHASE", "XXZ","TFIM", "fermionic"]):
        data = np.array(qfim_dict[c][str(N)])
        log_prob, log_edges, _ = log_histo(data)
        prob, edges = histo(data, bins=40)
        zero_prob, zero_eges = histo(data, reverse=True, bins=10)
        #prnt(f"{c} {N} = {np.var(data)}")
        #area = np.trapz(prob, edges)
        #axs[(N // 2) - 1].plot(log_edges, log_prob, color=pt.colours[c])
        w = np.ones_like(data[data > 2**-17]) * (1/(30 * N))
        axs[(N // 2) - 1].hist(np.log(data[data > 2**-17]), color=pt.colours[c], bins=30)
        axs[(N // 2) - 1].set_title(f"{N} qubits", fontsize=16)
        axs[(N // 2) - 1].set_xlabel(f"log $\\lambda$, $\\lambda > 0$", fontsize=16)
        axs[(N // 2) - 1].set_ylabel(f"Count", fontsize=16)
        
        
        axs2[(N // 2) - 1].plot(edges, prob, color=pt.colours[c])
        axs2[(N // 2) - 1].plot(zero_eges, zero_prob, color=pt.colours[c])
        axs2[(N // 2) - 1].set_title(f"{N} qubits")
        axs2[(N // 2) - 1].set_ylim(0, 0.3)

#%% PLOT E-VAL VARIANCES AS FUNCTION OF N FOR EACH CIRCUIT
c_N_var = []
c_N_mean = []
c_N_q = []

measure_names = ["Variance of $\\lambda$", "Mean of non-zero $\\lambda$",]

colours = []
markers = []

for count, c in enumerate(["TFIM", "fermionic","y_CPHASE", "XXZ", "generic_HE", "qg_circuit", "fsim", "zfsim"]):
    c_var = []
    c_mean = []
    #c_q = []
    for N in range(2, 10, 2):
        data = qfim_dict[c][str(N)]
        data = np.array(data)
        #c_var.append(np.var(data))
        n_param = n_params_dict[c][str(N)]
        #c_q.append(n_param)
        c_var.append(np.var(data))
        data = data[data > 2**-10]
        
        c_mean.append(np.mean(data))
        prob, edges, _ = log_histo(data)
    
    c_q = [i for i in range(2, 10, 2)]
    if c in ["fermionic", "XXZ"]:
        c_q = c_q[1:]
        c_var = c_var[1:]
        c_mean = c_mean[1:]

    c_N_var.append(c_var)
    c_N_mean.append(c_mean)
    c_N_q.append(c_q)
    colours.append(pt.colours[c])
    markers.append(pt.markers[c])


ls = ["-" for i in range(len(c_N_var))]
lws = [4 for i in range(len(c_N_var))]
ylabels = ["var{$\\lambda$}", "<$\\lambda$>", "a"]

for count, measure in enumerate([c_N_var, c_N_mean]):
    plot1D(measure, x=c_N_q, xlabelstring="N", ylabelstring=ylabels[count], legend=[], 
           customMarkerStyle=markers, customlinewidth=lws, 
           customplot1DLinestyle=ls, customColorList=colours,
           saveto="plots", dataname="", fontsize=14, legendcol=2, logy=True, logx=True) #dataname=eff_dim_h_up_to_10
    plt.gca().set_title(measure_names[count], fontsize=20)
    
    

#%% MEASURE EVAL VARIANCE OF RANDOM MATRIX AS FUNCTION OF MATRIX VARAINCE (NORMAL DIST)
var_space = np.logspace(-4, 4, 100)
eig_val_var = []

for var in var_space:
    current_eval_var = []
    for i in range(50):
        dim = 2**5
        rand_mat = var * np.random.randn(dim, dim)
        eigenvals, eigvects = np.linalg.eigh(rand_mat)
        current_eval_var.append(np.var(eigenvals))
    eig_val_var.append(np.mean(current_eval_var))

plt.figure()
plt.title("Var{$\\lambda$} vs $\\sigma$ for $M_{ij} \\sim \\mathcal{G}(0; \\sigma)$", fontsize=20)
plt.loglog(var_space, eig_val_var, lw=3)
plt.xlabel("$\\sigma$", fontsize=18)
plt.ylabel("Var{$\\lambda$}", fontsize=18)

#%% LOAD GRADIENTS
grad_dict = {"clifford": {}, "TFIM": {}, "XXZ": {}, "generic_HE": {}, "qg_circuit": {},
                 "fsim": {}, "fermionic": {}, "y_CPHASE": {}, "zfsim" : {}, "fixed_fsim": {}}

for N in range(2, 11):
    if N == 7:
        grad_arr = [[]*len(grad_dict.keys())]
    else:
        grad_arr = pt.get_data_array_from_dict(cd, "Gradients", fixed_N=N)[1]
        for count, c in enumerate(grad_dict.keys()):
            #c = to_load[count]
            grad_dict[c][str(N)] = grad_arr[count]


#%% PLOT TRACE OF QFIM AS FN OF PARAMETER NUMBER
trs = []
n_params = []
count2 = 0
colors = [] 
marker = []
for count, c in enumerate(["fermionic","y_CPHASE", "generic_HE", "qg_circuit", "fsim", "zfsim", "double_y_CPHASE"]):
    c_trs = []
    c_n_params = []
    for N in range(2, 10, 2):
        data = qfim_dict[c][str(N)]
        n_repeat = n_repeats_dict[c][str(N)]
        data = np.reshape(data, (len(data) // n_repeat, n_repeat))
        n_param = n_params_dict[c][str(N)]
        tr = np.sum(data, axis = 0)
        #print(tr.shape)
        tr = np.mean(tr)
        c_trs.append(tr)
        c_n_params.append(n_params)
    trs.append(c_trs)
    n_params.append(c_n_params)
    colors.append(pt.colours[c])
    marker.append(pt.markers[c])

n_qubits = [range(2, 10, 2) for i in range(len(c_N_var))]
ls = ["-" for i in range(len(c_N_var))]
lws = [4 for i in range(len(c_N_var))]

plot1D(trs , x=n_params, xlabelstring="N params", ylabelstring="tr($\\mathcal{F}$)", legend=[], 
       customMarkerStyle=marker, customlinewidth=lws, 
       customplot1DLinestyle=ls, customColorList=colors,
       saveto="plots", dataname="", fontsize=14, legendcol=2) #dataname=eff_dim_h_up_to_10

#%%
from matplotlib.lines import Line2D
plt.figure("LEgend")
new_lines = []
labels = ["generic_HE", "qg_circuit", "y_CPHASE", "clifford" ,"fsim", "TFIM", "XXZ", "fermionic", "zfsim", "fixed_fsim" ]
for c in labels:
    line = Line2D([0], [0], color=pt.colours[c], marker=pt.markers[c], lw=5, ms=10)
    new_lines.append(line)
plt.gca().legend(new_lines, labels, fontsize=16)
