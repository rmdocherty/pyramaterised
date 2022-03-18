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

to_load = [ "clifford", "TFIM", "XXZ", "generic_HE", "qg_circuit", "fsim", "fermionic", "y_CPHASE", "zfsim", "fixed_fsim"]
cd = pt.load_data(to_load, qubit_range=(2, 11), depth_range=(30, 31), loc="deep_circuits")

#%%
qfim_dict = {"clifford": {}, "TFIM": {}, "XXZ": {}, "generic_HE": {}, "qg_circuit": {},
                 "fsim": {}, "fermionic": {}, "y_CPHASE": {}, "zfsim" : {}, "fixed_fsim": {}}

for N in range(2, 11):
    if N == 7:
        qfim_arr = [[]]
    else:
        qfim_arr = pt.get_data_array_from_dict(cd, "QFIM_e-vals_dist", fixed_N=N)[1]
    for count, e in enumerate(qfim_arr):
        c = to_load[count]
        if e == []:
            pass
        else:
            qfim_dict[c][str(N)] = []
            for i in e[0]:
                qfim_dict[c][str(N)] = qfim_dict[c][str(N)] + list(i)
            
#%%
def histo(data, bins=100):
    prob, edges = np.histogram(data, bins=bins, range=(0, 1))#, range=(0, 1) #used to be 1, could be np.amax(F_samples)
    prob = prob / sum(prob) #normalise by sum of prob or length?
    #this F assumes bins go from 0 to 1. Calculate midpoints of bins from np.hist
    F = np.array([(edges[i - 1] + edges[i]) / 2 for i in range(1, len(edges))])
    return prob, F

def log_histo(data, bins=50):
    vals = np.log(data)
    vals = vals[vals > -16]
    mean_val = np.mean(np.exp(vals))
    prob, edges = np.histogram(vals, bins=bins)#, range=(0, 1) #used to be 1, could be np.amax(F_samples)
    prob = prob / sum(prob) #normalise by sum of prob or length?
    #this F assumes bins go from 0 to 1. Calculate midpoints of bins from np.hist
    F = np.array([(edges[i - 1] + edges[i]) / 2 for i in range(1, len(edges))])
    return prob, F, mean_val
#%%

# TFIM_prob, TFIM_F = log_histo(qfim_dict["TFIM"]["6"])
# generic_HE_prob, generic_HE_edges = log_histo(qfim_dict["generic_HE"]["6"])
# y_CPHASE_prob, y_CPHASE_edges = log_histo(qfim_dict["y_CPHASE"]["6"])
# fsim_prob, fsim_edges = log_histo(qfim_dict["fsim"]["6"])
# qg_circuit_prob, qg_circuit_edges = log_histo(qfim_dict["qg_circuit"]["6"])

# plt.plot(generic_HE_edges, generic_HE_prob, color=pt.colours["generic_HE"])
# #axs[0].hist(qfim_dict["generic_HE"]["6"], bins=60, color=pt.colours["generic_HE"])
# plt.plot(TFIM_F, TFIM_prob, color=pt.colours["TFIM"])
# #axs[1].hist(qfim_dict["TFIM"]["6"], bins=60, color=pt.colours["TFIM"])
# plt.plot(y_CPHASE_edges, y_CPHASE_prob, color=pt.colours["y_CPHASE"])
# #axs[2].hist(qfim_dict["y_CPHASE"]["6"], bins=60, color=pt.colours["y_CPHASE"])
# plt.plot(fsim_edges, fsim_prob, color=pt.colours["fsim"])
# plt.plot(qg_circuit_edges, qg_circuit_prob, color=pt.colours["qg_circuit"])

for c, qfim_arr in qfim_dict.items():
    if c in ['clifford', 'fixed_fsim', 'zfsim']:
        pass
    else:
        print(c, qfim_arr.keys())
        data = qfim_arr["8"]
        prob, edges, _ = log_histo(data)
        plt.plot(edges, prob, color=pt.colours[c])
        
#%%
fig, axs = plt.subplots(ncols=3, nrows=3, sharex=True, sharey=True)

for count, c in enumerate(["TFIM", "fermionic","y_CPHASE", "XXZ", "generic_HE", "qg_circuit", "fsim", "zfsim"]):
    for N in range(2, 12, 2):
        data = qfim_dict[c][str(N)]
        prob, edges, mean = log_histo(data)
        area = np.trapz(prob, edges)
        axs[count // 3][count % 3].plot(edges, prob, color=pt.colours[c], alpha = N / 10, label=str(mean))
        axs[count // 3][count % 3].legend()