#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 13:48:57 2022

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

#%%
to_load = ["clifford", "TFIM", "XXZ", "generic_HE", "qg_circuit", "fsim", "fermionic", "y_CPHASE", "zfsim", "fixed_fsim"]
cd = pt.load_data(to_load, qubit_range=(2, 11), depth_range=(30, 31), loc="deep_circuits")

fidelity_dict = {"clifford": {}, "TFIM": {}, "XXZ": {}, "generic_HE": {}, "qg_circuit": {},
                 "fsim": {}, "fermionic": {}, "y_CPHASE": {}, "zfsim": {}, "fixed_fsim": {}}

for N in range(2, 11):
    if N == 7:
        expr_arr = [[]]
    else:
        expr_arr = pt.get_data_array_from_dict(cd, "F", fixed_N=N)[1]
    print(N, len(expr_arr[0]))
    for count, e in enumerate(expr_arr):
        c = to_load[count]
        if e == []:
            pass
        else:
            fidelity_dict[c][str(N)] = e[0]

#%%
def histo(F_arr, filt=True):
    F = np.array(F_arr)
    if filt is True:
        F = F[F < 0.1]
    prob, edges = np.histogram(F, bins=int((75 / 10000) * len(F))) #used to be 1, could be np.amax(F_samples)
    prob = prob / sum(prob) #normalise by sum of prob or length?
    F = np.array([(edges[i - 1] + edges[i]) / 2 for i in range(1, len(edges))])
    return prob, F


def expr_fn(F_arr, D, filt=True):
    if len(F_arr) == 0:
        return 0
    P_pqc, F = histo(F_arr, filt=True)
    haar = (D - 1) * (1 - F)**(D - 2)
    P_haar = haar / sum(haar)
    expr = np.sum(scipy.special.kl_div(P_pqc, P_haar))
    plt.figure("F")
    plt.plot(F, P_pqc, label="$P_{PQC}$", lw=2)
    plt.plot(F, P_haar, label="$P_{Haar}$", lw=2)
    plt.xlabel("F", fontsize=18)
    plt.ylabel("Probability", fontsize=18)
    #plt.legend(fontsize=16)
    plt.title("generic_HE fidelity distribution N=4", fontsize=20)
    return expr


def log_histo(F_arr):
    one_minus_F = np.log(np.array(F_arr))
    prob, edges = np.histogram(one_minus_F, bins=int((75 / 10000) * len(one_minus_F))) #used to be 1, could be np.amax(F_samples)
    prob = prob / sum(prob) #normalise by sum of prob or length?
    F = np.array([(edges[i - 1] + edges[i]) / 2 for i in range(1, len(edges))])
    return prob, F


def log_expr(F_arr, D):
    if len(F_arr) == 0:
        return 0
    log_P_pqc, log_F = log_histo(F_arr)
    log_haar = np.log(D - 1) + (D - 2) * np.log(np.exp(log_F))
    log_P_haar = log_haar / sum(log_haar)
    log_expr = np.sum(scipy.special.kl_div(log_P_pqc, log_P_haar))
    plt.figure("log F")
    plt.plot(log_F, log_P_pqc, label="log $P_{PQC}$", lw=2)
    plt.plot(log_F, log_P_haar, label="log $P_{Haar}$", lw=2)
    plt.xlabel("log F", fontsize=18)
    plt.ylabel("Probability", fontsize=18)
    plt.legend(fontsize=16)
    plt.title("TFIM log fidelity distribution N=4", fontsize=20)
    return np.exp(log_expr)

#%%
he_f_arr = fidelity_dict["generic_HE"]["4"]
b = expr_fn(he_f_arr, 2**4)
#a = log_expr(he_f_arr, 8205476155104674)
#b = expr_fn(he_f_arr, 3.8205476155104674)
