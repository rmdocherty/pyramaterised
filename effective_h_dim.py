#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 20:08:54 2022

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

to_load = ["TFIM", "XXZ", "generic_HE", "qg_circuit", "fsim", "fermionic", "y_CPHASE", "zfsim"]
cd = pt.load_data(to_load, qubit_range=(2, 10), depth_range=(30, 31), loc="deep_circuits")
ecd = pt.load_data(to_load, qubit_range=(8, 11), depth_range=(50, 51), loc="exprs")
expr_arr = pt.get_data_array_from_dict(cd, "F", fixed_N=N)[1]


#%%
fidelity_dict = {"TFIM": {}, "XXZ": {}, "generic_HE": {}, "qg_circuit": {},
                 "fsim": {}, "fermionic": {}, "y_CPHASE": {}, "zfsim" : {}}

for N in range(2, 11):
    print(N)
    if N < 7:
        expr_arr = pt.get_data_array_from_dict(cd, "F", fixed_N=N)[1]
    elif N == 7:
        expr_arr = [[]]
    elif N > 7:
        expr_arr = pt.get_data_array_from_dict(ecd, "F", fixed_N=N-6)[1]
    for count, e in enumerate(expr_arr):
        c = to_load[count]
        if e == []:
            pass
        else:
            fidelity_dict[c][str(N)] = e[0]


#%%
def wrapper(n, F_samples=[], m=None):
    return m._expr(F_samples, n)

def log_wrapper(n, F_samples=[], m=None):
    return m._log_expr(F_samples, n)

def find_eff_H(circuit_f_samples, n):
    dummy_circuit = pqc.PQC(n)
    dummy_m = Measurements(dummy_circuit)
    if n > 7:
        wrap_fn = log_wrapper
    else:
        wrap_fn = wrapper
    out = scipy.optimize.minimize(wrap_fn, [4], args=(circuit_f_samples, dummy_m), method="BFGS")
    if out.success is True:
        return out.x
    else:
        return "Fail!"

#%%
eff_hs = []
all_ns = []
all_dns = []
all_diffs = []
colours = []
dcolours = []
markers = []
dmarkers = []
lws = []
ls = []

for c, n_f_dict in fidelity_dict.items():
    c_eff_hs = []
    ns = []
    diffs = []
    for str_n, f_arr in n_f_dict.items():
        n = int(str_n)
        eff_H = find_eff_H(f_arr, n)
        print(f"{c} eff H = {eff_H} for {n} qubits")
        c_eff_hs.append(np.log2(eff_H))
        diffs.append(n - np.log2(eff_H))
        ns.append(n)
    eff_hs.append(c_eff_hs)
    all_ns.append(ns)
    if c in ["generic_HE", "qg_circuit", "fsim", "y_CPHASE"]:
        all_diffs.append(diffs)
        all_dns.append(ns)
        dcolours.append(pt.colours[c])
        dmarkers.append(pt.markers[c])
    colours.append(pt.colours[c])
    markers.append(pt.markers[c])
    lws.append(4)
    ls.append("-")
    #plt.plot(ns, np.log(c_eff_hs), color=pt.colours[c], marker=pt.markers[c], lw=4, label=c)

#%%
plot1D(eff_hs, x=all_ns, xlabelstring="N", ylabelstring="log$_2$ dim $\\mathcal{H}_{eff}$", legend=fidelity_dict.keys(), 
       customMarkerStyle=markers, customlinewidth=lws, 
       customplot1DLinestyle=ls, customColorList=colours,
       saveto="plots", dataname="eff_dim_h_up_to_10", fontsize=14, legendcol=2)
#%%
plot1D(all_diffs, x=all_dns, xlabelstring="N", ylabelstring="N - log$_2$ dim $\\mathcal{H}_{eff}$", legend=["fsim", "generic_HE", "qg_circuit", "y_CPHASE"], 
       customMarkerStyle=dmarkers, customlinewidth=lws[:4], 
       customplot1DLinestyle=ls[:4], customColorList=dcolours,
       saveto="plots", dataname="eff_dim_diffs", fontsize=14, legendcol=2, logy=True)



#%%
plt.hist(fidelity_dict["y_CPHASE"]["10"], density=True, bins=2000)
plt.figure()
plt.hist(np.log(fidelity_dict["y_CPHASE"]["10"]), density=True, bins=2000)

#%%
c_eff_hs = []
for c in ["fermionic", "zfsim"]:
    eff_hs = []
    for p in range(1, 10):
        f_arr = np.load(f"data/exprs/{c}_{p}_10_expr_measurements.npy")
        eff_H = find_eff_H(f_arr, 10)
        eff_hs.append(eff_H[0])
    vals = np.arange(1, 10)
    print(type(eff_hs[0]))
    c_eff_hs.append(eff_hs)
    #plt.plot(vals, eff_hs, color=pt.colours[c], marker=pt.markers[c], lw=4, label=c)
#plt.plot(vals, scipy.special.comb(10, vals))
c_eff_hs.append(scipy.special.comb(10, vals))
x_data = [vals, vals, vals]
markers = [pt.markers["fermionic"], pt.markers["zfsim"], "."]
colours = [pt.colours["fermionic"], pt.colours["zfsim"], "red"]

plot1D(c_eff_hs, x=x_data, xlabelstring="K", ylabelstring="dim $\\mathcal{H}_{eff}$", legend=["fermionic", "zfsim", "N choose k"], 
       customMarkerStyle=markers, customlinewidth=[5, 5, 1], 
       customplot1DLinestyle=["-", "-", "--"], customColorList=colours,
       saveto="plots", dataname="combinatorial", fontsize=18, legendcol=1)

#%%

#%%
# for n in [2, 4, 6]:
#     for i, c in enumerate(to_load):    
#         expr_arr = pt.get_data_array_from_dict(cd, "F", fixed_N=n)[1]
#         eff_H = find_eff_H(expr_arr[i][0], n)
#         print(f"{c} eff H = {eff_H} for {N} qubits")


#%%
for i, c in enumerate(to_load):
    c_eff_hs = []
    for n in [2, 4, 6]:
        expr_arr = pt.get_data_array_from_dict(cd, "F", fixed_N=n)[1]
        eff_H = find_eff_H(expr_arr[i][0], n)
        print(f"{c} eff H = {eff_H}")
        c_eff_hs.append(eff_H)
    plt.plot([2, 4, 6], np.log(c_eff_hs) , color=pt.colours[c], marker=pt.markers[c], lw=4, label=c)

plt.legend()

#%%


