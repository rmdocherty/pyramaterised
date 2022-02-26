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

to_load = [ "clifford", "TFIM", "XXZ", "generic_HE", "qg_circuit", "fsim", "fermionic", "y_CPHASE", "zfsim",]
cd = pt.load_data(to_load, qubit_range=(2, 11), depth_range=(30, 31), loc="deep_circuits")
#ecd = pt.load_data(to_load, qubit_range=(8, 9), depth_range=(30, 31), loc="exprs")
expr_arr = pt.get_data_array_from_dict(cd, "F", fixed_N=N)[1]


#%%
fidelity_dict = {"clifford": {}, "TFIM": {}, "XXZ": {}, "generic_HE": {}, "qg_circuit": {},
                 "fsim": {}, "fermionic": {}, "y_CPHASE": {}, "zfsim" : {}}

for N in range(2, 11):
    if N == 7:
        expr_arr = [[]]
    else:
        expr_arr = pt.get_data_array_from_dict(cd, "F", fixed_N=N)[1]
    print(N, len(expr_arr[0]))
    # if N < 7:
    #     expr_arr = pt.get_data_array_from_dict(cd, "F", fixed_N=N)[1]
    # elif N == 7:
    #     expr_arr = [[]]
    # elif N > 7:
    #     expr_arr = pt.get_data_array_from_dict(cd, "F", fixed_N=N)[1]
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

def find_eff_H(circuit_f_samples, n, expr=False):
    dummy_circuit = pqc.PQC(n)
    dummy_m = Measurements(dummy_circuit)
    if n > 7: #change this later 
        wrap_fn = log_wrapper
    else:
        wrap_fn = wrapper
    out = scipy.optimize.minimize(wrap_fn, [4], args=(circuit_f_samples, dummy_m), method="BFGS") #Nelder-Mead
    if out.success is True and expr is True:
        return out.x, np.exp(dummy_m._log_expr(circuit_f_samples, 2**n))
    elif out.success is True:
        return out.x
    else:
        print(out)
        return "Fail"

#%%
eff_hs = []
all_ns = []
all_dns = []
all_exprs = []
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
    exprs = []
    for str_n, f_arr in n_f_dict.items():
        n = int(str_n)
        eff_H, expr = find_eff_H(f_arr, n, expr=True)
        print(f"{c} eff H = {eff_H}, expr = {expr} for {n} qubits")
        c_eff_hs.append(eff_H)
        exprs.append(expr)
        #diffs.append(n - eff_H)
        ns.append(n)
    eff_hs.append(c_eff_hs)
    all_ns.append(ns)
    all_exprs.append(exprs)
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
plot1D(eff_hs, x=all_ns, xlabelstring="N", ylabelstring="dim $\\mathcal{H}_{eff}$", legend=[], 
       customMarkerStyle=markers, customlinewidth=lws, 
       customplot1DLinestyle=ls, customColorList=colours,
       saveto="plots", dataname="eff_dim_h_up_to_10", fontsize=14, legendcol=2, logy=True)

#%%
# problem: load data is loading the 30q samples that odnt have expr measurements - soln: copy the 10 qubit measuremnts you want itno expr and use the ecd solution you used to have!
plot1D(all_exprs, x=all_ns, xlabelstring="N", ylabelstring="Expr, $D_{KL}$", legend=fidelity_dict.keys(), 
       customMarkerStyle=markers, customlinewidth=lws, 
       customplot1DLinestyle=ls, customColorList=colours,
       saveto="plots", dataname="exprs", fontsize=14, legendcol=2, logy=True)

#%%
plot1D(all_exprs, x=eff_hs, xlabelstring="dim $\\mathcal{H}_{eff}$", ylabelstring="Expr, $D_{KL}$", legend=[], 
       customMarkerStyle=markers, customlinewidth=[None for i in range(len(fidelity_dict.keys()))], 
       customplot1DLinestyle=[None for i in range(len(fidelity_dict.keys()))], customColorList=colours,
       saveto="plots", dataname="scatter", fontsize=14, legendcol=2)


#%%
plot1D(all_diffs, x=all_dns, xlabelstring="N", ylabelstring="N - log$_2$ dim $\\mathcal{H}_{eff}$", legend=["fsim", "generic_HE", "qg_circuit", "y_CPHASE"], 
       customMarkerStyle=dmarkers, customlinewidth=lws[:4], 
       customplot1DLinestyle=ls[:4], customColorList=dcolours,
       saveto="plots", dataname="eff_dim_diffs", fontsize=14, legendcol=2)



#%%
plt.figure()
F_samples = fidelity_dict["clifford"]["4"]
plt.hist(F_samples, bins=int((75 / 10000) * len(F_samples)))
#%%
TFIM_eff_H_10 = eff_hs[0][4]
F_range = np.linspace(0, 1, 4000)
haar = (TFIM_eff_H_10 - 1) * ((1 - F_range) ** (TFIM_eff_H_10 - 2)) #from definition in expr paper
plt.plot(F_range, haar)
#plt.hist(np.log(fidelity_dict["y_CPHASE"]["10"]), density=True, bins=2000)
#%%
dummy_m = Measurements(pqc.PQC(10))
dummy_m._gen_log_histo(F_samples)
dummy_m._expr(F_samples, 2**10)
out = dummy_m._log_expr(F_samples, 2**10)
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
#%%
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


