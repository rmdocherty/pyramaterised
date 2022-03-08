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

to_load = [ "clifford", "TFIM", "XXZ", "generic_HE", "qg_circuit", "fsim", "fermionic", "y_CPHASE", "zfsim", "fixed_fsim"]
cd = pt.load_data(to_load, qubit_range=(2, 11), depth_range=(30, 31), loc="deep_circuits")
#ecd = pt.load_data(to_load, qubit_range=(8, 9), depth_range=(30, 31), loc="exprs")
expr_arr = pt.get_data_array_from_dict(cd, "F", fixed_N=N)[1]


#%%
fidelity_dict = {"clifford": {}, "TFIM": {}, "XXZ": {}, "generic_HE": {}, "qg_circuit": {},
                 "fsim": {}, "fermionic": {}, "y_CPHASE": {}, "zfsim" : {}, "fixed_fsim": {}}

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
def wrapper(n, F_samples=[], m=None):
    return m._expr(F_samples, n, filt=0.2)

def log_wrapper(n, F_samples=[], m=None):
    return m._log_expr(F_samples, n)

def find_eff_H(circuit_f_samples, n, expr=False):
    dummy_circuit = pqc.PQC(n)
    dummy_m = Measurements(dummy_circuit)
    if n > 10: #change this later 
        wrap_fn = log_wrapper
    else:
        wrap_fn = wrapper
    out = scipy.optimize.minimize(wrap_fn, [4], args=(circuit_f_samples, dummy_m), method="BFGS") #Nelder-Mead
    if out.success is True and expr is True:
        return out.x[0], dummy_m._expr(circuit_f_samples, 2**n)
    elif out.success is True:
        return out.x[0]
    else:
        print(out)
        return 0
    
def average_F(F_samples, t=2): #only put to t-th power (not 2t-th) as our fidelity is already squared
    return np.mean(np.array(F_samples)**(t))

def welch(D, t=2):
    return 1/(scipy.special.comb(D + t - 1, t))

def welch_wrapper(D, F_samples=[], t=2):
    w = welch(D, t)
    avg_F = average_F(F_samples, t) 
    return np.abs(avg_F - w )

def find_welch_bound(circuit_f_samples, t, n):
    out = scipy.optimize.minimize(welch_wrapper, [n], args=(circuit_f_samples, t), method="Nelder-Mead")
    if out.success is True:
        return out.x[0]
    else:
        print(out)
        return 0

#%%
eff_hs = []
ds = []
all_ns = []
all_dns = []
all_exprs = []
all_adjusted_exprs = []
all_diffs = []
colours = []
dcolours = []
markers = []
dmarkers = []
lws = []
ls = []
all_stds = []

find_welch = True
find_eff_h = True
find_errors = False

for c, n_f_dict in fidelity_dict.items():
    c_eff_hs = []
    c_ds = []
    ns = []
    diffs = []
    exprs = []
    eff_exprs = []
    c_stds = []
    for str_n, f_arr in n_f_dict.items():
        n = int(str_n)
        if find_welch is True and find_eff_h is True:
            d = find_welch_bound(f_arr, 2, n)
            eff_H, expr = find_eff_H(f_arr, n, expr=True)
            dummy = Measurements(n)
            adjusted_expr = dummy._expr(f_arr, eff_H)
            if find_errors is True:
                std = pt.resample(f_arr, 20, 10000, d, dummy)
                c_stds.append(std)
        elif find_eff_h is True:
            eff_H, expr = find_eff_H(f_arr, n, expr=True)
            d = 0
            adjusted_expr = Measurements(n)._expr(f_arr, eff_H)
        else:
            d = find_welch_bound(f_arr, 2, n)
            eff_H, expr = 0
            adjusted_expr = Measurements(n)._expr(f_arr, d)
        print(f"{c} eff H = {eff_H}, d = {d}, expr = {expr} for {n} qubits")
        c_eff_hs.append(eff_H)
        c_ds.append(d)
        exprs.append(expr)
        eff_exprs.append(adjusted_expr)
        #diffs.append(n - eff_H)
        ns.append(n)
    eff_hs.append(c_eff_hs)
    all_adjusted_exprs.append(eff_exprs)
    ds.append(c_ds)
    all_ns.append(ns)
    all_exprs.append(exprs)
    all_stds.append(c_stds)
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
       saveto="plots", dataname="", fontsize=14, legendcol=2, logy=True) #dataname=eff_dim_h_up_to_10

N_range = np.arange(2, 14, 2)
exp = 2**N_range
plt.gca().plot(N_range, exp, label="$\\mathcal{H}_{\mathrm{full}} = 2^{N}$", color="#63768D", ls="--", lw=1)
nck = scipy.special.comb(N_range, N_range // 2)
plt.gca().plot(N_range, nck, label="$\\mathcal{H}_{\mathrm{sym}} = \\mathrm{N\:choose\:k}$", color="red", ls="--", lw=1)

plt.xlim(1.5, 10.5)
plt.legend(fontsize=18)
#%%
plot1D(ds, x=all_ns, xlabelstring="N", ylabelstring="D", legend=[], 
       customMarkerStyle=markers, customlinewidth=lws, 
       customplot1DLinestyle=ls, customColorList=colours,
       saveto="plots", dataname="", fontsize=14, legendcol=2, logy=True) #dataname=eff_dim_h_up_to_10

N_range = np.arange(2, 14, 2)
exp = 2**N_range
plt.gca().plot(N_range, exp, label="$\\mathcal{H}_{\mathrm{full}} = 2^{N}$", color="#63768D", ls="--", lw=1)
nck = scipy.special.comb(N_range, N_range // 2)
plt.gca().plot(N_range, nck, label="$\\mathcal{H}_{\mathrm{sym}} = \\mathrm{N\:choose\:k}$", color="red", ls="--", lw=1)
#plt.gca().plot(N_range, (4/3)**( 3 * N_range), label="$W^{N}$", color="green", ls="--", lw=1)
plt.xlim(1.5, 10.5)
plt.legend(fontsize=18)

#%%
# problem: load data is loading the 30q samples that odnt have expr measurements - soln: copy the 10 qubit measuremnts you want itno expr and use the ecd solution you used to have!
plot1D(all_exprs, x=all_ns, xlabelstring="N", ylabelstring="Expr", legend=fidelity_dict.keys(), 
       customMarkerStyle=markers, customlinewidth=lws, 
       customplot1DLinestyle=ls, customColorList=colours,
       saveto="plots", dataname="exprs", fontsize=14, legendcol=2, logy=True)
#%%
# problem: load data is loading the 30q samples that odnt have expr measurements - soln: copy the 10 qubit measuremnts you want itno expr and use the ecd solution you used to have!
plot1D(all_adjusted_exprs, x=all_ns, xlabelstring="N", ylabelstring="Eff Expr", legend=fidelity_dict.keys(), 
        customMarkerStyle=markers, customlinewidth=lws, 
        customplot1DLinestyle=ls, customColorList=colours,
        saveto="plots", dataname="exprs", fontsize=14, legendcol=2, logy=True, custom_error_y=all_stds)


#%%
plot1D(eff_hs, x=all_ns, xlabelstring="N", ylabelstring="D", legend=[], 
       customMarkerStyle=markers, customlinewidth=[None for i in range(len(fidelity_dict.keys()))], 
       customplot1DLinestyle=[None for i in range(len(fidelity_dict.keys()))], customColorList=colours,
       saveto="plots", dataname="", fontsize=14, legendcol=2, logy=True)


#%%
plot1D(all_diffs, x=all_dns, xlabelstring="N", ylabelstring="N - log$_2$ dim $\\mathcal{H}_{eff}$", legend=["fsim", "generic_HE", "qg_circuit", "y_CPHASE"], 
       customMarkerStyle=dmarkers, customlinewidth=lws[:4], 
       customplot1DLinestyle=ls[:4], customColorList=dcolours,
       saveto="plots", dataname="", fontsize=14, legendcol=2)



#%%
plt.figure("4 q plot")
N = 2**4
plt.title("Clifford Circuit fidelity dist., N = 4", fontsize=20)
F_samples = fidelity_dict["clifford"]["4"]
prob, edges = np.histogram(F_samples, bins=int((75 / 10000) * len(F_samples)), range=(0,1)) #int((75 / 10000) * len(F_samples))
plt.plot(edges[:-1], prob/np.amax(prob), lw=3, label="Fidelities")
haar = (N - 1) * ((1 - edges) ** (N - 2))
plt.plot(edges[:-1], haar[:-1]/np.amax(haar), ls="--", lw=3, label="Haar / Max(Haar)")
points = np.append(2.0**(-1*np.arange(0, 5)), 0)
plt.plot(points, [0 for i in range(len(points))], marker="*", ms=10, ls="None", lw=0, color="red", label="$0, 2^{-n}, n \in 0, .., N$")
#plt.figure("4 q hist")
#plt.hist(F_samples, bins=int((75 / 10000) * len(F_samples)))
plt.xlabel("Fidelity", fontsize=18)
plt.ylabel("Probability", fontsize=18)
plt.legend(fontsize=16)


plt.figure("4 q plot generic")
N = 2**4
plt.title("generic_HE Circuit fidelity dist., N = 4", fontsize=20)
F_samples = fidelity_dict["generic_HE"]["4"]
prob, edges = np.histogram(F_samples, bins=int((75 / 10000) * len(F_samples)), range=(0,1)) #int((75 / 10000) * len(F_samples))
plt.plot(edges[:-1], prob/np.amax(prob), lw=3, label="Fidelities")
haar = (N - 1) * ((1 - edges) ** (N - 2))
plt.plot(edges[:-1], haar[:-1]/np.amax(haar), ls="--", lw=3, label="Haar / Max(Haar)")
#points = np.append(2.0**(-1*np.arange(0, 5)), 0)
#plt.plot(points, [0 for i in range(len(points))], marker="*", ms=10, ls="None", lw=0, color="red", label="$0, 2^{-n}, n \in 0, .., N$")
#plt.figure("4 q hist")
#plt.hist(F_samples, bins=int((75 / 10000) * len(F_samples)))
plt.xlabel("Fidelity", fontsize=18)
plt.ylabel("Probability", fontsize=18)
plt.legend(fontsize=16)


plt.figure("6 q plot")
plt.title("Clifford Circuit fidelity dist., N = 6", fontsize=20)
N = 2** 6
F_samples = fidelity_dict["clifford"]["6"]
prob, edges = np.histogram(F_samples, bins=len(F_samples), range=(0,1)) #int((75 / 10000) * len(F_samples))
plt.plot(edges[:-1], prob/np.amax(prob), lw=3, label="Fidelities")
haar = (N - 1) * ((1 - edges) ** (N - 2))
plt.plot(edges[:-1], haar[:-1]/np.amax(haar), ls="--", lw=3, label="Haar / Max(Haar)")
points = np.append(2.0**(-1*np.arange(0, 7)), 0)
plt.plot(points, [0 for i in range(len(points))], marker="*", ms=10, ls="None", lw=0, color="red", label="$0, 2^{-n}, n \in 0, .., N$")
#plt.figure("4 q hist")
#plt.hist(F_samples, bins=int((75 / 10000) * len(F_samples)))
plt.xlabel("Fidelity", fontsize=18)
plt.ylabel("Probability", fontsize=18)
plt.legend(fontsize=16)

plt.figure("8 q plot")
plt.title("Clifford Circuit fidelity dist., N = 8", fontsize=20)
N = 2** 8
F_samples = fidelity_dict["clifford"]["8"]
prob, edges = np.histogram(F_samples, bins=len(F_samples), range=(0,1)) #int((75 / 10000) * len(F_samples))
plt.plot(edges[:-1], prob/np.amax(prob), lw=3, label="Fidelities")
haar = (N - 1) * ((1 - edges) ** (N - 2))
plt.plot(edges[:-1], haar[:-1]/np.amax(haar), ls="--", lw=3, label="Haar / Max(Haar)")
points = np.append(2.0**(-1*np.arange(0, 9)), 0)
plt.plot(points, [0 for i in range(len(points))], marker="*", ms=10, ls="None", lw=0, color="red", label="$0, 2^{-n}, n \in 0, .., N$")
#plt.figure("4 q hist")
#plt.hist(F_samples, bins=int((75 / 10000) * len(F_samples)))
plt.xlabel("Fidelity", fontsize=18)
plt.ylabel("Probability", fontsize=18)
plt.legend(fontsize=16)
#%%
TFIM_eff_H_10 = eff_hs[1][1]
F_range = np.linspace(0, 1, 4000)

f = fidelity_dict["TFIM"]["4"]
dummy = Measurements(4)
prob, edges = dummy._gen_histo(f, filt=0.2)

haar = (TFIM_eff_H_10 - 1) * ((1 - edges) ** (TFIM_eff_H_10 - 2)) #from definition in expr paper
haar = haar / max(haar)

plt.plot(edges, prob, label='TFIM')
plt.plot(edges, haar, label='Haar')
plt.legend()
#plt.hist(np.log(fidelity_dict["y_CPHASE"]["10"]), density=True, bins=2000)
#%%

plt.figure("6 q plot")
plt.title("Generic HE Circuit fidelity dist., N = 6", fontsize=20)
N = 2** 6
F_samples = fidelity_dict["generic_HE"]["6"]
prob, edges = np.histogram(F_samples, bins=int((75 / 10000) * len(F_samples)), range=(0,1)) #int((75 / 10000) * len(F_samples))
plt.plot(edges[:-1], prob/np.amax(prob), lw=1, label="Fidelities")
haar = (N - 1) * ((1 - edges) ** (N - 2))
plt.plot(edges[:-1], haar[:-1]/np.amax(haar), ls="--", lw=3, label="Haar / Max(Haar)")

#plt.hist(F_samples, bins=int((75 / 10000) * len(F_samples)))
plt.xlabel("Fidelity", fontsize=18)
plt.ylabel("Probability", fontsize=18)
plt.legend(fontsize=16)

plt.figure("6 q TFIM plot")
plt.title("TFIM  Circuit fidelity dist., N = 6", fontsize=20)
N = eff_hs[1][2]
F_samples = fidelity_dict["TFIM"]["6"]
prob, edges = np.histogram(F_samples, bins=int((75 / 10000) * len(F_samples)), range=(0,1)) #int((75 / 10000) * len(F_samples))
plt.plot(edges[:-1], prob/np.amax(prob), lw=1, label="Fidelities")
haar = (N - 1) * ((1 - edges) ** (N - 2))
plt.plot(edges[:-1], haar[:-1]/np.amax(haar), ls="--", lw=3, label="Haar / Max(Haar)")

#plt.hist(F_samples, bins=int((75 / 10000) * len(F_samples)))
plt.xlabel("Fidelity", fontsize=18)
plt.ylabel("Probability", fontsize=18)
plt.legend(fontsize=16)

plt.figure("6 q fixed_fsim plot")
plt.title("Fixed fsim  Circuit fidelity dist., N = 6", fontsize=20)
N = 2** 6
F_samples = fidelity_dict["fixed_fsim"]["6"]
prob, edges = np.histogram(F_samples, bins=int((75 / 10000) * len(F_samples)), range=(0,1)) #int((75 / 10000) * len(F_samples))
plt.plot(edges[:-1], prob/np.amax(prob), lw=1, label="Fidelities")
haar = (N - 1) * ((1 - edges) ** (N - 2))
plt.plot(edges[:-1], haar[:-1]/np.amax(haar), ls="--", lw=3, label="Haar / Max(Haar)")

#plt.hist(F_samples, bins=int((75 / 10000) * len(F_samples)))
plt.xlabel("Fidelity", fontsize=18)
plt.ylabel("Probability", fontsize=18)
plt.legend(fontsize=16)



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


