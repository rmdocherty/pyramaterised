#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 16:28:02 2021

@author: ronan
"""

from measurement import Measurements
from helper_functions import pretty_subplot
import matplotlib.pyplot as plt
import qutip as qt
import numpy as np
import random
import multiprocessing as mp
import PQC_lib as pqc

random.seed(1)
CORE_N = 4


def measure(circuit, sample_N):
    m = Measurements(circuit)
    overlaps, qs = m.reuse_states(sample_N)
    return overlaps, qs

def multi(circuit, total_N):
    overlaps, entanglements = [], []
    pool = mp.Pool(processes=CORE_N)
    split_N = [total_N//CORE_N for i in range(CORE_N)]
    results = [pool.apply(measure, args=(circuit, x)) for x in split_N]
    for tup in results:
        overlap = tup[0]
        entanglement = tup[1]
        overlaps = overlaps + overlap
        entanglements = entanglements + entanglement
    m = Measurements(circuit)
    Q, Q_std = np.mean(entanglements), np.std(entanglements)
    print(len(overlaps))
    expr = m._expr(overlaps, m._QC._n_qubits)
    print(f"Entanglement is {Q} +/- {Q_std}")
    print(f"Expressibility is {expr}")
    return Q, expr

#%%1min16
Q_L = []
expr_L = []
for L in range(4, 6):
    circuit_9 = pqc.PQC(4, L)
    layer = [pqc.H(0, 4), pqc.H(1, 4), pqc.H(2, 4), pqc.H(3,4), 
             pqc.CPHASE([3,2], 4), pqc.CPHASE([2,1], 4), pqc.CPHASE([1,0], 4),
             pqc.R_x(0, 4), pqc.R_x(1, 4), pqc.R_x(2, 4), pqc.R_x(3, 4)]
    circuit_9.set_gates(layer)
    q, e = multi(circuit_9, 5000)
    Q_L.append(q)
    expr_L.append(e)
#%%