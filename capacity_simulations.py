#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 13:15:43 2022

@author: ronan
"""
import circuit_structures as cs
import measure_all as measure

circuits = ["Circuit_9"]
n_qubits = 10

for c in circuits:
    for n in range(2, 10):
        if n % 2 == 1 and (c == "fermionic" or c == "fsim"):
            pass
        else:
            for p in range(1, 14):
                out = measure.measure_everything(c, n, p, n_repeats=0, n_samples=200, train=False, save=True)
