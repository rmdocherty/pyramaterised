#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 13:15:43 2022

@author: ronan
"""
import measure_all as measure

circuits = ["y_CPHASE", "NPQC"]#["fsim", "generic_HE", "qg_circuit", "fermionic", "TFIM", "XXZ", "yCPHASE", "NPQC"]

for c in circuits:
    for n in range(2, 13):
        if n % 2 == 1 and c in ["fermionic", "XXZ"]:
            pass
        else:
            for p in range(1,14):
                if c == "NPQC" and p >= 2**(n // 2):
                    pass
                elif c == "TFIM" and (p < 8 or n < 12):
                    pass
                else:
                    out = measure.measure_everything(c, n, p, n_repeats=0, n_samples=0, 
                                                         train=False, save=True, n_qfim=30)
