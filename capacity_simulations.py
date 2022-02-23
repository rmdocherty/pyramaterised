#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 13:15:43 2022

@author: ronan
"""
import measure_all as measure

circuits = ["fermionic", "fsim", "zfsim", "XXZ", "TFIM", "generic_HE", "qg_circuit", "y_CPHASE", "NPQC"]#["fermionic", "fsim", "zfsim", "XXZ"]

for c in circuits:
    for n in range(9, 11):
        if n % 2 == 1 and c in ["fermionic", "XXZ", "fsim", "zfsim", "TFIM"]:
            pass
        else:
            for p in range(120,121):
                print(c, n, p)
                if c == "NPQC" and p >= 2**(n // 2):
                    pass
                else:
                    if c == "zfsim":
                        out = measure.measure_everything("fsim", n, p, n_repeats=0, n_samples=600, 
                                                         train=False, save=True, n_qfim=0, rotator='z')
                    else:
                        out = measure.measure_everything(c, n, p, n_repeats=0, n_samples=600, 
                                                         train=False, save=True, n_qfim=0)
