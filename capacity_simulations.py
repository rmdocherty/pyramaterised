#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 13:15:43 2022

@author: ronan
"""
import measure_all as measure

circuits = ["fixed_fsim", "clifford", "qg_circuit", "y_CPHASE"]#["fermionic", "fsim", "zfsim", "XXZ"]

for c in circuits:
    for n in range(4, 5):
        if n % 2 == 1 and c in ["fermionic", "XXZ", "fsim", "zfsim", "TFIM", "fixed_fsim"]:
            pass
        else:
            for p in range(1,31):
                print(c, n, p)
                if c == "NPQC" and p >= 2**(n // 2):
                    pass
                else:
                    if c == "zfsim":
                        out = measure.measure_everything("fsim", n, p, n_repeats=0, n_samples=600, 
                                                         train=False, save=True, n_qfim=30, rotator='z')
                    elif c == "clifford":
                        out = measure.measure_everything(c, n, p, n_repeats=0, n_samples=600, 
                                                         train=False, save=True, n_qfim=30, start="clifford")
                    else:
                        out = measure.measure_everything(c, n, p, n_repeats=0, n_samples=600, 
                                                         train=False, save=True, n_qfim=30, start="random")
