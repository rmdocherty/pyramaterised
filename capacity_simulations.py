#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 13:15:43 2022

@author: ronan
"""
import measure_all as measure

circuits = ["XXZ", "fsim", "zfsim", "TFIM"]#["fermionic", "fsim", "zfsim", "XXZ"]

for c in circuits:
    for n in range(8, 11):
        if n % 2 == 1 and c in ["fermionic", "XXZ", "fsim", "zfsim", "TFIM"]:
            pass
        else:
            if n < 8:
                p = 30
                g = 10
            elif n > 7:
                p = 220
                g = 4
            print(c, n, p)
            if c == "NPQC" and p >= 2**(n // 2):
                pass
            else:
                if c == "zfsim":
                    out = measure.measure_everything("fsim", n, p, n_repeats=0, n_samples=0, 
                                                     train=False, save=True, n_qfim=g, rotator='z')
                else:
                    out = measure.measure_everything(c, n, p, n_repeats=0, n_samples=0, 
                                                     train=False, save=True, n_qfim=g, start="random")

for n in [4, 6]:
    out = measure.measure_everything("XXZ", n, 120, n_repeats=0, n_samples=600, 
                                     train=False, save=True, n_qfim=5, start="random")