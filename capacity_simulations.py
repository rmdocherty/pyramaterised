#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 13:15:43 2022

@author: ronan
"""
import measure_all as measure

circuits = ["fsim"]
n_qubits = 10

for c in circuits:
    for n in range(2, 10):
        if c == "fsim":
            rot_str = "z"
        else:
            rot_str = ""

        if n % 2 == 1 and c in ["fermionic", "XXZ"]:
            pass
        else:
            for p in range(1, 14):
                if c == "NPQC" and p >= 2**(n // 2):
                    pass
                else:
                    out = measure.measure_everything(c, n, p, n_repeats=0, n_samples=200, train=False, save=True, rotator=rot_str)
    #Deep circuit properties
    for c_prime in ["fsim", "NPQC"]:
        if c_prime == "fsim":
            rot_str = "z"
        else:
            rot_str = ""
        for n in range(5, 8):
            out = measure.measure_everything(c_prime, n, 3*n, n_repeats=0, n_samples=200, train=False, save=True, rotator=rot_str)
        
        out = measure.measure_everything(c_prime, 8, 4 * 8, n_repeats=0, n_samples=200, train=False, save=True, rotator=rot_str)
        out = measure.measure_everything(c_prime, 9, 7 * 9, n_repeats=0, n_samples=200, train=False, save=True, rotator=rot_str)
