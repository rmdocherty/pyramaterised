#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 15:32:41 2022

@author: ronan
"""

import pickle

with open('data/capacity/Circuit_1_TFIM_7q_1l_10r_clifford.pickle', 'rb') as f:
    x = pickle.load(f)
    print(x["Metatdata"]["N_qubits"])