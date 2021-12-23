#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 15:52:47 2021

@author: ronan
"""
import random

from mpi4py import MPI
from measure_all import measure_everything

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    SPEED_RATIOS = [40, 20, 15, 10, 5, 1]
    total = sum(SPEED_RATIOS)
    circuits = ["NPQC", "TFIM", "TFIM_modified", "XXZ", "Circuit_1", "Circuit_2", "Circuit_9", "qg_circuit", "generic_HE"]
    data = []
    for q in range(2, 8):
        for c in circuits:    
            for l in range(1, 8):
                for H in ['ZZ', 'TFIM']:
                    for start in ['random', 'clifford']:
                        data.append([q, c, l, H, start])

    num_allocated = [int(len(data) * i / total) for i in SPEED_RATIOS]
    slices = [0] + [sum(num_allocated[:i]) for i in range(1, len(num_allocated))] + [len(data)]
    allocated_data = [data[slices[i]:slices[i + 1]] for i in range(len(slices) - 1)]
   
    print(f'Comptuting data of {len(data)} circuits')
else:
   data = None

allocated_data = comm.scatter(allocated_data, root=0)
out = [measure_everything(i[1], i[0], i[2], 10, 500, hamiltonian=i[3], start=i[4], train=False, save=True) for i in allocated_data]
