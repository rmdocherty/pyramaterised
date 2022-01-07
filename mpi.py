
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 15:52:47 2021

@author: ronan
"""
import random
import numpy as np
from  mpi4py import MPI
from measure_all import measure_everything

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#circuits = ["NPQC", "TFIM", "TFIM_modified", "Circuit_1", "Circuit_2", "Circuit_9", "qg_circuit", "generic_HE"]
circuits = ["NPQC", "TFIM", "qg_circuit", "generic_HE"]
data = []

for c in circuits:    
    for l in range(1, 11):
        for H in ['ZZ']:
            for start in ['random', 'clifford']:
                for train_for in ["magic", "gkp"]:
                    for q in range(2, 9):
                        if l > (q // 2) - 1 and c == "NPQC":
                            pass
                        else:
                            data.append([q, c, l, H, start, train_for])

if rank == 0:
    N_CORES = [4, 4] #4, 4, 2, 2]
    SPEED_RATIOS = [100, 100, 100, 100, 100, 100, 100, 100] # 60, 60, 60, 60, 40, 40, 40, 40, 10, 10, 2, 2]
    total = sum(SPEED_RATIOS)
    num_allocated = [int(len(data) * i / total) for i in SPEED_RATIOS]
    num_allocated = np.array(num_allocated, dtype=np.int64)
    print(num_allocated)
    slices = [0] + [sum(num_allocated[:i]) for i in range(1, len(num_allocated))]
    slices = np.array(slices, dtype=np.int64)
    #print(slices)
    range_data = np.array([i for i in range(len(data))])
    allocated_data = [range_data[slices[i]:slices[i + 1]] for i in range(len(slices) - 1)]
    recieve = np.zeros(len(data), dtype=np.int64)
    #print(len(range_data), len(num_allocated), len(slices), size, len(SPEED_RATIOS))
    print(f'Comptuting data of {len(data)} circuits')
else:
   #data = None
   slices = None
   range_data = None
   num_allocated = np.zeros(size, dtype=np.int64)

recieve = np.zeros(len(data), dtype=np.int64)

range_data = comm.Scatterv([range_data, num_allocated, slices, MPI.DOUBLE], recieve, root=0)

n_non_zero = np.count_nonzero(recieve)
for count, i in enumerate(recieve):
    if i != 0:
        print(i, data[i])
        out = measure_everything(data[i][1], data[i][0], data[i][2], 10, 0, hamiltonian=data[i][3], start=data[i][4], train=True, train_method="BFGS", save=True, train_for=data[i][5])
        print(f"Process {rank} has completed circuit {count} of {n_non_zero}")
