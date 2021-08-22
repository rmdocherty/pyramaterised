#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 18:12:45 2021

@author: ronan

Does our QC code match up to the results from Tobias' code?
"""
from QC import QuantumCircuit
from measurement import Measurements

q = QuantumCircuit(4, 3, "chain", "cnot")
print("4 qubit chain topology connected indices")
print(q._gen_entanglement_indices())

print("5 qubit chain topology connected indices")
q2 = QuantumCircuit(5, 3, "chain", "cnot")
print(q2._gen_entanglement_indices())

#%%
q.run() #should ouput 0.46135870050914374

#%%
q3 = QuantumCircuit(4, 1, "chain", "cnot") #single layer as in paper?
M = Measurements(q)
#EFD = M.get_effective_quantum_dimension(10**-12)
#print(f"Effective quantum dimension is {EFD}")
expr = M.expressibility(5000, graphs=True)
