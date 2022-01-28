#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 15:40:57 2022

@author: ronan
"""
import PQC_lib as pqc
import qutip as qt
import numpy as np
from circuit_structures import gen_TFIM_layers, TFIM_hamiltonian
from memory_profiler import profile
from measurement import Measurements

N, p = 13, 4
g_0, h_0 = 1, 0

TFIM = pqc.PQC(N)
#need to use |+> as initial state for TFIM model
plus_state = (1/np.sqrt(2)) * (qt.basis(2,0) + qt.basis(2,1))
final_state = qt.tensor([plus_state for i in range(N)])
#TFIM.set_initial_state(plus_state)

hamiltonian = TFIM_hamiltonian(N, g=g_0, h=h_0)
groundstate_energy, groundstate = hamiltonian.groundstate()
a = hamiltonian.eigenenergies()
#print(groundstate_energy, groundstate)
TFIM.set_H(hamiltonian)

TFIM_layers = gen_TFIM_layers(p, N)
for l in TFIM_layers:
    TFIM.add_layer(l)

@profile()
def test():
    for i in range(1):
        TFIM.run()
        m = Measurements(TFIM)
        #m.efficient_measurements(50)
        m._get_QFI()

test()
