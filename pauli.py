#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 16:53:53 2021

@author: ronan
"""

from itertools import product
import qutip as qt
import numpy as np


def compute_projector(N):
    pauli_list = [qt.qeye(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    strings = product(pauli_list, repeat=N)
    P_n = [qt.tensor(list(s)) for s in strings]
    empty = qt.Qobj(np.array([[0, 0], [0, 0]]))
    proj = qt.tensor([qt.tensor([empty for i in range(N)]) for i in range(4)])
    for P in P_n:
        proj += qt.tensor([P for i in range(4)])
    proj *= (2**N)**-2
    return proj


q = compute_projector(4)
print(q)
