#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 11:22:14 2021

@author: ronan
"""

import qutip as qt
import numpy as np
from helper_functions import genFockOp


class Gate():
    n_qubit = 4


class PRot(Gate):
    def __init__(self, qubit, theta):
        self._act_on_q = qubit
        self._theta = theta
        self._operation = self._set_op()

    def _set_op(self):
        self._gate = qt.qeye
        return qt.qeye(Gate.n_qubit)

    def derivative(self):
        n_qubit_range = range(0, Gate.n_qubit)
        focks = [genFockOp(self._gate(), i, Gate.n_qubit, 2) for i in n_qubit_range]
        derivs = -1j * focks[self._act_on_q] / 2
        return derivs


class R_x(PRot):
    def _set_op(self):
        self._gate = qt.qip.operations.rx
        return self._gate(self._theta, N=Gate.n_qubit, target=self._act_on_q)


class R_y(PRot):
    def _set_op(self):
        self._gate = qt.qip.operations.ry
        return self._gate(self._theta, N=Gate.n_qubit, target=self._act_on_q)


class R_z(PRot):
    def _set_op(self):
        self._gate = qt.qip.operations.rz
        return self._gate(self._theta, N=Gate.n_qubit, target=self._act_on_q)


class EntGate(Gate):
    def __init__(self, qubits):
        self._q1, self._q2 = qubits[0], qubits[1]
        self._operation = self._set_op()

    def _set_op(self):
        self._gate = qt.qeye
        return qt.qeye(Gate.n_qubit)


class CNOT(EntGate):
    def _set_op(self):
        gate = qt.qip.operations.cnot
        return gate(Gate.n_qubit, self._q1, self._q2)


class Chain(EntGate): #TODO - parameterie in terms of which entangler we want
    def _set_op(self):
        indices = [[2 * j, 2 * j + 1] for j in range(self._n_qubits // 2)]
        return [CNOT(index_pairs) for index_pairs in indices]


class PQC():
    def __init__(self, n_qubits, n_layers):
        Gate.n_qubits = n_qubits
        self._n_layers = n_layers

a = R_x([0], np.pi/4)