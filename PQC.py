#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 11:22:14 2021

@author: ronan
"""

import qutip as qt
import numpy as np
from itertools import chain
from helper_functions import genFockOp, flatten, prod, general_prod

rng = np.random.default_rng(1)


class Gate():
    n_qubit = 4

    def __mul__(self, b):
        if isinstance(b, Gate):
            return self._operation * b._operation
        else:
            return self._operation * b


class PRot(Gate):
    def __init__(self, q_on, q_N):
        self._q_on = q_on
        self._q_N = q_N
        self._theta = 0
        self._operation = self._set_op()

    def _set_op(self):
        self._gate = qt.qeye
        return qt.qeye([self._q_N, self._q_N])

    def set_theta(self, theta):
        self._theta = theta
        self._operation = self._set_op()

    def derivative(self):
        n_qubit_range = range(0, self._q_N)
        focks = [genFockOp(self._gate(), i, self._q_N, 2) for i in n_qubit_range]
        derivs = -1j * focks[self._q_on] / 2
        return derivs


class R_x(PRot):
    def _set_op(self):
        self._gate = qt.qip.operations.rx
        return self._gate(self._theta, N=self._q_N, target=self._q_on)


class R_y(PRot):
    def _set_op(self):
        self._gate = qt.qip.operations.ry
        return self._gate(self._theta, N=self._q_N, target=self._q_on)


class R_z(PRot):
    def _set_op(self):
        self._gate = qt.qip.operations.rz
        return self._gate(self._theta, N=self._q_N, target=self._q_on)


class H(R_y):
    def set_theta(self, angle):
        self._theta = np.pi / 4
        self._operation = self._set_op()


class EntGate(Gate):
    def __init__(self, qs_on, q_N):
        self._q1, self._q2 = qs_on[0], qs_on[1]
        self._q_N = q_N
        self._operation = self._set_op()

    def _set_op(self):
        self._gate = qt.qeye
        return qt.qeye(Gate.n_qubit)


class CNOT(EntGate):
    def _set_op(self):
        gate = qt.qip.operations.cnot
        return gate(self._q_N, self._q1, self._q2)


class CPHASE(EntGate):
    def _set_op(self):
        gate = qt.qip.operations.csign
        return gate(self._q_N, self._q1, self._q2)


class iSWAP(EntGate):
    def _set_op(self):
        gate = qt.qip.operations.sqrtiswap
        return gate(self._q_N, self._q1, self._q2)


class Chain(EntGate):
    def __init__(self, entangler):
        self._entangler = entangler
        self._operation = self._set_op()

    def _set_op(self):
        N = self._q_N
        top_connections = [[2 * j, 2 * j + 1] for j in range(N // 2)]
        bottom_connections = [[2 * j + 1, 2 * j + 2] for j in range((N - 1) // 2)]
        indices = top_connections + bottom_connections
        entangling_layer = [self._entangler(index_pair) for index_pair in indices][::-1]
        return prod(entangling_layer)


class AllToAll(EntGate):
    def __init__(self, entangler):
        self._entangler = entangler
        self._operation = self._set_op()

    def _set_op(self):
        N = self._q_N
        nested_temp_indices = []
        for i in range(N - 1):
            for j in range(i + 1, N):
                nested_temp_indices.append(rng.perumtation([i, j]))
        indices = flatten(nested_temp_indices)
        entangling_layer = [self._entangler(index_pair) for index_pair in indices][::-1]
        return prod(entangling_layer)


class PQC():
    def __init__(self, n_qubits, n_layers):
        Gate.n_qubits = n_qubits
        self._n_qubits = n_qubits
        self._n_layers = n_layers
        self.initial_layer = qt.qeye([self._n_qubits, self._n_qubits])

    def set_initialiser(self, layer):
        self.initial_layer = layer

    def set_gates(self, layer):
        layers = layer * self._n_layers
        self.gates = layers

    def set_params(self, random=True, angles=[]):
        paramterised = [g for g in self.gates if isinstance(g, PRot)]
        for count, p in enumerate(paramterised):
            if random is True:
                angle = rng.random(1)[0] * 2 * np.pi
            else:
                angle = angles[count]
            p.set_theta(angle)

    def initialise(self):
        circuit_state = qt.tensor([qt.basis(2, 0) for i in range(self._n_qubits)])
        #for i in self.initial_layer:
        #    circuit_state = i * circuit_state
        self.set_params()
        for g in self.gates:
            circuit_state = g * circuit_state
        return circuit_state

    def gen_quantum_state(self, energy_out=False):
        self._quantum_state = qt.Qobj(self.initialise())
        return self._quantum_state
