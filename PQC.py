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
    def __init__(self, qubit, theta):
        self._act_on_q = qubit
        self._theta = theta
        self._operation = self._set_op()

    def _set_op(self):
        self._gate = qt.qeye
        return qt.qeye(2**Gate.n_qubit)

    def set_theta(self, theta):
        self._theta = theta
        self._operation = self._set_op()

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


class H(R_y):
    def set_theta(self, theta):
        self._theta = np.pi / 4


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


class CPHASE(EntGate):
    def _set_op(self):
        gate = qt.qip.operations.csign
        return gate(Gate.n_qubit, self._q1, self._q2)


class iSWAP(EntGate):
    def _set_op(self):
        gate = qt.qip.operations.sqrtiswap
        return gate(Gate.n_qubit, self._q1, self._q2)


class Chain(EntGate):
    def __init__(self, entangler):
        self._entangler = entangler
        self._operation = self._set_op()

    def _set_op(self):
        N = Gate.n_qubit
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
        N = Gate.n_qubit
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
        self.initial_layer = qt.tensor([qt.qeye(2) for i in range(self._n_qubits)])

    def set_initialiser(self, layer):
        self.initial_layer = layer

    def set_gates(self, layer):
        layers = layer * self._n_layers
        #print(layers)
        for i in layers:
            i.n_qubit = self._n_qubits
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
        for i in self.initial_layer:
            circuit_state = i * circuit_state
        self.set_params()
        for g in self.gates:
            print(Gate.n_qubit)
            print(g._operation, circuit_state)
            circuit_state = g * circuit_state
        return circuit_state

    def gen_quantum_state(self, energy_out=False):
        self._quantum_state = qt.Qobj(self.initialise())
        return self._quantum_state
    
#CAN ONLY EDIT CLASS VARIABLES FOR INSTANCES OF OBJECTS - WOULD NEED TO DO ON A PER GATE BASIS!!
