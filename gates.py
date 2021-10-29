#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 11:24:31 2021

@author: ronan
"""

import qutip as qt
import numpy as np
from helper_functions import genFockOp


def iden(N):
    return qt.tensor([qt.qeye(2) for i in range(N)])


class Gate():

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

    def __repr__(self):
        name = type(self).__name__
        angle = self._theta
        string = f"{name}({angle:.2f})@q{self._q_on}"
        return string


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
        return qt.qeye(self._q_N)

    def __repr__(self):
        return f"{type(self).__name__}@q{self._q_1},q{self._q_2}"


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
    def __init__(self, entangler, q_N):
        self._entangler = entangler
        self._q_N = q_N
        self._operation = self._set_op()

    def _set_op(self):
        N = self._q_N
        top_connections = [[2 * j, 2 * j + 1] for j in range(N // 2)]
        bottom_connections = [[2 * j + 1, 2 * j + 2] for j in range((N - 1) // 2)]
        indices = top_connections + bottom_connections
        entangling_layer = [self._entangler(index_pair, N) for index_pair in indices][::-1]
        out = iden(N)
        for i in entangling_layer:
            out = i * out
        return out

    def __repr__(self):
        return f"CHAIN connected {self._entangler.__name__}s"


class AllToAll(EntGate):
    def __init__(self, entangler, q_N):
        self._entangler = entangler
        self._q_N = q_N
        self._operation = self._set_op()

    def _set_op(self):
        N = self._q_N
        nested_temp_indices = []
        for i in range(N - 1):
            for j in range(i + 1, N):
                nested_temp_indices.append(rng.perumtation([i, j]))
        indices = flatten(nested_temp_indices)
        entangling_layer = [self._entangler(index_pair, N) for index_pair in indices][::-1]
        out = iden(N)
        for i in entangling_layer:
            out = i * out
        return out

    def __repr__(self):
        return f"ALL connected {self._entangler.__name__}s"
