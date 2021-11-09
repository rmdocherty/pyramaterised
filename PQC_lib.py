#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 11:22:14 2021

@author: ronan
"""
#%% Imports
import qutip as qt
import numpy as np
from itertools import chain
from copy import copy, deepcopy
from helper_functions import genFockOp, flatten, prod

rng = np.random.default_rng(1)

#%% =============================GATES=============================


def iden(N):
    return qt.tensor([qt.qeye(2) for i in range(N)])


class Gate():
    """Parent class for all gate types to inherit from - describes the behaviour
    when any Gate (or derived classes) is multiplied. Want to ensure that the
    gate always uses its underlying qutip gate representation when multiplied."""

    def __mul__(self, b):
        if isinstance(b, Gate):
            return self._operation * b._operation
        else:
            return self._operation * b

    def __rmul__(self, b):
        if isinstance(b, Gate):
            return b._operation * self._operation
        else:
            return b * self._operation

#%% Rotation gates


class PRot(Gate):
    """A class to described how parametrised rotation gates work - they have a
    qubit they operate on, a total number of qubits in the system (so gate can
    be extended to that dimension) and an angle that the gate rotates by."""

    def __init__(self, q_on, q_N):
        self._q_on = q_on
        self._q_N = q_N
        self._theta = 0
        self._is_param = True
        self._operation = self._set_op()

    def _set_op(self):
        """Change this for different PRots to change their behaviour"""
        self._gate = qt.qeye
        self._pauli = qt.qeye(2)
        return qt.qeye([self._q_N, self._q_N])

    def set_theta(self, theta):
        self._theta = theta
        self._operation = self._set_op()

    def derivative(self):
        """Take the derivative of the PRot - this generates the pauli gate
        associated with the gate type (i.e R_x -> sigma_x) operating on given
        qubit and multiplies it by j/2."""
        focks = genFockOp(self._pauli, self._q_on, self._q_N, 2)
        deriv = -1j * focks / 2
        return deriv

    def __repr__(self):
        name = type(self).__name__
        angle = self._theta
        string = f"{name}({angle:.2f})@q{self._q_on}"
        return string


class R_x(PRot):
    def _set_op(self):
        self._gate = qt.qip.operations.rx
        self._pauli = qt.sigmax()
        return self._gate(self._theta, N=self._q_N, target=self._q_on)


class R_y(PRot):
    def _set_op(self):
        self._gate = qt.qip.operations.ry
        self._pauli = qt.sigmay()
        return self._gate(self._theta, N=self._q_N, target=self._q_on)


class R_z(PRot):
    def _set_op(self):
        self._gate = qt.qip.operations.rz
        self._pauli = qt.sigmaz()
        return self._gate(self._theta, N=self._q_N, target=self._q_on)

#%% Fixed angle single-qubit rotations


class H(PRot):
    """Hadamard gate."""

    def __init__(self, q_on, q_N):
        self._q_on = q_on
        self._q_N = q_N
        self._theta = np.pi / 2
        self._is_param = False
        self._operation = self._set_op()

    def set_theta(self, angle):
        return None

    def _set_op(self):
        """Hadamard gate is just sigma_x * R_y(pi/2)"""
        ops = qt.qip.operations
        self._gate = ops.ry
        return ops.x_gate(self._q_N, self._q_on) * self._gate(np.pi / 2, N=self._q_N, target=self._q_on)


class sqrtH(H):
    """Sqrt hadarmard, used in the quantum geometry circuit."""

    def _set_op(self):
        """sqrt Hadamard gate is just R_y(pi/4)"""
        self._theta = np.pi / 4
        ops = qt.qip.operations
        self._gate = ops.ry
        return self._gate(np.pi / 4, N=self._q_N, target=self._q_on)


class fixed_R_y(sqrtH):
    """Fixed R_y rotation by pi/2. Needs to inherit from sqrtH so that it
    isn't counted as a parameterised gate later in NPQC code."""

    def _set_op(self):
        self._theta = np.pi / 2
        ops = qt.qip.operations
        self._gate = ops.ry
        return self._gate(np.pi / 2, N=self._q_N, target=self._q_on)


class S(sqrtH):
    def _set_op(self):
        self._theta = np.pi / 2
        ops = qt.qip.operations
        self._gate = ops.phasegate
        return self._gate(np.pi / 2, N=self._q_N, target=self._q_on)

#%% Entangling gates


class EntGate(Gate):
    """A class to described how entangling gates work - they have the
    qubits they operate on (control and target) and a total number of qubits
    in the system. Works the same way as rotation gates, i.e changing the
    _set_op() method to use the right qutip gate."""

    def __init__(self, qs_on, q_N):
        self._q1, self._q2 = qs_on[0], qs_on[1]
        self._q_N = q_N
        self._is_param = False
        self._operation = self._set_op()

    def _set_op(self):
        self._gate = qt.qeye
        return qt.qeye(self._q_N)

    def __repr__(self):
        return f"{type(self).__name__}@q{self._q1},q{self._q2}"


class CNOT(EntGate):
    def _set_op(self):
        gate = qt.qip.operations.cnot
        return gate(self._q_N, self._q1, self._q2)


class CPHASE(EntGate):
    def _set_op(self):
        """The CPHASE gate not a real cphase gate, defined in papers as CZ gate."""
        gate = qt.qip.operations.cz_gate
        return gate(self._q_N, self._q1, self._q2)


class sqrtiSWAP(EntGate):
    def _set_op(self):
        gate = qt.qip.operations.sqrtiswap
        return gate(self._q_N, self._q1, self._q2)


def CZ(EntGate):
    def _set_op(self):
        ops = qt.qip.operations
        self._gate = ops.cz_gate
        return self._gate(self._q_N, self._q1, self._q2)

#%% Block entangling gates


class CHAIN(EntGate):
    """Can make a Chain of a given entangling gate by generating all indices
    and making an entangler between all these indices."""

    def __init__(self, entangler, q_N):
        self._entangler = entangler
        self._q_N = q_N
        self._is_param = False
        self._operation = self._set_op()

    def _set_op(self):
        N = self._q_N
        top_connections = [[2 * j, 2 * j + 1] for j in range(N // 2)]
        bottom_connections = [[2 * j + 1, 2 * j + 2] for j in range((N - 1) // 2)]
        indices = top_connections + bottom_connections
        entangling_layer = prod([self._entangler(index_pair, N) for index_pair in indices][::-1])
        return entangling_layer

    def __repr__(self):
        return f"CHAIN connected {self._entangler.__name__}s"


class ALLTOALL(EntGate):
    """Define AllToAll in similar way to Chain block for a generic entangler."""

    def __init__(self, entangler, q_N):
        self._entangler = entangler
        self._q_N = q_N
        self._is_param = False
        self._operation = self._set_op()

    def _set_op(self):
        N = self._q_N
        nested_temp_indices = []
        for i in range(N - 1):
            for j in range(i + 1, N):
                nested_temp_indices.append(rng.perumtation([i, j]))
        indices = flatten(nested_temp_indices)
        entangling_layer = prod([self._entangler(index_pair, N) for index_pair in indices][::-1])
        return entangling_layer

    def __repr__(self):
        return f"ALL connected {self._entangler.__name__}s"

#%% 2 qubit rotation gates


class R_zz(PRot, EntGate):
    def __init__(self, qs_on, q_N):
        self._q1, self._q2 = qs_on[0], qs_on[1]
        self._q_N = q_N
        self._theta = 0
        self._is_param = True
        self._operation = self._set_op()

    def _set_op(self):
        self._gate = qt.qip.operations.rz
        self._pauli = qt.sigmaz() #are these derivatives right?
        g1 = self._gate(self._theta, N=self._q_N, target=self._q1)
        g2 = self._gate(self._theta, N=self._q_N, target=self._q2)
        return qt.tensor(g1, g2)

    def __repr__(self):
        name = type(self).__name__
        angle = self._theta
        return f"{name}({angle})@q{self._q1},q{self._q2}"


class R_xx(R_zz):
    def _set_op(self):
        self._gate = qt.qip.operations.rx
        self._pauli = qt.sigmax()
        g1 = self._gate(self._theta, N=self._q_N, target=self._q1)
        g2 = self._gate(self._theta, N=self._q_N, target=self._q2)
        return qt.tensor(g1, g2)


class R_yy(R_zz):
    def _set_op(self):
        self._gate = qt.qip.operations.ry
        self._pauli = qt.sigmay()
        g1 = self._gate(self._theta, N=self._q_N, target=self._q1)
        g2 = self._gate(self._theta, N=self._q_N, target=self._q2)
        return qt.tensor(g1, g2)


#%% =============================CIRCUIT=============================


class PQC():
    """A class to define an n qubit wide, n layer deep Parameterised Quantum
    Circuit."""

    def __init__(self, n_qubits):
        self._n_qubits = n_qubits
        self._n_layers = 0
        self._layers = []

    def add_layer(self, layer, n=1):
        """Add $n layers to PQC._layers"""
        for i in range(n):
            self._layers.append(deepcopy(layer))
        self._n_layers += n
        self.set_gates() #update PQC.gates when adding

    def set_layer(self, layer, pos):
        """Set nth layer of PQC to given layer. Will throw error if pos not available"""
        self._layers[pos] = deepcopy(layer)
        self.set_gates()

    def set_gates(self):
        """For each layer in layer, append it to gates. If layers is a nested list
        of layers, then gates is a flat list of each gate operation in order. Then
        iterate through gates and update a list that says if a gate is parameterised
        or not (-1), and which parameterised gate it is, i.e is it the 1st, 2nd, ..."""
        layers = []
        for i in self._layers:
            layers = layers + i
        self.gates = layers
        self._parameterised = []
        param_count = 0
        for gate in self.gates:
            if gate._is_param:
                self._parameterised.append(param_count)
                param_count += 1
            else:
                self._parameterised.append(-1)

    def get_params(self):
        angles = [g._theta for g in self.gates if g._is_param]
        return angles

    def set_params(self, angles=[]):
        """Set the parameters of every parameterised gate (i.e inherits from PRot)
        in the circuit. Can set either randomly or from a specified list"""
        parameterised = [g for g in self.gates if g._is_param]
        for count, p in enumerate(parameterised):
            if angles != []:
                angle = angles[count]
            else: #use random params
                angle = rng.random(1)[0] * 2 * np.pi
            p.set_theta(angle)

    def run(self, angles=[]):
        """Set |psi> of a PQC by multiplying the basis state by the gates."""
        circuit_state = qt.tensor([qt.basis(2, 0) for i in range(self._n_qubits)])
        self.set_params(angles=angles)
        for g in self.gates:
            circuit_state = g * circuit_state
        return circuit_state

    def gen_quantum_state(self, energy_out=False):
        """Get a Qobj of |psi> for measurements."""
        self._quantum_state = qt.Qobj(self.run())
        if energy_out is True:
            e = self.energy()
            print(f"Energy of state is {e}")
        return self._quantum_state

    def energy(self):
        """Get energy of |psi>, the initial quantum state"""
        Z0 = genFockOp(qt.sigmaz(), 0, self._n_qubits, 2)
        Z1 = genFockOp(qt.sigmaz(), 1, self._n_qubits, 2)
        H = Z0 * Z1
        energy = qt.expect(H, self._quantum_state)
        return energy

    def take_derivative(self, g_on):
        """Get the derivative of the ith parameter of the circuit and return
        the circuit where the ith gate is multiplied by its derivative."""
        #need to find which gate the ith parameterised gate is
        p_loc = self._parameterised.index(g_on)
        gates = self.gates
        #copy so don't overwrite later - much better than deepcopying whole circuit!
        gate = copy(gates[p_loc])
        #find the derivative using the gate's derivative method
        deriv = gate.derivative()
        #set pth gate to be deriv * gate
        gates[p_loc] = deriv * gate
        #act the derivative of circuit on |0>
        circuit_state = qt.tensor([qt.basis(2, 0) for i in range(self._n_qubits)])
        for g in gates:
            circuit_state = g * circuit_state
        #reset the gate back to what it was originally
        gates[p_loc] = gate
        return circuit_state

    def get_gradients(self):
        """Get the n_params circuits with the ith derivative multiplied in and
        then apply them to the basis state."""
        gradient_state_list = []
        n_params = len([i for i in self._parameterised if i > -1])
        for i in range(n_params):
            gradient = self.take_derivative(i)
            gradient_state_list.append(gradient)
        return gradient_state_list

    def __repr__(self):
        line = f"A {self._n_qubits} qubit, {self._n_layers} layer deep PQC. \n"
        for count, l in enumerate(self._layers):
            line += f"Layer {count}: {l} \n"
        return line
