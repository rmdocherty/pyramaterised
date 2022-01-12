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

    def __add__(self, b):
        if isinstance(b, Gate):
            return self._operation + b._operation
        else:
            return self._operation + b

    def __radd__(self, b):
        if isinstance(b, Gate):
            return b._operation + self._operation
        else:
            return b + self._operation


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
        self._param_count = 1

        self._set_properties()
        self._fock = genFockOp(self._pauli, self._q_on, self._q_N, 2)
        self._operation = self._set_op()

    def _set_op(self):
        return self._gate(self._theta, N=self._q_N, target=self._q_on)
    
    def _set_properties(self):
        self._gate = iden
        self._pauli = iden

    def set_theta(self, theta):
        self._theta = theta
        self._operation = self._set_op()

    def derivative(self):
        """Take the derivative of the PRot - this generates the pauli gate
        associated with the gate type (i.e R_x -> sigma_x) operating on given
        qubit and multiplies it by j/2."""
        deriv = -1j * self._fock / 2
        return deriv

    def flip_pauli(self):
        self._pauli = -1 * self._pauli

    def __repr__(self):
        name = type(self).__name__
        angle = self._theta
        string = f"{name}({angle:.2f})@q{self._q_on}"
        return string


class R_x(PRot):
    def _set_properties(self):
        self._gate = qt.qip.operations.rx
        self._pauli = qt.sigmax()


class R_y(PRot):
    def _set_properties(self):
        self._gate = qt.qip.operations.ry
        self._pauli = qt.sigmay()


class R_z(PRot):
    def _set_properties(self):
        self._gate = qt.qip.operations.rz
        self._pauli = qt.sigmaz()


#%% Fermionic specific gates

class negative_R_z(R_z):
    def set_theta(self, theta):
        self._theta = -1 * theta
        self._operation = self._set_op()


class offset_R_z(R_z):
    def __init__(self, q_on, q_N, offset):
        self._q_on = q_on
        self._q_N = q_N
        self._theta = 0
        self._offset = offset 
        self._is_param = True
        self._param_count = 1


        self._set_properties()
        self._fock = genFockOp(self._pauli, self._q_on, self._q_N, 2)
        self._operation = self._set_op()

    def set_theta(self, theta):
        self._theta = theta + self._offset
        self._operation = self._set_op() 

#%% Fixed angle single-qubit rotations


class H(PRot):
    """Hadamard gate."""

    def __init__(self, q_on, q_N):
        self._q_on = q_on
        self._q_N = q_N
        self._theta = np.pi / 2
        self._is_param = False
        self._param_count = 0
        self._operation = self._set_op()

    def set_theta(self, angle):
        return None

    def _set_op(self):
        """Hadamard gate is just sigma_x * R_y(pi/2)"""
        ops = qt.qip.operations
        self._gate = ops.ry
        return ops.x_gate(self._q_N, self._q_on) * self._gate(np.pi / 2, N=self._q_N, target=self._q_on)


class sqrtH(H):
    def _set_op(self):
        ops = qt.qip.operations
        self._gate = ops.ry
        return np.sqrt(ops.x_gate(self._q_N, self._q_on) * self._gate(np.pi / 2, N=self._q_N, target=self._q_on))

class X(H):
    def _set_op(self):
        """Pauli X gate"""
        ops = qt.qip.operations
        return ops.x_gate(self._q_N, self._q_on)


class fixed_R_y(R_y):
    """Fixed R_y rotation by angle theta. Isn't parameterised and angle can't
    be changed after initialization."""

    def __init__(self, q_on, q_N, theta):
        self._q_on = q_on
        self._q_N = q_N
        self._theta = theta
        self._is_param = False
        self._param_count = 0
        self._set_properties()
        self._operation = self._set_op()

    def set_theta(self, theta):
        return None


class fixed_R_z(R_z):
    """Fixed R_y rotation by angle theta. Isn't parameterised and angle can't
    be changed after initialization."""

    def __init__(self, q_on, q_N, theta):
        self._q_on = q_on
        self._q_N = q_N
        self._theta = theta
        self._is_param = False
        self._param_count = 0
        self._set_properties()
        self._operation = self._set_op()

    def set_theta(self, theta):
        return None


class S(H):
    def _set_op(self):
        self._theta = np.pi / 2
        ops = qt.qip.operations
        self._gate = ops.phasegate
        return self._gate(np.pi / 2, N=self._q_N, target=self._q_on)


class T(H):
    """T-gate."""

    def _set_op(self):
        ops = qt.qip.operations
        self._gate = ops.t_gate
        return self._gate(N=self._q_N, target=self._q_on)

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
        self._param_count = 0
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
        return gate(self._q_N, [self._q1, self._q2])


class CZ(EntGate):
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
        self._param_count = 0
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
        self._param_count = 0
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

    
#%% Arbitrary Gate    
    
class ARBGATE(Gate):
    
    def __init__(self, Ham):
        self._Ham = Ham
        self._theta = 0
        self._is_param = True
        self._param_count = 1
        self._operation = self._set_op()
 
 
    def _set_op(self):
        return (-1j*self._theta*self._Ham).expm() 
    
     
    def set_theta(self, theta):
        self._theta = theta
        self._operation = self._set_op()  
        
    def derivative(self):
        deriv = -1j * self._Ham
        return deriv
    
    def __repr__(self):
        name = type(self).__name__
        angle = self._theta
        string = f"{name}({angle:.2f})"
        return string
    
    
    
#%% Sharing Parameters


class shared_parameter(PRot):
    def __init__(self, layer, q_N):
        self._layer = layer
        self._theta = 0
        self._q_N = q_N
        self._is_param = True
        self._param_count = 1
        self._operation = self._set_op()

    def derivative(self):
        """H=sum(H_s) -> d_theta U = d_theta (e^i*H*theta) = sum(H_s * U)"""
        deriv = 0
        for g in self._layer:
            single_deriv = g.derivative()
            deriv = deriv + single_deriv #multiply or plus?
        return deriv

    def set_theta(self, theta):
        self._theta = theta
        for gate in self._layer:
            gate.set_theta(theta)
        self._operation = self._set_op()

    def _set_op(self):
        operation = prod(self._layer[::-1])
        return operation

    def flip_pauli(self):
        for g in self._layer:
            g.flip_pauli()

    def __repr__(self):
        return f"Block of {self._layer}"


#%% 2 qubit rotation gates

#big question - should the second qubit angle be -1 * theta ???


class RR(PRot):
    def __init__(self, qs_on, q_N):
        self._q1, self._q2 = qs_on[0], qs_on[1]
        self._q_N = q_N
        self._theta = 0
        self._is_param = True
        self._param_count = 1
        self._set_properties()
        #self._fock1 = genFockOp(self._pauli, self._q1, self._q_N, 2)
        #self._fock2 = genFockOp(self._pauli, self._q2, self._q_N, 2)
        self._I = iden(self._q_N)
        self._operation = self._set_op()

    def _set_properties(self):
        self._gate = iden
        self._pauli = iden    

    def _set_op(self): #analytic expression for exponent of pauli is cos(x)*I + sin(x)*pauli_str
        return np.cos(self._theta / 2) * self._I - 1j * np.sin(self._theta / 2) * self._fock1 * self._fock2

    def derivative(self):
        """Derivative of XX/YY/ZZ is -i * tensor(sigmai, sigmai) /2"""
        deriv = -1j * (self._fock1 * self._fock2) / 2
        return deriv

    def __repr__(self):
        name = type(self).__name__
        angle = self._theta
        return f"{name}({angle:.2f})@q{self._q1},q{self._q2}"


class R_zz(RR):
    def _set_properties(self):
        self._gate = qt.qip.operations.rz
        self._fock1 = qt.qip.operations.z_gate(N=self._q_N, target=self._q1)
        self._fock2 = qt.qip.operations.z_gate(N=self._q_N, target=self._q2)


class R_xx(RR):
    def _set_properties(self):
        self._gate = qt.qip.operations.rx
        self._fock1 = qt.qip.operations.x_gate(N=self._q_N, target=self._q1)
        self._fock2 = qt.qip.operations.x_gate(N=self._q_N, target=self._q2)


class R_yy(RR):
    def _set_properties(self):
        self._gate = qt.qip.operations.ry
        self._fock1 = qt.qip.operations.y_gate(N=self._q_N, target=self._q1)
        self._fock2 = qt.qip.operations.y_gate(N=self._q_N, target=self._q2)


class RR_block(shared_parameter):
    def __init__(self, rotator, q_N):
        self._rotator = rotator
        self._theta = 0
        self._q_N = q_N
        self._is_param = True
        self._param_count = 1
        self._layer = self.gen_layer()
        self._operation = self._set_op()
    
    def gen_layer(self):
        N = self._q_N
        indices = []
        for i in range(N):
            index_pair = [i, (i + 1) % N] #boundary condition that N+1 = 0
            indices.append(index_pair)
        layer = [self._rotator(index_pair, N) for index_pair in indices]
        return layer
    
    def set_theta(self, theta):
        self._theta = theta
        for gate in self._layer:
            gate.set_theta(theta)
        self._operation = self._set_op()

    def _set_op(self):
        
        operation = prod(self._layer[::-1])
        return operation

    def __repr__(self):
        return f"RR block of {self._layer}"
    
    
def fsim_gate(theta, phi, N=None, control=0, target=1):
    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return qt.qip.operations.gate_expand_2toN(fsim_gate(theta, phi), N, control, target)
    return qt.Qobj([[1,                   0,                   0,                  0],
                    [0,       np.cos(theta), -1j * np.sin(theta),                  0],
                    [0, -1j * np.sin(theta),       np.cos(theta),                  0],
                    [0,                   0,                   0, np.exp(-1j * phi)]],
                    dims=[[2, 2], [2, 2]])

def fsim_gate_d_theta(theta, phi, N=None, control=0, target=1):
    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return qt.qip.operations.gate_expand_2toN(fsim_gate_d_theta(theta, phi), N, control, target)
    return qt.Qobj([[1,                   0,                   0,                  0],
                    [0,       -1 * np.sin(theta), 1j * np.cos(theta),                  0],
                    [0, 1j * np.cos(theta),      -1 *  np.sin(theta),                  0],
                    [0,                   0,                   0, np.exp(-1j * phi)]],
                    dims=[[2, 2], [2, 2]])

def fsim_gate_d_phi(theta, phi, N=None, control=0, target=1):
    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return qt.qip.operations.gate_expand_2toN(fsim_gate_d_phi(theta, phi), N, control, target)
    return qt.Qobj([[1,                   0,                   0,                  0],
                    [0,       np.cos(theta), -1j * np.sin(theta),                  0],
                    [0, -1j * np.sin(theta),       np.cos(theta),                  0],
                    [0,                   0,                   0, -1j * np.exp(-1j * phi)]],
                    dims=[[2, 2], [2, 2]])

class fSim(PRot):
    def __init__(self, qs_on, q_N):
        self._q1, self._q2 = qs_on
        self._q_N = q_N
        self._theta = 0
        self._phi = 0
        self._is_param = True
        self._param_count = 2

        self._operation = self._set_op()

    def _set_op(self):
        return fsim_gate(self._theta, self._phi, N=self._q_N, control=self._q1, target=self._q2)

    def set_theta(self, theta):
        self._theta = theta
        self._operation = self._set_op()

    def set_phi(self, phi):
        self._phi = phi
        self._operation = self._set_op()

    def derivative(self, param):
        if param == 1: #i.e d_theta
            deriv = fsim_gate_d_theta(self._theta, self._phi, N=self._q_N, control=self._q1, target=self._q2)
        elif param == 2: #i.e d_phi
            deriv = fsim_gate_d_phi(self._theta, self._phi, N=self._q_N, control=self._q1, target=self._q2)
        return deriv

    def flip_pauli(self):
        pass

    def __repr__(self):
        name = type(self).__name__
        angle1 = self._theta
        angle2 = self._phi
        string = f"{name}({angle1:.2f},{angle2:.2f})@q{self._q1, self._q2}"
        return string


#%% =============================CIRCUIT=============================


class PQC():
    """A class to define an n qubit wide, p layer deep Parameterised Quantum
    Circuit."""

    def __init__(self, n_qubits):
        self._n_qubits = n_qubits
        self._n_layers = 0
        self._layers = []
        if n_qubits >= 2:
            self.set_H('ZZ')
        self.initial_state = qt.tensor([qt.basis(2, 0) for i in range(self._n_qubits)])
        self._quantum_state = self.initial_state

    def set_H(self, H):
        if H == 'ZZ':
            Z0 = genFockOp(qt.sigmaz(), 0, self._n_qubits, 2)
            Z1 = genFockOp(qt.sigmaz(), 1, self._n_qubits, 2)
            self.H = Z0 * Z1
        else:
            self.H = H

    def set_initial_state(self, state):
        self.initial_state = qt.tensor([state for i in range(self._n_qubits)])

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

    def get_layer(self, pos):
        return self._layers[pos]

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
            param_count += gate._param_count
            for i in range(gate._param_count):
                param_count += 1
                self._parameterised.append(param_count)
            else:
                self._parameterised.append(-1)
        self.n_params = len([i for i in self._parameterised if i > -1])

    def get_params(self):
        angles = []
        for g in self.gates:
            if g._param_count == 2:
                angles.append(g._theta)
                angles.append(g._phi)
            elif g._param_count == 1:
                angles.append(g._theta)
        return angles

    def set_params(self, angles=[]):
        """Set the parameters of every parameterised gate (i.e inherits from PRot)
        in the circuit. Can set either randomly or from a specified list"""
        parameterised = [g for g in self.gates if g._is_param]
        param_counter = 0
        for g in parameterised:
            if g._param_count == 2:
                if angles != []:
                    angle1 = angles[param_counter]
                    angle2 = angles[param_counter + 1]
                else: #use random params
                    angle1 = rng.random(1)[0] * 2 * np.pi
                    angle2 = rng.random(1)[0] * 2 * np.pi
                g.set_theta(angle1)
                g.set_phi(angle2)
                param_counter += 2

            elif g._param_count == 1:
                if angles != []:
                    angle1 = angles[param_counter]
                else: #use random params
                    angle1 = rng.random(1)[0] * 2 * np.pi
                g.set_theta(angle1)
                param_counter += 1

    def run(self, angles=[]):
        """Set |psi> of a PQC by multiplying the basis state by the gates."""
        circuit_state = self.initial_state
        self.set_params(angles=angles)
        for g in self.gates:
            circuit_state = g * circuit_state
        return circuit_state

    def gen_quantum_state(self, energy_out=False):
        """Get a Qobj of |psi> for measurements."""
        self._quantum_state = qt.Qobj(self.run())
        if energy_out is True:
            e = self.cost()
            print(f"Energy of state is {e}")
        return self._quantum_state

    def cost(self, theta):
        """Get energy of |psi>, the initial quantum state"""
        self._quantum_state = self.run(angles=theta)
        psi = self._quantum_state
        energy = qt.expect(self.H, psi)
        return energy

    def _fidelity(self, target_state):
        #get fidelity w.r.t target state
        fidelity = np.abs(self._quantum_state.overlap(target_state))**2
        return fidelity

    def flip_deriv(self):
        parameterised = [g for g in self.gates if g._is_param]
        for g in parameterised:
            g.flip_pauli()

    def take_derivative(self, g_on, param=0):
        """Get the derivative of the ith parameter of the circuit and return
        the circuit where the ith gate is multiplied by its derivative."""
        #need to find which gate the ith parameterised gate is
        g_loc = self.gates.index(g_on) 
        #copy so don't overwrite later - much better than deepcopying whole circuit!
        gate = copy(self.gates[g_loc])
        #find the derivative using the gate's derivative method
        if param == 0:
            deriv = gate.derivative()
        else: #find ith deriv for multi parameterised gates
            deriv = gate.derivative(param)
        #set pth gate to be deriv * gate
        self.gates[g_loc] = deriv * gate
        #act the derivative of circuit on |0>
        circuit_state = self.initial_state
        for g in self.gates:
            circuit_state = g * circuit_state
        #reset the gate back to what it was originally
        self.gates[g_loc] = gate
        return circuit_state

    def get_gradients(self):
        """Get the n_params circuits with the ith derivative multiplied in and
        then apply them to the basis state."""
        gradient_state_list = []
        parameterised = [i for i in self.gates if i._param_count > 0]
        for g in parameterised:
            if g._param_count == 1:
                gradient = self.take_derivative(g)
                gradient_state_list.append(gradient)
            elif g._param_count == 2:
                gradient1 = self.take_derivative(g, param=1)
                gradient2 = self.take_derivative(g, param=2)
                gradient_state_list.append(gradient1)
                gradient_state_list.append(gradient2)
        return gradient_state_list

    def __repr__(self):
        line = f"A {self._n_qubits} qubit, {self._n_layers} layer deep PQC. \n"
        for count, l in enumerate(self._layers):
            line += f"Layer {count}: {l} \n"
        return line
