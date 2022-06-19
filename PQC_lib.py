#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 11:22:14 2021

@author: ronan
"""
#%% Imports
import qutip as qt
import numpy as np
from itertools import permutations
from copy import copy, deepcopy

from helper_functions import genFockOp, prod
from typing import Tuple, Union, Type, Literal

rng = np.random.default_rng(1)
#%% =============================TYPES=============================
QuantumGate = Union['Gate', qt.Qobj]
DoubleParamGate = 'fSim'
Gradient: qt.Qobj
# Won't use more than 20 qubits in simulations
QubitIndex = int #Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
QubitList = Union[list[QubitIndex], tuple[QubitIndex, ...]]
# Can't have a circuit with 0 qubits
QubitNumber = int #Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
Angle = Union[int, float]

Layer = list['Gate']
RotationLayer = list['PRot']
EntanglingLayer = list['EntGate']
#%% =============================GATES=============================


def iden(N: QubitNumber) -> qt.Qobj:
    return qt.tensor([qt.qeye(2) for i in range(N)])


class Gate():
    """Parent class for all gate types to inherit from - describes the behaviour
    when any Gate (or derived classes) is multiplied. Want to ensure that the
    gate always uses its underlying qutip gate representation when multiplied
    or added, so overwrite the mul, rmul (*) and add, radd (+) functions."""

    def __init__(self, q_N: QubitNumber) -> None:
        self.q_N: QubitNumber = q_N
        self.operation: qt.Qobj = iden(q_N)
        self.param_count: int = 0
        self.is_param: bool = False
        self.theta: Angle = 0
        self.phi: Angle = 0

    def __mul__(self, b: QuantumGate) -> QuantumGate:
        if isinstance(b, Gate):
            return self.operation * b.operation
        else:
            return self.operation * b

    def __rmul__(self, b: QuantumGate) -> QuantumGate:
        if isinstance(b, Gate):
            return b.operation * self.operation
        else:
            return b * self.operation

    def __add__(self, b: QuantumGate) -> QuantumGate:
        if isinstance(b, Gate):
            return self.operation + b.operation
        else:
            return self.operation + b

    def __radd__(self, b: QuantumGate) -> QuantumGate:
        if isinstance(b, Gate):
            return b.operation + self.operation
        else:
            return b + self.operation
    
    def set_theta(self, theta: Angle) -> None:
        return

    def set_phi(self, phi: Angle) -> None:
        return

    def derivative(self) -> Gradient:
        return iden(self.q_N)
    
    def parameterised_derivative(self, param: Literal[1,2]) -> Gradient:
        return self.derivative()

    def flip_pauli(self) -> None:
        pass

#%% Rotation gates


class PRot(Gate):
    """A class to described how parametrised rotation gates work - they have a
    qubit they operate on, a total number of qubits in the system (so gate can
    be extended to that dimension) and an angle that the gate rotates by."""

    def __init__(self, q_on: QubitIndex, q_N: QubitNumber) -> None:
        self.q_on: QubitIndex = q_on
        self.q_N: QubitNumber = q_N
        self.theta: Angle = 0
        self.is_param: bool = True
        self.param_count: int = 1

        self.set_properties()
        self.fock: qt.Qobj = genFockOp(self.pauli, self.q_on, self.q_N, 2)
        self.operation: qt.Qobj = self.get_op()

    def get_op(self) -> qt.Qobj:
        return self.gate(self.theta, N=self.q_N, target=self.q_on)
    
    def set_properties(self) -> None:
        self.gate: qt.Qobj = iden
        self.pauli: qt.Qobj = iden

    def set_theta(self, theta: Angle) -> None:
        self.theta = theta
        self.operation = self.get_op()

    def derivative(self) -> Gradient:
        """Take the derivative of the PRot - this generates the pauli gate
        associated with the gate type (i.e R_x -> sigma_x) operating on given
        qubit and multiplies it by j/2."""
        deriv: qt.Qobj = -1j * self.fock / 2
        return deriv

    def flip_pauli(self) -> None:
        self.pauli = -1 * self.pauli

    def __repr__(self) -> str:
        name: str = type(self).__name__
        angle: float = self.theta
        string: str = f"{name}({angle:.2f})@q{self.q_on}"
        return string

class I(PRot):
    def set_properties(self) -> None:
        self.gate: qt.Qobj = iden
        self.pauli: qt.Qobj = iden(self.q_N)

    def get_op(self) -> qt.Qobj:
        return iden(self.q_N)
    

class R_x(PRot):
    def set_properties(self) -> None:
        self.gate: qt.Qobj = qt.qip.operations.rx
        self.pauli: qt.Qobj = qt.sigmax()


class R_y(PRot):
    def set_properties(self) -> None:
        self.gate: qt.Qobj = qt.qip.operations.ry
        self.pauli: qt.Qobj = qt.sigmay()


class R_z(PRot):
    def set_properties(self) -> None:
        self.gate: qt.Qobj = qt.qip.operations.rz
        self.pauli: qt.Qobj = qt.sigmaz()


#%% Fermionic specific gates

class negative_R_z(R_z):
    def set_theta(self, theta: Angle) -> None:
        self.theta = -1 * theta
        self.operation = self.get_op()
    
    def derivative(self) -> Gradient:
        deriv: qt.Qobj = 1j * self.fock / 2
        return deriv


class offset_R_z(R_z):
    def __init__(self, q_on: QubitIndex, q_N: QubitNumber, offset: Angle) -> None:
        self.q_on: QubitIndex = q_on
        self.q_N: QubitNumber = q_N
        self.theta: Angle = 0
        self.offset: Angle = offset 
        self.is_param: bool = True
        self.param_count: int = 1


        self.set_properties()
        self.fock: qt.Qobj = genFockOp(self.pauli, self.q_on, self.q_N, 2)
        self.operation: qt.Qobj = self.get_op()

    def set_theta(self, theta: Angle):
        self.theta = theta + self.offset
        self.operation = self.get_op() 

#%% Fixed angle single-qubit rotations


class H(PRot):
    """Hadamard gate."""

    def __init__(self, q_on: QubitIndex, q_N: QubitNumber) -> None:
        self.q_on: QubitIndex = q_on
        self.q_N: QubitNumber = q_N
        self.theta: Angle = np.pi / 2
        self.is_param: bool = False
        self.param_count: int = 0
        self.operation: qt.Qobj = self.get_op()

    def set_theta(self, angle: Angle) -> None:
        """angle added as argument but not used in case we call set_theta on this class."""
        return None

    def get_op(self) -> qt.Qobj:
        """Hadamard gate is just sigma_x * R_y(pi/2)"""
        ops = qt.qip.operations
        self.gate: qt.Qobj = ops.ry
        return ops.x_gate(self.q_N, self.q_on) * self.gate(np.pi / 2, N=self.q_N, target=self.q_on)


class sqrtH(H):
    def get_op(self) -> qt.Qobj:
        ops = qt.qip.operations
        self.gate: qt.Qobj = ops.ry
        return np.sqrt(ops.x_gate(self.q_N, self.q_on) * self.gate(np.pi / 2, N=self.q_N, target=self.q_on))

class X(H):
    def get_op(self) -> qt.Qobj:
        """Pauli X gate"""
        ops = qt.qip.operations
        return ops.x_gate(self.q_N, self.q_on)


class fixed_R_y(R_y):
    """Fixed R_y rotation by angle theta. Isn't parameterised and angle can't
    be changed after initialization."""

    def __init__(self, q_on: QubitIndex, q_N: QubitNumber, theta: Angle) -> None:
        self.q_on: QubitIndex = q_on
        self.q_N: QubitNumber = q_N
        self.theta: Angle = theta
        self.is_param: bool = False
        self.param_count: int = 0
        self.set_properties()
        self.operation: qt.Qobj = self.get_op()

    def set_theta(self, theta: Angle) -> None:
        return None


class fixed_R_z(R_z):
    """Fixed R_y rotation by angle theta. Isn't parameterised and angle can't
    be changed after initialization."""

    def __init__(self, q_on: QubitIndex, q_N: QubitNumber, theta: Angle) -> None:
        self.q_on: QubitIndex = q_on
        self.q_N: QubitNumber = q_N
        self.theta: Angle = theta
        self.is_param: bool = False
        self.param_count: int = 0
        self.set_properties()
        self.operation: qt.Qobj = self.get_op()

    def set_theta(self, theta: Angle) -> None:
        return None


class S(H):
    def get_op(self) -> qt.Qobj:
        self.theta = np.pi / 2
        ops = qt.qip.operations
        self.gate = ops.phasegate
        return self.gate(np.pi / 2, N=self.q_N, target=self.q_on)


class T(H):
    """T-gate."""

    def get_op(self) -> qt.Qobj:
        ops = qt.qip.operations
        self.gate = ops.t_gate
        return self.gate(N=self.q_N, target=self.q_on)

#%% Entangling gates


class EntGate(Gate):
    """A class to described how entangling gates work - they have the
    qubits they operate on (control and target) and a total number of qubits
    in the system. Works the same way as rotation gates, i.e changing the
    get_op() method to use the right qutip gate."""

    def __init__(self, qs_on: QubitList, q_N: QubitNumber) -> None:
        self.q1, self.q2 = qs_on[0], qs_on[1]
        self.q_N: QubitNumber = q_N
        self.is_param: bool = False
        self.param_count: int = 0
        self.operation: qt.Qobj = self.get_op()

    def get_op(self) -> qt.Qobj:
        self.gate = qt.qeye
        return qt.qeye(self.q_N)

    def __repr__(self) -> str:
        return f"{type(self).__name__}@q{self.q1},q{self.q2}"


class CNOT(EntGate):
    def get_op(self) -> qt.Qobj:
        gate = qt.qip.operations.cnot
        return gate(self.q_N, self.q1, self.q2)


class CPHASE(EntGate):
    def get_op(self) -> qt.Qobj:
        """The CPHASE gate not a real cphase gate, defined in papers as CZ gate."""
        gate = qt.qip.operations.cz_gate
        return gate(self.q_N, self.q1, self.q2)


class sqrtiSWAP(EntGate):
    def get_op(self) -> qt.Qobj:
        gate = qt.qip.operations.sqrtiswap
        return gate(self.q_N, [self.q1, self.q2])


class CZ(EntGate):
    def get_op(self) -> qt.Qobj:
        gate = qt.qip.operations.cz_gate
        return gate(self.q_N, self.q1, self.q2)
    

#%% Block entangling gates


class CHAIN(EntGate):
    """Can make a Chain of a given entangling gate by generating all indices
    and making an entangler between all these indices."""

    def __init__(self, entangler: Type[EntGate], q_N: QubitNumber) -> None:
        self.entangler: Type[EntGate] = entangler
        self.q_N: QubitNumber = q_N
        self.is_param: bool = False
        self.param_count: int = 0
        self.operation: qt.Qobj = self.get_op()

    def get_op(self) -> qt.Qobj:
        N: QubitNumber = self.q_N
        top_connections: list[QubitList] = [[2 * j, 2 * j + 1] for j in range(N // 2)]
        bottom_connections: list[QubitList] = [[2 * j + 1, 2 * j + 2] for j in range((N - 1) // 2)]
        indices: list[QubitList] = top_connections + bottom_connections
        entangling_layer: qt.Qobj = prod([self.entangler(index_pair, N) for index_pair in indices][::-1])
        return entangling_layer

    def __repr__(self) -> str:
        return f"CHAIN connected {self.entangler.__name__}s"


class ALLTOALL(EntGate):
    """Define AllToAll in similar way to Chain block for a generic entangler."""

    def __init__(self, entangler: Type[EntGate], q_N: QubitNumber) -> None:
        self.entangler: Type[EntGate] = entangler
        self.q_N: QubitNumber = q_N
        self.is_param: bool = False
        self.param_count: int = 0
        self.operation: qt.Qobj = self.get_op()

    def get_op(self) -> qt.Qobj:
        N: QubitNumber = self.q_N
        indices = list(permutations(range(N), 2))
        entangling_layer = prod([self.entangler(index_pair, N) for index_pair in indices][::-1])
        return entangling_layer

    def __repr__(self):
        return f"ALL connected {self.entangler.__name__}s"

    
#%% Arbitrary Gate    
    
class ARBGATE(Gate):
    
    def __init__(self, Ham):
        self._Ham = Ham
        self.theta = 0
        self.is_param = True
        self.param_count: int = 1
        self.operation = self.get_op()
 
    def get_op(self) -> qt.Qobj:
        exponent = (-1j*self.theta*self._Ham)
        mat = exponent.expm()
        return mat

    def flip_pauli(self):
        self.pauli = -1 * self.pauli
         
    def set_theta(self, theta):
        self.theta = theta
        self.operation = self.get_op()  
        
    def derivative(self):
        deriv = -1j * self._Ham / 2
        return deriv
    
    def __repr__(self):
        name = type(self).__name__
        angle = self.theta
        string = f"{name}({angle:.2f})"
        return string
    
    
    
#%% Sharing Parameters


class shared_parameter(PRot):
    def __init__(self, layer: RotationLayer, q_N: QubitNumber, commute: bool=True):
        self.layer = layer
        self.theta: Angle = 0
        self.q_N: QubitNumber = q_N
        self.is_param: bool = True
        self.param_count: int = 1
        self.commute: bool = commute
        self.operation: qt.Qobj = self.get_op()

    def derivative(self) -> Gradient:
        """H=sum(H_s) -> d_theta U = d_theta (e^i*H*theta) = sum(H_s * U)"""
        deriv: qt.Qobj = 0
        if self.commute is True: # can do this i.f.f all gates in layer commute
            for g in self.layer:
                single_deriv: qt.Qobj = g.derivative()
                deriv = deriv + single_deriv
        else:
            for count, g in enumerate(self.layer):
                current_gate_deriv: qt.Qobj = g.derivative()
                new_layer: RotationLayer = copy(self.layer)
                new_layer[count] = current_gate_deriv * new_layer[count]
                deriv = deriv + prod(new_layer[::-1])
            deriv = deriv * self.operation.conj() #in pqc.take_derivative we multiply by gates at the end, need to get rid of that
        return deriv

    def set_theta(self, theta: Angle) -> None:
        self.theta = theta
        for gate in self.layer:
            gate.set_theta(theta)
        self.operation = self.get_op()

    def get_op(self) -> qt.Qobj:
        operation = prod(self.layer[::-1])
        return operation

    def flip_pauli(self) -> None:
        for g in self.layer:
            g.flip_pauli()

    def __repr__(self) -> str:
        return f"Block of {self.layer}"


#%% 2 qubit rotation gates

#big question - should the second qubit angle be -1 * theta ???


class RR(PRot):
    def __init__(self, qs_on: QubitList, q_N: QubitNumber) -> None:
        self.q1, self.q2 = qs_on[0], qs_on[1]
        self.q_N: QubitNumber = q_N
        self.theta = 0
        self.is_param = True
        self.param_count: int = 1
        self.set_properties()
        self.fock1: qt.Qobj = genFockOp(self.pauli, self.q1, self.q_N, 2) #hmmmmm
        self.fock2: qt.Qobj = genFockOp(self.pauli, self.q2, self.q_N, 2)
        self._I = iden(self.q_N)
        self.operation: qt.Qobj = self.get_op()

    def set_properties(self) -> None:
        self.gate: qt.Qobj = iden
        self.pauli: qt.Qobj = iden    

    def get_op(self) -> qt.Qobj: #analytic expression for exponent of pauli is cos(x)*I + sin(x)*pauli_str
        return np.cos(self.theta / 2) * self._I - 1j * np.sin(self.theta / 2) * self.fock1 * self.fock2

    def derivative(self) -> Gradient:
        """Derivative of XX/YY/ZZ is -i * tensor(sigmai, sigmai) /2"""
        deriv = -1j * (self.fock1 * self.fock2) / 2
        return deriv

    def __repr__(self) -> str:
        name: str = type(self).__name__
        angle: Angle = self.theta
        return f"{name}({angle:.2f})@q{self.q1},q{self.q2}"


class R_zz(RR):
    def set_properties(self) -> None:
        self.gate: qt.Qobj = qt.qip.operations.rz
        self.fock1 = qt.qip.operations.z_gate(N=self.q_N, target=self.q1)
        self.fock2 = qt.qip.operations.z_gate(N=self.q_N, target=self.q2)


class R_xx(RR):
    def set_properties(self) -> None:
        self.gate: qt.Qobj = qt.qip.operations.rx
        self.fock1 = qt.qip.operations.x_gate(N=self.q_N, target=self.q1)
        self.fock2 = qt.qip.operations.x_gate(N=self.q_N, target=self.q2)


class R_yy(RR):
    def set_properties(self) -> None:
        self.gate: qt.Qobj = qt.qip.operations.ry
        self.fock1 = qt.qip.operations.y_gate(N=self.q_N, target=self.q1)
        self.fock2 = qt.qip.operations.y_gate(N=self.q_N, target=self.q2)


class RR_block(shared_parameter):
    def __init__(self, rotator: Type[RR], q_N: QubitNumber):
        self.rotator: Type[RR] = rotator
        self.theta: Angle = 0
        self.q_N: QubitNumber = q_N
        self.is_param: bool = True
        self.param_count: int = 1
        self.layer: RotationLayer = self.gen_layer()
        self.operation: qt.Qobj = self.get_op()
        self.commute = True
    
    def gen_layer(self):
        N: int = self.q_N
        indices: list[QubitList] = []
        for i in range(N):
            index_pair: QubitList = [i, (i + 1) % N] #boundary condition that N+1 = 0
            indices.append(index_pair)
        layer: RotationLayer = [self.rotator(index_pair, N) for index_pair in indices]
        return layer
    
    def set_theta(self, theta: Angle) -> None:
        self.theta = theta
        for gate in self.layer:
            gate.set_theta(theta)
        self.operation = self.get_op()

    def get_op(self) -> qt.Qobj:
        operation = prod(self.layer[::-1])
        return operation

    def __repr__(self) -> str:
        return f"RR block of {self.layer}"
    
    
def fsim_gate(theta: Angle, phi: Angle, N=None, control: int=0, target: int=1) -> qt.Qobj:
    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return qt.qip.operations.gate_expand_2toN(fsim_gate(theta, phi), N, control, target)
    return qt.Qobj([[1,                   0,                   0,                  0],
                    [0,       np.cos(theta), -1j * np.sin(theta),                  0],
                    [0, -1j * np.sin(theta),       np.cos(theta),                  0],
                    [0,                   0,                   0, np.exp(-1j * phi)]],
                    dims=[[2, 2], [2, 2]])

def fsim_gate_d_theta(theta: Angle, phi: Angle, N=None, control: int=0, target: int=1) -> qt.Qobj:
    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return qt.qip.operations.gate_expand_2toN(fsim_gate_d_theta(theta, phi), N, control, target)
    return qt.Qobj([[1,                   0,                   0,                  0],
                    [0,       -1 * np.sin(theta), -1j * np.cos(theta),                  0],
                    [0, -1j * np.cos(theta),      -1 *  np.sin(theta),                  0],
                    [0,                   0,                   0, np.exp(-1j * phi)]],
                    dims=[[2, 2], [2, 2]])

def fsim_gate_d_phi(theta: Angle, phi: Angle, N=None, control: int=0, target: int=1) -> qt.Qobj:
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
    def __init__(self, qs_on: QubitList, q_N: QubitNumber) -> None:
        self.q1, self.q2 = qs_on
        self.q_N: QubitNumber = q_N
        self.theta: Angle = 0
        self.phi: Angle = 0
        self.is_param: bool = True
        self.param_count: int = 2

        self.operation: qt.Qobj = self.get_op()

    def get_op(self) -> qt.Qobj:
        return fsim_gate(self.theta, self.phi, N=self.q_N, control=self.q1, target=self.q2)

    def set_theta(self, theta: Angle) -> None:
        self.theta = theta
        self.operation = self.get_op()

    def set_phi(self, phi: Angle) -> None:
        self.phi = phi
        self.operation = self.get_op()

    def parameterised_derivative(self, param: Literal[1,2]) -> Gradient: # can only take deriv w.r.t 1st or 2nd param so use literal type
        deriv: qt.Qobj
        if param == 1: #i.e d_theta
            deriv = fsim_gate_d_theta(self.theta, self.phi, N=self.q_N, control=self.q1, target=self.q2)
        elif param == 2: #i.e d_phi
            deriv = fsim_gate_d_phi(self.theta, self.phi, N=self.q_N, control=self.q1, target=self.q2)
        return deriv

    def flip_pauli(self) -> None:
        pass

    def __repr__(self) -> str:
        name = type(self).__name__
        angle1 = self.theta
        angle2 = self.phi
        string = f"{name}({angle1:.2f},{angle2:.2f})@q{self.q1, self.q2}"
        return string

def fixed_fsim_gate(theta, N: int=None, control: int=0, target: int=1) -> qt.Qobj:
    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return qt.qip.operations.gate_expand_2toN(fixed_fsim_gate(theta), N, control, target)
    return qt.Qobj([[1,                   0,                   0,                  0],
                    [0,       np.cos(theta), -1j * np.sin(theta),                  0],
                    [0, -1j * np.sin(theta),       np.cos(theta),                  0],
                    [0,                   0,                   0,                  1]],
                    dims=[[2, 2], [2, 2]])

def fixed_fsim_gate_d_theta(theta, N: int=None, control: int=0, target: int=1) -> qt.Qobj:
    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return qt.qip.operations.gate_expand_2toN(fixed_fsim_gate_d_theta(theta), N, control, target)
    return qt.Qobj([[1,                   0,                   0,                  0],
                    [0,       -1 * np.sin(theta), -1j * np.cos(theta),                  0],
                    [0, -1j * np.cos(theta),      -1 *  np.sin(theta),                  0],
                    [0,                   0,                   0,                       1]],
                    dims=[[2, 2], [2, 2]])

class fixed_fSim(PRot):
    def __init__(self, qs_on, q_N):
        self.q1, self.q2 = qs_on
        self.q_N: QubitNumber = q_N
        self.theta = 0
        self.is_param = True
        self.param_count: int = 1

        self.operation = self.get_op()

    def get_op(self) -> qt.Qobj:
        return fixed_fsim_gate(self.theta, N=self.q_N, control=self.q1, target=self.q2)
    
    def derivative(self) -> Gradient:
        return fixed_fsim_gate_d_theta(self.theta, N=self.q_N, control=self.q1, target=self.q2)

    def flip_pauli(self) -> None:
        pass


#%% =============================CIRCUIT=============================


class PQC():
    """A class to define an n qubit wide, p layer deep Parameterised Quantum
    Circuit."""

    def __init__(self, n_qubits: QubitNumber) -> None:
        self.n_qubits: QubitNumber = n_qubits
        self.n_layers: int = 0
        self.layers: list[Layer] = []
        if n_qubits >= 2:
            self.set_H('ZZ')
        self.initial_state: qt.Qobj = qt.tensor([qt.basis(2, 0) for i in range(self.n_qubits)])
        self.state: qt.Qobj = self.initial_state

    def set_H(self, H: Union[str, qt.Qobj]):
        if H == 'ZZ':
            Z0: qt.Qobj = genFockOp(qt.sigmaz(), 0, self.n_qubits, 2)
            Z1: qt.Qobj = genFockOp(qt.sigmaz(), 1, self.n_qubits, 2)
            self.H = Z0 * Z1
        else:
            self.H = H

    def set_initial_state(self, state: qt.Qobj) -> None:
        self.initial_state = qt.tensor([state for i in range(self.n_qubits)])

    def add_layer(self, layer: Layer, n: int=1) -> None:
        """Add $n layers to PQC.layers"""
        for i in range(n):
            self.layers.append(deepcopy(layer))
        self.n_layers += n
        self.set_gates() #update PQC.gates when adding

    def set_layer(self, layer: Layer, pos: int) -> None:
        """Set nth layer of PQC to given layer. Will throw error if pos not available"""
        self.layers[pos] = deepcopy(layer)
        self.set_gates()

    def get_layer(self, pos: int) -> Layer:
        return self.layers[pos]

    def set_gates(self) -> None:
        """For each layer in layer, append it to gates. If layers is a nested list
        of layers, then gates is a flat list of each gate operation in order. Then
        iterate through gates and update a list that says if a gate is parameterised
        or not (-1), and which parameterised gate it is, i.e is it the 1st, 2nd, ..."""
        layers: Layer = []
        for i in self.layers:
            layers = layers + i
        self.gates: Layer = layers
        self.parameterised: list[int] = []
        total_param_count: int = 0
        for gate in self.gates:
            current_param_count: int = gate.param_count
            total_param_count += current_param_count
            for j in range(current_param_count):
                total_param_count += 1
                self.parameterised.append(total_param_count)
            else:
                self.parameterised.append(-1)
        self.n_params = total_param_count

    def get_params(self) -> list[Angle]:
        angles: list[Angle] = []
        for g in self.gates:
            if g.param_count == 2:
                angles.append(g.theta)
                angles.append(g.phi)
            elif g.param_count == 1:
                angles.append(g.theta)
            else:
                pass
        return angles

    def set_params(self, angles: Union[list[Angle], Literal["random"]]) -> None:
        """Set the parameters of every parameterised gate (i.e inherits from PRot)
        in the circuit. Can set either randomly or from a specified list"""
        parameterised: list[Gate] = [g for g in self.gates if g.is_param]
        param_counter: int = 0
        for g in parameterised:
            if g.param_count == 2:
                angle1: Angle
                angle2: Angle
                # start by checking if it's random so otherwise type check knows it will be type list[Angle]
                if angles == "random": #use random params
                    angle1 = rng.random(1)[0] * 2 * np.pi
                    angle2 = rng.random(1)[0] * 2 * np.pi
                elif angles != []:
                    angle1 = angles[param_counter]
                    angle2 = angles[param_counter + 1]
                else:
                    raise Exception("No parameters supplied!")
                g.set_theta(angle1)
                g.set_phi(angle2)
                param_counter += 2

            elif g.param_count == 1:
                if angles == "random": #use random params
                    angle1 = rng.random(1)[0] * 2 * np.pi
                elif angles != []:
                    angle1 = angles[param_counter]
                else:
                    raise Exception("No parameters supplied!")
                g.set_theta(angle1)
                param_counter += 1
    
    def run(self, angles: Union[list[Angle], Literal["random"]]) -> qt.Qobj:
        """Set |psi> of a PQC by multiplying the basis state by the gates when gates
        parameterised by angles."""
        circuit_state = self.initial_state
        self.set_params(angles=angles)
        for g in self.gates:
            circuit_state = g * circuit_state
        return circuit_state

    def gen_quantum_state(self, energy_out=False):
        """Get a Qobj of |psi> for measurements."""
        self.state = qt.Qobj(self.run())
        if energy_out is True:
            e = self.cost()
            print(f"Energy of state is {e}")
        return self.state

    def cost(self, theta):
        """Get energy of |psi>, the initial quantum state"""
        self.state = self.run(angles=theta)
        psi = self.state
        energy = qt.expect(self.H, psi)
        return energy

    def fidelity(self, target_state: qt.Qobj) -> float:
        #get fidelity w.r.t target state
        fidelity: float = np.abs(self.state.overlap(target_state))**2
        return fidelity

    def flip_deriv(self) -> None:
        parameterised: list[Gate] = [g for g in self.gates if g.is_param]
        for g in parameterised:
            g.flip_pauli()

    def take_derivative(self, g_on: Gate, param: Literal[0, 1, 2]=0) -> Gradient:
        """Get the derivative of the ith parameter of the circuit and return
        the circuit where the ith gate is multiplied by its derivative."""
        #need to find which gate the ith parameterised gate is
        g_loc: int = self.gates.index(g_on) 
        #copy so don't overwrite later - much better than deepcopying whole circuit!
        gate: Gate = copy(self.gates[g_loc])
        #find the derivative using the gate's derivative method
        if param == 0:
            deriv = gate.derivative()
        else: #find ith deriv for multi parameterised gates
            deriv = gate.parameterised_derivative(param)
        #set pth gate to be deriv * gate
        self.gates[g_loc] = deriv * gate
        #act the derivative of circuit on |0>
        circuit_state: qt.Qobj = self.initial_state
        for g in self.gates:
            circuit_state = g * circuit_state
        #reset the gate back to what it was originally
        self.gates[g_loc] = gate
        return circuit_state

    def get_gradients(self) -> list[Gradient]:
        """Get the n_params circuits with the ith derivative multiplied in and
        then apply them to the basis state."""
        gradient_state_list: list[Gradient] = []
        parameterised: list[Gate] = [i for i in self.gates if i.param_count > 0]
        for count, g in enumerate(parameterised):
            if g.param_count == 1:
                gradient: Gradient = self.take_derivative(g)
                gradient_state_list.append(gradient)
            elif g.param_count == 2:
                gradient1: Gradient = self.take_derivative(g, param=1) 
                g_prime = self.gates[count] # copy op in take deriv changes ref of ith gate so need to 'find' it again
                gradient2 = self.take_derivative(g_prime, param=2)
                gradient_state_list.append(gradient1)
                gradient_state_list.append(gradient2)
        return gradient_state_list

    def __repr__(self) -> str:
        line = f"A {self.n_qubits} qubit, {self.n_layers} layer deep PQC. \n"
        for count, l in enumerate(self.layers):
            line += f"Layer {count}: {l} \n"
        return line
