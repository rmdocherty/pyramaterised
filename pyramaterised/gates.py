import qutip as qt
import numpy as np
import operator

from itertools import permutations
from copy import copy, deepcopy
from functools import reduce
from typing import Tuple, Union, Type, Literal, TypeAlias

rng = np.random.default_rng(1)
# %% =============================TYPES=============================
QuantumGate = Union["Gate", qt.Qobj]
DoubleParamGate = "fSim"
Gradient: TypeAlias = qt.Qobj
# Won't use more than 20 qubits in simulations
QubitIndex = (
    int  # Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
)
QubitList = Union[list[QubitIndex], tuple[QubitIndex, ...]]
# Can't have a circuit with 0 qubits
QubitNumber = int  # Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
Angle = Union[int, float]

Layer = list["Gate"]
RotationLayer = list["PRot"]
EntanglingLayer = list["EntGate"]
# %% =============================GATES=============================


def prod(factors):
    return reduce(operator.mul, factors, 1)


def flatten(l):
    return [item for sublist in l for item in sublist]


# tensors operators together
def genFockOp(op, position, size, levels=2, opdim=0):
    opList = [qt.qeye(levels) for x in range(size - opdim)]
    opList[position] = op
    return qt.tensor(opList)


def iden(N: QubitNumber) -> qt.Qobj:
    return qt.tensor([qt.qeye(2) for i in range(N)])


class Gate:
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

    def parameterised_derivative(self, param: Literal[1, 2]) -> Gradient:
        return self.derivative()

    def flip_pauli(self) -> None:
        pass


# %% Rotation gates


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


# %% Fermionic specific gates


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


# %% Fixed angle single-qubit rotations


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
        return ops.x_gate(self.q_N, self.q_on) * self.gate(
            np.pi / 2, N=self.q_N, target=self.q_on
        )


class sqrtH(H):
    def get_op(self) -> qt.Qobj:
        ops = qt.qip.operations
        self.gate: qt.Qobj = ops.ry
        return np.sqrt(
            ops.x_gate(self.q_N, self.q_on)
            * self.gate(np.pi / 2, N=self.q_N, target=self.q_on)
        )


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


# %% Entangling gates


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


# %% Block entangling gates


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
        bottom_connections: list[QubitList] = [
            [2 * j + 1, 2 * j + 2] for j in range((N - 1) // 2)
        ]
        indices: list[QubitList] = top_connections + bottom_connections
        entangling_layer: qt.Qobj = prod(
            [self.entangler(index_pair, N) for index_pair in indices][::-1]
        )
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
        entangling_layer = prod(
            [self.entangler(index_pair, N) for index_pair in indices][::-1]
        )
        return entangling_layer

    def __repr__(self):
        return f"ALL connected {self.entangler.__name__}s"


# %% Arbitrary Gate


class ARBGATE(Gate):
    def __init__(self, Ham):
        self._Ham = Ham
        self.theta = 0
        self.is_param = True
        self.param_count: int = 1
        self.operation = self.get_op()

    def get_op(self) -> qt.Qobj:
        exponent = -1j * self.theta * self._Ham
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


# %% Sharing Parameters


class shared_parameter(PRot):
    def __init__(self, layer: RotationLayer, q_N: QubitNumber, commute: bool = True):
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
        if self.commute is True:  # can do this i.f.f all gates in layer commute
            for g in self.layer:
                single_deriv: qt.Qobj = g.derivative()
                deriv = deriv + single_deriv
        else:
            for count, g in enumerate(self.layer):
                current_gate_deriv: qt.Qobj = g.derivative()
                new_layer: RotationLayer = copy(self.layer)
                new_layer[count] = current_gate_deriv * new_layer[count]
                deriv = deriv + prod(new_layer[::-1])
            deriv = (
                deriv * self.operation.conj()
            )  # in pqc.take_derivative we multiply by gates at the end, need to get rid of that
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


# %% 2 qubit rotation gates

# big question - should the second qubit angle be -1 * theta ???


class RR(PRot):
    def __init__(self, qs_on: QubitList, q_N: QubitNumber) -> None:
        self.q1, self.q2 = qs_on[0], qs_on[1]
        self.q_N: QubitNumber = q_N
        self.theta = 0
        self.is_param = True
        self.param_count: int = 1
        self.set_properties()
        self.fock1: qt.Qobj = genFockOp(self.pauli, self.q1, self.q_N, 2)  # hmmmmm
        self.fock2: qt.Qobj = genFockOp(self.pauli, self.q2, self.q_N, 2)
        self._I = iden(self.q_N)
        self.operation: qt.Qobj = self.get_op()

    def set_properties(self) -> None:
        self.gate: qt.Qobj = iden
        self.pauli: qt.Qobj = iden

    def get_op(
        self,
    ) -> (
        qt.Qobj
    ):  # analytic expression for exponent of pauli is cos(x)*I + sin(x)*pauli_str
        return (
            np.cos(self.theta / 2) * self._I
            - 1j * np.sin(self.theta / 2) * self.fock1 * self.fock2
        )

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
        self.pauli: qt.Qobj = qt.sigmaz()


class R_xx(RR):
    def set_properties(self) -> None:
        self.gate: qt.Qobj = qt.qip.operations.rx
        self.fock1 = qt.qip.operations.x_gate(N=self.q_N, target=self.q1)
        self.fock2 = qt.qip.operations.x_gate(N=self.q_N, target=self.q2)
        self.pauli: qt.Qobj = qt.sigmax()


class R_yy(RR):
    def set_properties(self) -> None:
        self.gate: qt.Qobj = qt.qip.operations.ry
        self.fock1 = qt.qip.operations.y_gate(N=self.q_N, target=self.q1)
        self.fock2 = qt.qip.operations.y_gate(N=self.q_N, target=self.q2)
        self.pauli: qt.Qobj = qt.sigmay()


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
            index_pair: QubitList = [i, (i + 1) % N]  # boundary condition that N+1 = 0
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


def fsim_gate(
    theta: Angle, phi: Angle, N=None, control: int = 0, target: int = 1
) -> qt.Qobj:
    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return qt.qip.operations.gate_expand_2toN(
            fsim_gate(theta, phi), N, control, target
        )
    return qt.Qobj(
        [
            [1, 0, 0, 0],
            [0, np.cos(theta), -1j * np.sin(theta), 0],
            [0, -1j * np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, np.exp(-1j * phi)],
        ],
        dims=[[2, 2], [2, 2]],
    )


def fsim_gate_d_theta(
    theta: Angle, phi: Angle, N=None, control: int = 0, target: int = 1
) -> qt.Qobj:
    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return qt.qip.operations.gate_expand_2toN(
            fsim_gate_d_theta(theta, phi), N, control, target
        )
    return qt.Qobj(
        [
            [1, 0, 0, 0],
            [0, -1 * np.sin(theta), -1j * np.cos(theta), 0],
            [0, -1j * np.cos(theta), -1 * np.sin(theta), 0],
            [0, 0, 0, np.exp(-1j * phi)],
        ],
        dims=[[2, 2], [2, 2]],
    )


def fsim_gate_d_phi(
    theta: Angle, phi: Angle, N=None, control: int = 0, target: int = 1
) -> qt.Qobj:
    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return qt.qip.operations.gate_expand_2toN(
            fsim_gate_d_phi(theta, phi), N, control, target
        )
    return qt.Qobj(
        [
            [1, 0, 0, 0],
            [0, np.cos(theta), -1j * np.sin(theta), 0],
            [0, -1j * np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, -1j * np.exp(-1j * phi)],
        ],
        dims=[[2, 2], [2, 2]],
    )


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
        return fsim_gate(
            self.theta, self.phi, N=self.q_N, control=self.q1, target=self.q2
        )

    def set_theta(self, theta: Angle) -> None:
        self.theta = theta
        self.operation = self.get_op()

    def set_phi(self, phi: Angle) -> None:
        self.phi = phi
        self.operation = self.get_op()

    def parameterised_derivative(
        self, param: Literal[1, 2]
    ) -> Gradient:  # can only take deriv w.r.t 1st or 2nd param so use literal type
        deriv: qt.Qobj
        if param == 1:  # i.e d_theta
            deriv = fsim_gate_d_theta(
                self.theta, self.phi, N=self.q_N, control=self.q1, target=self.q2
            )
        elif param == 2:  # i.e d_phi
            deriv = fsim_gate_d_phi(
                self.theta, self.phi, N=self.q_N, control=self.q1, target=self.q2
            )
        return deriv

    def flip_pauli(self) -> None:
        pass

    def __repr__(self) -> str:
        name = type(self).__name__
        angle1 = self.theta
        angle2 = self.phi
        string = f"{name}({angle1:.2f},{angle2:.2f})@q{self.q1, self.q2}"
        return string


def fixed_fsim_gate(theta, N: int = None, control: int = 0, target: int = 1) -> qt.Qobj:
    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return qt.qip.operations.gate_expand_2toN(
            fixed_fsim_gate(theta), N, control, target
        )
    return qt.Qobj(
        [
            [1, 0, 0, 0],
            [0, np.cos(theta), -1j * np.sin(theta), 0],
            [0, -1j * np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1],
        ],
        dims=[[2, 2], [2, 2]],
    )


def fixed_fsim_gate_d_theta(
    theta, N: int = None, control: int = 0, target: int = 1
) -> qt.Qobj:
    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return qt.qip.operations.gate_expand_2toN(
            fixed_fsim_gate_d_theta(theta), N, control, target
        )
    return qt.Qobj(
        [
            [1, 0, 0, 0],
            [0, -1 * np.sin(theta), -1j * np.cos(theta), 0],
            [0, -1j * np.cos(theta), -1 * np.sin(theta), 0],
            [0, 0, 0, 1],
        ],
        dims=[[2, 2], [2, 2]],
    )


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
        return fixed_fsim_gate_d_theta(
            self.theta, N=self.q_N, control=self.q1, target=self.q2
        )

    def flip_pauli(self) -> None:
        pass
