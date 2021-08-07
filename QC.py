#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 7 17:10:00 2021

@author: ronan
"""
import qutip as qt
import numpy as np
from helper_functions import prod, flatten, genFockOp, natural_num_check

rng = np.random.default_rng(1)


class QuantumCircuit():
    """
    QuantumCircuit.

    Class to generate a $layer deep VQC with $n_qubits in $topology layout using
    $entanglers. Can then generate circuit from set params.
    """

    def __init__(self, n_qubits, layers, topology, entangler):
        self._n_qubits = natural_num_check(n_qubits, "Number of qubits ")
        self._layers = natural_num_check(layers, "Number of layers ")
        self._n_params = n_qubits * layers

        if topology.lower() in ["chain", "c", "0"]:
            self._topology = "chain"
        elif topology.lower() in ["all", "all-to-all", "a", "1"]:
            self._topology = "all"
        else:
            raise Exception("Please supply a valid topology.")

        if entangler.lower() in ["cnot", "not", "0"]:
            self._entangler = "cnot"
        elif entangler.lower() in ["cphase", "phase", "1"]:
            self._entangler = "cphase"
        elif entangler.lower() in ["iswap", "swap", "2"]:
            self._entangler = "iswap"
        else:
            raise Exception("Please supply a valid entangler!")

        self._gradient_state_list = [] #to be appended to later
        self._gradient_list = np.zeros(12)
        self._gen_single_qubit_ops() #initialise ops

    def _gen_single_qubit_ops(self):
        """Returns the operator arrays as np arrays and saves them as attributes bc they're useful"""
        n_qubit_range = range(0, self._n_qubits)
        levels = 2 #don't know what this is
        self._opZ = [genFockOp(qt.sigmaz(), i, self._n_qubits, levels) for i in n_qubit_range]
        self._opY = [genFockOp(qt.sigmay(), i, self._n_qubits, levels) for i in n_qubit_range]
        self._opX = [genFockOp(qt.sigmax(), i, self._n_qubits, levels) for i in n_qubit_range]
        self._opId = genFockOp(qt.qeye(levels), 0, self._n_qubits)

        self._H = self._opZ[0] * self._opZ[1] #also used later

    def _gen_initial_rotations(self):
        size = [self._layers, self._n_qubits]
        #angles for circuit - this is the theta/parameters
        initial_angles = rng.random(size) * 2 * np.pi
        #rotations for circuit in each layer, 1: X, 2: Y, 3: Z
        initial_pauli = rng.integers(1, 4, size)
        return initial_angles, initial_pauli

    def _gen_entanglement_indices(self):
        """Generate the qubit index pairs for connected qubits based on QC topology"""
        if self._topology == "chain":
            top_connections = [[2 * j, 2 * j + 1] for j in range(self._n_qubits // 2)]
            bottom_connections = [[2 * j + 1, 2 * j + 2] for j in range((self._n_qubits - 1) // 2)]
            entangling_gate_indices = top_connections + bottom_connections #concat
        elif self._topology == "all":
            nested_temp_indices = []
            for i in range(self._n_qubits - 1):
                for j in range(i + 1, self._n_qubits):
                    nested_temp_indices.append(rng.perumtation([i, j]))
            entangling_gate_indices = flatten(nested_temp_indices)
        return entangling_gate_indices

    def _gen_entangling_layer(self, entangling_gate_indices):
        """Given connect qubit indices and type of entangler, generate the entangling layer
        (i.e large matrix/tensor product)"""
        if self._entangler == "cnot":
            gate = qt.qip.operations.cnot
            gate_list = [gate(self._n_qubits, j, k) for j, k in entangling_gate_indices]
        elif self._entangler == "cphase":
            gate = qt.qip.operations.csign
            gate_list = [gate(self._n_qubits, j, k) for j, k in entangling_gate_indices]
        elif self._entangler == "iswap":
            gate = qt.qip.operations.sqrtiswap
            gate_list = [gate(self._n_qubits, [j, k]) for j, k in entangling_gate_indices]
        entangling_layer = prod(gate_list[::-1]) #reverse so unitaries applied in correct order
        return entangling_layer

    def _gen_intial_state(self):
        """Apply initial hadamard rotation to basis state of n_qubits |0>"""
        levels = 2 #is levels qubit dimension i.e always 2
        initial_state = qt.tensor([qt.basis(levels, 0) for i in range(self._n_qubits)])
        hadamards = [qt.qip.operations.ry(np.pi / 4) for i in range(self._n_qubits)]
        applied_rotations = qt.tensor(hadamards) * initial_state
        return applied_rotations

    def _add_to_rotation_operations(self, angle, ini_pauli_sigma, rot_op):
        """Operates in place on rot_op and appends the angle parameterised correct rotation gate
        to rot_op."""
        ops = qt.qip.operations
        if ini_pauli_sigma == 1: #X
            rot_op.append(ops.rx(angle))
        elif ini_pauli_sigma == 2: #Y
            rot_op.append(ops.ry(angle))
        elif ini_pauli_sigma == 3: #Z
            rot_op.append(ops.rz(angle))
        return rot_op

    def _multiply_in_derivative(self, ini_pauli_sigma, qubit, circuit_state):
        if ini_pauli_sigma == 1: #X
            deriv = (-1j * self._opX[qubit] / 2)
        elif ini_pauli_sigma == 2: #Y
            deriv = (-1j * self._opY[qubit] / 2)
        elif ini_pauli_sigma == 3: #Z
            deriv = (-1j * self._opZ[qubit] / 2)
        return deriv * circuit_state

    def run(self):
        entangling_gate_indices = self._gen_entanglement_indices()
        entangling_layer = self._gen_entangling_layer(entangling_gate_indices)

        initial_angles, initial_pauli = self._gen_initial_rotations()

        for param in range(-1, self._n_params):
            counter = 0
            circuit_state = self._gen_intial_state()

            for layer in range(self._layers):
                rot_op = []

                for qubit in range(self._n_qubits):
                    angle = initial_angles[layer][qubit]
                    ini_pauli_sigma = initial_pauli[layer][qubit]
                    rot_op = self._add_to_rotation_operations(angle, ini_pauli_sigma, rot_op)

                    if counter == param: #multiply in derivative of rotation operator
                        circuit_state = self._multiply_in_derivative(ini_pauli_sigma, qubit, circuit_state)
                    counter += 1

                circuit_state = qt.tensor(rot_op) * circuit_state
                circuit_state = entangling_layer * circuit_state

            if param == -1:
                ground_state = qt.Qobj(circuit_state)
                energy = qt.expect(self._H, ground_state)
                print(f"Energy of state is {energy}")
            else:
                self._gradient_state_list.append(qt.Qobj(circuit_state))
                overlap = ground_state.overlap(self._H * circuit_state)
                self._gradient_list[param] = 2 * np.real(overlap)
                

        