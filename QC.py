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
    $entanglers. Can then generate circuit from set params. For usage see test.py,
    but initialise w/ circuit = QuantumCircuit(4, 3 "chain", "cnot") to intialise
    a 4 qubit, 3 layer chain connected, CNOT entangled PQC. Then run it with
    circuit.run() to generate gradients for each param.
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
        self._gradient_list = np.zeros(self._n_params)

        self._gen_single_qubit_ops() #initialise ops

    def _gen_single_qubit_ops(self):
        """Returns the operator arrays as np arrays and saves them as attributes bc they're useful"""
        n_qubit_range = range(0, self._n_qubits)
        levels = 2 #don't know what this is
        self._opZ = [genFockOp(qt.sigmaz(), i, self._n_qubits, levels) for i in n_qubit_range]
        self._opY = [genFockOp(qt.sigmay(), i, self._n_qubits, levels) for i in n_qubit_range]
        self._opX = [genFockOp(qt.sigmax(), i, self._n_qubits, levels) for i in n_qubit_range]
        self._opId = genFockOp(qt.qeye(levels), 0, self._n_qubits)

        self._H = self._opZ[0] * self._opZ[1] #ZZ operator on first 2 qubits - why this hamiltonian?

    def _gen_initial_rotations(self):
        """
        Generate the initial parameters (theta) of the QC i.e both the type of
        rotation (X,Y,Z) and the angle (randomised between 0 and 2 pi).
        Returns:
            initial_angles: [float, ...]
            List of angles that a given gate rotates by, between 0 and 2 pi
            initial_pauli: [int, ...]
            List of what type of Pauli matrix a given rotation is (i.e X, Y, Z)
        """
        size = [self._layers, self._n_qubits]
        #angles for circuit - this is the theta/parameters
        initial_angles = rng.random(size) * 2 * np.pi
        #rotations for circuit in each layer, 1: X, 2: Y, 3: Z
        initial_pauli = rng.integers(1, 4, size)
        return initial_angles, initial_pauli

    def _gen_entanglement_indices(self):
        """
        Generate the qubit index pairs for connected qubits based on QC topology
        Returns:
            entangling_gate_indices: [[int, int], ...]
            A list of list (pairs) of integers that represent gate connections
            in the QC.
        TODO: add the alternating topolgy here! -how does it work for odd n_qubit?
        """
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
        """
        Given connect qubit indices and type of entangler, generate the entangling layer
        Returns:
            entangling_layer: qutip.qobj.Qobj
            A 16 by 16 (or 2*4 by 2*4 in tensor product space) matrix that represents
            the entangling layer operation.
        """
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
        """
        Apply initial hadamard rotation to basis state of n_qubits in |0> state
        Returns:
            applied_rotations: qutip.qobj.Qobj
            A ket representing the action of initial hadamard gate on basis stae,
            16 by 1 / 2**4 by 2**0 vector.
        """
        levels = 2 #is levels qubit dimension i.e always 2
        initial_state = qt.tensor([qt.basis(levels, 0) for i in range(self._n_qubits)])
        hadamards = [qt.qip.operations.ry(np.pi / 4) for i in range(self._n_qubits)]
        applied_rotations = qt.tensor(hadamards) * initial_state
        return applied_rotations

    def _add_to_rotation_operations(self, angle, ini_pauli_sigma, rot_op):
        """
        For given angle and type of Pauli matrix, make the correct rotation gate
        from qutip and append it in place to the rot_op array.
        Returns:
            rot_op [qt.qobj.Qobj, ...]
            A list of each rotation gate, a qutip operator matrix.
        """
        ops = qt.qip.operations
        if ini_pauli_sigma == 1: #X
            rot_op.append(ops.rx(angle))
        elif ini_pauli_sigma == 2: #Y
            rot_op.append(ops.ry(angle))
        elif ini_pauli_sigma == 3: #Z
            rot_op.append(ops.rz(angle))
        return rot_op

    def _multiply_in_derivative(self, ini_pauli_sigma, qubit, circuit_state):
        """
        Rotators have form G=e^(-igt), g=sigma_k/2 so derivative of rotator is
        -i/2*sigma_k * G. Then deriv * circuit state calculates the impact of
        taking derivative w.r.t that parameter.
        """
        if ini_pauli_sigma == 1: #X
            deriv = (-1j * self._opX[qubit] / 2)
        elif ini_pauli_sigma == 2: #Y
            deriv = (-1j * self._opY[qubit] / 2)
        elif ini_pauli_sigma == 3: #Z
            deriv = (-1j * self._opZ[qubit] / 2)
        return deriv * circuit_state

    def _update_circuit_state(self, param, ini_ang, ini_pauli, entng_layer):
        counter = 0
        circuit_state = self._gen_intial_state() #this resets circuit each time so not sequentially updating every param

        for layer in range(self._layers):
            rot_op = []

            for qubit in range(self._n_qubits):
                angle = ini_ang[layer][qubit]
                ini_pauli_sigma = ini_pauli[layer][qubit]
                rot_op = self._add_to_rotation_operations(angle, ini_pauli_sigma, rot_op)

                if counter == param: #multiply in derivative of rotation operator
                    circuit_state = self._multiply_in_derivative(ini_pauli_sigma, qubit, circuit_state)
                counter += 1

            circuit_state = qt.tensor(rot_op) * circuit_state
            circuit_state = entng_layer * circuit_state
        return circuit_state

    def gen_quantum_state(self):
        entangling_gate_indices = self._gen_entanglement_indices()
        entangling_layer = self._gen_entangling_layer(entangling_gate_indices)

        initial_angles, initial_pauli = self._gen_initial_rotations()
        initial_circuit_state = self._update_circuit_state(-1, initial_angles, initial_pauli, entangling_layer)

        self._quantum_state = qt.Qobj(initial_circuit_state) #quantum state generated by circuit - side effect here
        energy = qt.expect(self._H, self._quantum_state)
        #print(f"Energy of state is {energy}")
        return entangling_layer, initial_angles, initial_pauli

    def run(self):
        """
        First, generate the entangling layer and initial rotators. On -1th
        iteration calculate inital quantum state then for next n_param iterations
        calculate the gradient for that parameter, which is needed to optimise
        the parameter. Returns (i.e mutates) the gradient state list and gradient
        list and outputs the energy of the quantum state.
        """
        ent_layer, ini_ang, ini_pauli = self.gen_quantum_state()

        for param in range(0, self._n_params):
            circuit_state = self._update_circuit_state(param, ini_ang, ini_pauli, ent_layer)
            self._gradient_state_list.append(qt.Qobj(circuit_state)) #state with gradient applied for p-th parameter
            #gradient of circuit is given by 2*real(<\psi|H|\partial_p\psi>)
            overlap = self._quantum_state.overlap(self._H * circuit_state)
            self._gradient_list[param] = 2 * np.real(overlap)
