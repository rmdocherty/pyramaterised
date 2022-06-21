#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 11:22:14 2021

@author: ronan
"""
#%% Imports
from .gates import *


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
        """Set Hamiltonian PQC uses for training and gradient calculation. Default is 'ZZ'
        Hamiltonian but custom qutip hamiltonians can be set."""
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
                if type(angles) != str:
                    angle1 = angles[param_counter]
                    angle2 = angles[param_counter + 1]
                elif angles == "random": #use random params
                    angle1 = rng.random(1)[0] * 2 * np.pi
                    angle2 = rng.random(1)[0] * 2 * np.pi
                else:
                    raise Exception("No parameters supplied!")
                g.set_theta(angle1)
                g.set_phi(angle2)
                param_counter += 2

            elif g.param_count == 1:
                if type(angles) != str:
                    angle1 = angles[param_counter]
                elif angles == "random": #use random params
                    angle1 = rng.random(1)[0] * 2 * np.pi
                else:
                    raise Exception("No parameters supplied!")
                g.set_theta(angle1)
                param_counter += 1
    
    def run(self, angles: Union[list[Angle], Literal["random"]]) -> qt.Qobj:
        """Get |psi> of a PQC by multiplying the basis state by the gates when gates
        parameterised by angles."""
        circuit_state = self.initial_state
        self.set_params(angles=angles)
        for g in self.gates:
            circuit_state = g * circuit_state
        return circuit_state

    def update_state(self, angles: Union[list[Angle], Literal["random"]]) -> qt.Qobj:
        """Set quantum state of circuit from angles and return it. Modifies an attribute."""
        self.state = qt.Qobj(self.run(angles))
        return self.state

    def cost(self, angles: Union[list[Angle], Literal["random"]]) -> float:
        """Get energy of |psi>, the initial quantum state via <psi|H|psi>"""
        self.state = self.run(angles=angles)
        psi = self.state
        energy: float = qt.expect(self.H, psi)
        return energy

    def fidelity(self, target_state: qt.Qobj) -> float:
        """Compute and return F = |<psi|phi>|**2"""
        fidelity: float = np.abs(self.state.overlap(target_state))**2
        return fidelity

    def flip_deriv(self) -> None:
        parameterised: list[Gate] = [g for g in self.gates if g.is_param]
        for g in parameterised:
            g.flip_pauli()

    def take_derivative(self, g_on: Gate, param: Literal[0, 1, 2]=0) -> Gradient:
        """Get the derivative of the ith parameter of the circuit and return
        the circuit where the ith gate is multiplied by its derivative.
        Returns:
            circuit_state: Gradient of circuit w.r.t ith parameter.
        """
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
        then apply them to the basis state.
        Returns:
            gradient_state_list: list of gradients with respect to each parameter in the circuit
        """
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
