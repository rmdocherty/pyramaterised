#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 12:19:54 2021

@author: ronan
"""
from .circuit import *
import qutip as qt
import numpy as np
import random

# ============================== HE CIRCUITS ==============================
LEN_CLIFF_STRING = 3


def clifford_circuit_layers(p, N, method='random'):
    clifford_gates = [H, S, CZ, CNOT]
    layers = []
    qs = list(range(N))

    if method.lower() == 'random':
        for i in range(p):
            layer = []
            for n in range(N):
                gate = random.choice(clifford_gates)
                if issubclass(gate, PRot): #can't check is_param of this as not instantised yet - could make class variable?
                    q_on = random.randint(0, N - 1)
                    layer.append(gate(q_on, N))
                elif issubclass(gate, EntGate): #entangling gate
                    q_1, q_2 = random.sample(qs, k=2) #use sample so can't pick same option twice
                    layer.append(gate([q_1, q_2], N))
            layers.append(layer)
    else:
        single_qubit_ops = clifford_gates[:2]
        for i in range(p):
            layer = []
            qs_on = random.sample(qs, k=N // 2)
            strings = [random.choices(single_qubit_ops, k=LEN_CLIFF_STRING) for i in qs_on]
            #add single qubit clifford strings
            count = 0
            for string in strings:
                for gate in string:
                    layer.append(gate(qs_on[count], N))
                count += 1
            layer.append(CHAIN(CNOT, N)) #add entangling layer
            layers.append(layer)
    return layers


def gen_shift_list(p, N):
    #lots of -1 as paper indexing is 1-based and list-indexing 0 based
    A = [i for i in range(N // 2)]
    s = 1
    shift_list = np.zeros(2**(N // 2), dtype=np.int32) #we have at most 2^(N/2) layers
    while A != []:
        r = A.pop(0) #get first elem out of A
        shift_list[s - 1] = r #a_s
        qs = [i for i in range(1, s)] #count up from 1 to s-1
        for q in qs:
            shift_list[s + q - 1] = shift_list[q - 1] #a_s+q = a_q
        s = 2 * s
    return shift_list


def NPQC_layers(p, N):
    """Generate layers of the NPQC circuit which has identity QFIM when initialised with reference angles."""
    #started with fixed block of N R_y and N R_x as first layer
    initial_layer = [R_y(i, N) for i in range(N)] + [R_z(i, N) for i in range(N)]
    angles = [np.pi / 2 for i in range(N)] + [0 for i in range(N)]
    layers = [initial_layer]
    shift_list = gen_shift_list(p, N)
    for i in range(0, p - 1):
        p_layer = []
        a_l = shift_list[i]
        fixed_rots, cphases = [], []
        #U_ent layer - not paramterised and shouldn't be counted as such!
        for k in range(1, 1 + N // 2):
            q_on = 2 * k - 2
            rotation = fixed_R_y(q_on, N, np.pi / 2) #NB these fixed gates aren't parametrised and shouldn't be counted in angles
            fixed_rots.append(rotation)
            U_ent = CPHASE([q_on, ((q_on + 1) + 2 * a_l) % N], N)
            cphases.append(U_ent)
        p_layer = fixed_rots + cphases #need fixed r_y to come before c_phase

        #rotation layer - R_y then R_z on each kth qubit
        for k in range(1, N // 2 + 1):
            q_on = 2 * k - 2
            p_layer = p_layer + [R_y(q_on, N), R_z(q_on, N)]
            #R_y gates have theta_i = pi/2 for theta_r
            angles.append(np.pi / 2)
            #R_z gates have theta_i = 0 for theta_r
            angles.append(0)
        layers.append(p_layer)
    return layers, angles


def string_to_entangler(string):
    """Given entangler string, return which gate it corresponds to."""
    lower = string.lower()
    if lower == "cnot":
        entangler = CNOT
    elif lower == "cphase":
        entangler = CPHASE
    elif lower == "cz":
        entangler = CZ
    elif lower == "sqrtiswap":
        entangler = sqrtiSWAP
    else:
        raise Exception("Must supply a valid entangler!")
    return entangler

"""These circuits are defined in arXiv:1905.10876v1"""
def circuit_1_layers(p, N):
    layer = [R_x(i, N) for i in range(N)] + [R_y(i, N) for i in range(N)]
    layers = []
    for i in range(p):
        layers.append(layer)
    return layers


def circuit_2_layers(p, N, ent_str="cnot"):
    entangler = string_to_entangler(ent_str)
    layer = [R_x(i, N) for i in range(N)] + [R_z(i, N) for i in range(N)] + [CHAIN(entangler, N)]
    layers = []
    for i in range(p):
        layers.append(layer)
    return layers


def circuit_9_layers(p, N, ent_str="cphase"):
    entangler = string_to_entangler(ent_str)
    layer = [H(i, N) for i in range(N)] + [CHAIN(entangler, N)] + [R_x(i, N) for i in range(N)]
    layers = []
    for i in range(p):
        layers.append(layer)
    return layers


def qg_circuit_layers(p, N, ent_str="cnot"):
    """This circuit defined in arXiv:2102.01659v1"""
    entangler = string_to_entangler(ent_str)
    init_layer = [fixed_R_y(i, N, np.pi / 4) for i in range(N)]
    layer1 = [R_z(i, N) for i in range(N)] + [CHAIN(entangler, N)]
    layer2 = [R_x(i, N) for i in range(N)] + [CHAIN(entangler, N)]
    layer3 = [R_z(i, N) for i in range(N)] + [CHAIN(entangler, N)]
    layer = layer1 + layer2 + layer3
    layers = [init_layer]
    for i in range(p):
        layers.append(layer)
    return layers


def generic_HE_layers(p, N, ent_str="cnot"):
    """A generic Hardware Efficient """
    entangler = string_to_entangler(ent_str)
    init_layer = [fixed_R_y(i, N, np.pi / 4) for i in range(N)]
    layer = [R_y(i, N) for i in range(N)] + [R_z(i, N) for i in range(N)] + [CHAIN(entangler, N)]
    layers = [init_layer] 
    for i in range(p):
        layers.append(layer)
    return layers

def clifford_HE_layers(p, N, ent_str="cnot"):
    """A clifford circuit (when angles set to clifford angles)."""
    entangler = string_to_entangler(ent_str)
    layer = [R_y(i, N) for i in range(N)] + [R_z(i, N) for i in range(N)] + [CHAIN(entangler, N)]
    layers = [] 
    for i in range(p):
        layers.append(layer)
    return layers

def y_CPHASE_layers(p, N):
    """A y-CPHASE circuit of y rotations and CPHASe entanglers. Has interesting properties relative to other
    hardware efficient circuits."""
    layers = []
    layer = [R_y(i, N) for i in range(N)] + [CHAIN(CPHASE, N)]
    for i in range(p):
        layers.append(layer)
    return layers

def double_y_CPHASE_layers(p, N):
    """Same as y-CPHASE circuit but has same number of parameters as other HE circuits (to ensure properties
     not due to lower parameter number)."""
    layers = []
    layer = [R_y(i, N) for i in range(N)] + [R_y(i, N) for i in range(N)] + [CHAIN(CPHASE, N)]
    for i in range(p):
        layers.append(layer)
    return layers

# ============================== PROBLEM INSPIRED CIRCUITS ==============================


def TFIM_layers(p, N):
    """TFIM circuit based on Transverse-Field-Ising-Model and DOI:10.1103/prxquantum.1.020319"""
    initial_layer = [H(i, N) for i in range(N)]
    layers = [initial_layer]
    for i in range(p):
        first_half = [RR_block(R_zz, N)]
        second_layer = [R_x(i, N) for i in range(N)]
        second_half = [shared_parameter(second_layer, N)]
        layer = first_half + second_half
        layers.append(layer)
    return layers


def modified_TFIM_layers(p, N):
    layers = []
    for i in range(p):
        first_half = [RR_block(R_zz, N)]
        second_layer = [R_x(i, N) for i in range(N)]
        second_half = [shared_parameter(second_layer, N)]
        third_layer = [R_z(i, N) for i in range(N)]
        third_half = [shared_parameter(third_layer, N)]
        layer = first_half + second_half + third_half
        layers.append(layer)
    return layers


def TFIM_hamiltonian(N, g, h=0):
    """TFIM Hamiltonian - used for minimisation."""
    H = 0
    for i in range(N):
        i_plus = (i + 1) % N
        H += genFockOp(qt.sigmaz(), i, N) * genFockOp(qt.sigmaz(), i_plus, N) + g * genFockOp(qt.sigmax(), i, N) + h * genFockOp(qt.sigmaz(), i, N)
    H = -1 * H
    return H


def XXZ_layers(p, N, commute=False):
    """XXZ circuit from DOI:10.1103/prxquantum.1.020319"""
    even_indices = []
    odd_indices = []
    for i in range(1, (N // 2) + 1): # -1 to convert from 1 indexed paper defn to 0 indexed qubit lists
        even_bot, even_top = (2 * i - 1) - 1, (2 * i) - 1
        even_indices.append((even_bot, even_top))
        odd_bot, odd_top = (2 * i) - 1, ((2 * i + 1) - 1) % N #need mod N here
        odd_indices.append((odd_bot, odd_top))

    init_x_bits = [X(i, N) for i in range(N)]
    init_H = [H(i, N) for i in range(N) if i % 2 == 0] #H on even links
    init_CNOT = [CNOT([i, j], N) for i, j in even_indices]
    #layers = [init_x_bits, init_H, init_CNOT]
    layers = []
    for l in range(p):
        layer = []
        ZZ_1 = [R_zz((i, j), N) for i, j in odd_indices]
        YY_XX_1 = [R_yy((i, j), N) for i, j in odd_indices] + [R_xx((i, j), N) for i, j in odd_indices]
        ZZ_2 = [R_zz((i, j), N) for i, j in even_indices]
        YY_XX_2 = [R_yy((i, j), N) for i, j in even_indices] + [R_xx((i, j), N) for i, j in even_indices]
        theta = [shared_parameter(ZZ_1, N)]
        phi = [shared_parameter(YY_XX_1, N, commute=commute)]
        beta = [shared_parameter(ZZ_2, N)]
        gamma = [shared_parameter(YY_XX_2, N, commute=commute)]
        layer = theta + phi + beta + gamma
        layers.append(layer)
    return layers


def gen_theta_block(q1, q2, N):
    """Unit used in fermionic circuit."""
    block = []
    block.append(sqrtiSWAP([q1, q2], N))

    minus_rot = negative_R_z(q1, N)
    offset_rot = offset_R_z(q2, N, np.pi)
    shared_param = shared_parameter([minus_rot, offset_rot], N)
    block.append(shared_param)

    block.append(sqrtiSWAP([q1, q2], N))

    fixed = fixed_R_z(q2, N, np.pi)
    block.append(fixed)
    return block


def list_to_pairs(x):
    pairs = [(x[i], x[i + 1]) for i in range(0, len(x) - 1, 2)]
    return pairs


def fermionic_circuit_layers(p, N):
    """Fermionic circuit: diamond shaped arrangement of 'theta blocks'."""
    layers = []
    for j in range(p):
        blocks = []
        layer = []
        for n in range(1, 1 + N // 2):
            first_half = [i for i in np.arange(N // 2, N // 2 - n, -1)]
            first_half.reverse()
            snd_half = [i for i in np.arange(1 + N // 2, 1 + N // 2 + n, 1)]
            combined = first_half + snd_half
            pairs = list_to_pairs(combined)
            blocks.append(pairs)
        rev = list(reversed(blocks))[1:]
        blocks = blocks + rev # symmetry
        for b in blocks:
            layer = []
            for x in b:
                block = gen_theta_block(x[0] - 1, x[1] - 1, N) #-1 to go from indices -> qubits
                for bl in block:
                    layer.append(bl)
            layers.append(layer)
    return layers


def fSim_circuit_layers(p, N, rotator='y', fixed=False):
    """Circuit made up of google fSim gates and rotations. Can construct the zfsim gate by
    fixing the rotations to z gates (rather than the default y gates). Uses periodic
    boundary conditions to choose connections."""
    r = rotator.lower()
    if r == 'y':
        rot_gate = R_y
    elif r == 'x':
        rot_gate = R_x
    elif r == 'z':
        rot_gate = R_z
    else:
        raise Exception("Please supply a valid single qubit rotator")
    layers = []
    for l in range(p):
        layer = [rot_gate(i, N) for i in range(N)]
        if N % 2 == 0:
            offset = l % 2
            for i in range(offset, N, 2):
                if fixed is True:
                    gate = fixed_fSim([i, (i+1)%N], N)
                else:
                    gate = fSim([i, (i+1)%N], N)
                layer.append(gate)
        else:
            offset = l % N
            loop_at_boundary = offset % 2
            pairs = []
            indices = [i for i in range(N)]
            indices.pop(offset) #remove this guy from indices
            if loop_at_boundary == 1:
                bottom = indices.pop(0)
                top = indices.pop(-1)
                pairs.append((bottom, top))
            else:
                pass
            connect_indices = [(i, i + 1) for i in range(0, len(indices), 2)]
            pairs = pairs + [(indices[i], indices[j]) for i, j in connect_indices]
            for pair in pairs:
                if fixed is True:
                    gate = fixed_fSim([pair[0], pair[1]], N)
                else:
                    gate = fSim([pair[0], pair[1]], N)
                layer.append(gate)
            layer.append(rot_gate(offset, N))
        layers.append(layer)
    return layers

def add_layers(circuit, layers):
    for l in layers:
        circuit.add_layer(l)
    return circuit 


def generate_circuit(circuit_type, N, p, hamiltonian="ZZ", rotator='', shuffle=True):
    """Create N qubit, p layer circuit given its string representation."""
    circuit = PQC(N)

    if circuit_type == "NPQC":
        layers, theta_ref = NPQC_layers(p, N)
    elif circuit_type == "TFIM":
        layers = TFIM_layers(p, N)
    elif circuit_type == "TFIM_modified":
        layers = modified_TFIM_layers(p, N)
    elif circuit_type == "XXZ":
        layers = XXZ_layers(p, N)
        init_state = [qt.basis(2, 1) for i in range(N // 2)] + [qt.basis(2, 0) for i in range(N // 2, N)]
        if shuffle:
            random.shuffle(init_state)
        tensored = qt.tensor(init_state)
        circuit.initial_state = tensored
    elif circuit_type == "Circuit_1":
        layers = circuit_1_layers(p, N)
    elif circuit_type == "Circuit_2":
        layers = circuit_2_layers(p, N)
    elif circuit_type == "Circuit_9":
        layers = circuit_9_layers(p, N)
    elif circuit_type == "qg_circuit":
        layers = qg_circuit_layers(p, N)
    elif circuit_type == "generic_HE":
        layers = generic_HE_layers(p, N)
    elif circuit_type == "clifford":
        layers = clifford_HE_layers(p, N)
    elif circuit_type == "y_CPHASE":
        layers = y_CPHASE_layers(p, N)
    elif circuit_type == "double_y_CPHASE":
        layers = double_y_CPHASE_layers(p, N)
    elif circuit_type == "fermionic":
        layers = fermionic_circuit_layers(p, N)
        init_state = [qt.basis(2, 1) for i in range(N // 2)] + [qt.basis(2, 0) for i in range(N // 2, N)]
        if shuffle:
            random.shuffle(init_state)
        tensored = qt.tensor(init_state)
        circuit.initial_state = tensored #need to set half as |1> an half as |0>
    elif circuit_type == "zfsim":
        layers = fSim_circuit_layers(p, N, rotator='z')
        init_state = [qt.basis(2, 1) for i in range(N // 2)] + [qt.basis(2, 0) for i in range(N // 2, N)]
        if shuffle:
            random.shuffle(init_state)
        tensored = qt.tensor(init_state)
        circuit.initial_state = tensored 
    elif circuit_type == "fsim":
        if rotator in ['x', 'y', 'z']:
            layers = fSim_circuit_layers(p, N, rotator=rotator)
        else:
            layers = fSim_circuit_layers(p, N)
        init_state = [qt.basis(2, 1) for i in range(N // 2)] + [qt.basis(2, 0) for i in range(N // 2, N)]
        if shuffle:
            random.shuffle(init_state)
        tensored = qt.tensor(init_state)
        circuit.initial_state = tensored #need to set half as |1> an half as |0>
    elif circuit_type == "fixed_fsim":
        layers = fSim_circuit_layers(p, N, rotator='z', fixed=True)
        init_state = [qt.basis(2, 1) for i in range(N // 2)] + [qt.basis(2, 0) for i in range(N // 2, N)]
        if shuffle:
            random.shuffle(init_state)
        tensored = qt.tensor(init_state)
        circuit.initial_state = tensored

    for l in layers:
        circuit.add_layer(l)

    return circuit