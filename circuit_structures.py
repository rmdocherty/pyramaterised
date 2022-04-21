#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 12:19:54 2021

@author: ronan
"""
import PQC_lib as pqc
import qutip as qt
import numpy as np
import random
from measurement import Measurements
from helper_functions import genFockOp



# ============================== HE CIRCUITS ==============================
LEN_CLIFF_STRING = 3


def gen_clifford_circuit(p, N, method='random'):
    clifford_gates = [pqc.H, pqc.S, pqc.CZ, pqc.CNOT]
    layers = []
    qs = list(range(N))

    if method.lower() == 'random':
        for i in range(p):
            layer = []
            for n in range(N):
                gate = random.choice(clifford_gates)
                if issubclass(gate, pqc.PRot): #can't check is_param of this as not instantised yet - could make class variable?
                    q_on = random.randint(0, N - 1)
                    layer.append(gate(q_on, N))
                elif issubclass(gate, pqc.EntGate): #entangling gate
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
            layer.append(pqc.CHAIN(pqc.CNOT, N)) #add entangling layer
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
    #started with fixed block of N R_y and N R_x as first layer
    initial_layer = [pqc.R_y(i, N) for i in range(N)] + [pqc.R_z(i, N) for i in range(N)]
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
            rotation = pqc.fixed_R_y(q_on, N, np.pi / 2) #NB these fixed gates aren't parametrised and shouldn't be counted in angles
            fixed_rots.append(rotation)
            U_ent = pqc.CPHASE([q_on, ((q_on + 1) + 2 * a_l) % N], N)
            cphases.append(U_ent)
        p_layer = fixed_rots + cphases #need fixed r_y to come before c_phase

        #rotation layer - R_y then R_z on each kth qubit
        for k in range(1, N // 2 + 1):
            q_on = 2 * k - 2
            p_layer = p_layer + [pqc.R_y(q_on, N), pqc.R_z(q_on, N)]
            #R_y gates have theta_i = pi/2 for theta_r
            angles.append(np.pi / 2)
            #R_z gates have theta_i = 0 for theta_r
            angles.append(0)
        layers.append(p_layer)
    return layers, angles

def find_overparam_point(circuit, layer_index_list, epsilon=1e-3):
    layers_to_add = [circuit.get_layer(i) for i in layer_index_list]
    prev_rank, rank_diff = 0, 1
    count = 0
    while rank_diff > epsilon and count < 1e6:
        for l in layers_to_add:
            circuit.add_layer(l)
        circuit.gen_quantum_state()
        circuit_m = Measurements(circuit)
        QFI = circuit_m._get_QFI()
        rank = np.linalg.matrix_rank(QFI)
        rank_diff = np.abs(rank - prev_rank)
        print(f"Iteration {count}, r0={prev_rank}, r1={rank}, delta = {rank_diff}")
        prev_rank = rank
        count += 1
    return count


def string_to_entangler(string):
    lower = string.lower()
    if lower == "cnot":
        entangler = pqc.CNOT
    elif lower == "cphase":
        entangler = pqc.CPHASE
    elif lower == "cz":
        entangler = pqc.CZ
    elif lower == "sqrtiswap":
        entangler = pqc.sqrtiSWAP
    else:
        raise Exception("Must supply a valid entangler!")
    return entangler


def circuit_1(p, N):
    layer = [pqc.R_x(i, N) for i in range(N)] + [pqc.R_y(i, N) for i in range(N)]
    layers = []
    for i in range(p):
        layers.append(layer)
    return layers


def circuit_2(p, N, ent_str="cnot"):
    entangler = string_to_entangler(ent_str)
    layer = [pqc.R_x(i, N) for i in range(N)] + [pqc.R_z(i, N) for i in range(N)] + [pqc.CHAIN(entangler, N)]
    layers = []
    for i in range(p):
        layers.append(layer)
    return layers


def circuit_9(p, N, ent_str="cphase"):
    entangler = string_to_entangler(ent_str)
    layer = [pqc.H(i, N) for i in range(N)] + [pqc.CHAIN(entangler, N)] + [pqc.R_x(i, N) for i in range(N)]
    layers = []
    for i in range(p):
        layers.append(layer)
    return layers


def qg_circuit(p, N, ent_str="cnot"):
    entangler = string_to_entangler(ent_str)
    init_layer = [pqc.fixed_R_y(i, N, np.pi / 4) for i in range(N)]
    layer1 = [pqc.R_z(i, N) for i in range(N)] + [pqc.CHAIN(entangler, N)]
    layer2 = [pqc.R_x(i, N) for i in range(N)] + [pqc.CHAIN(entangler, N)]
    layer3 = [pqc.R_z(i, N) for i in range(N)] + [pqc.CHAIN(entangler, N)]
    layer = layer1 + layer2 + layer3
    layers = [init_layer]
    for i in range(p):
        layers.append(layer)
    return layers


def generic_HE(p, N, ent_str="cnot"):
    entangler = string_to_entangler(ent_str)
    init_layer = [pqc.fixed_R_y(i, N, np.pi / 4) for i in range(N)]
    layer = [pqc.R_y(i, N) for i in range(N)] + [pqc.R_z(i, N) for i in range(N)] + [pqc.CHAIN(entangler, N)]
    layers = [init_layer] 
    for i in range(p):
        layers.append(layer)
    return layers

def clifford_HE(p, N, ent_str="cnot"):
    entangler = string_to_entangler(ent_str)
    layer = [pqc.R_y(i, N) for i in range(N)] + [pqc.R_z(i, N) for i in range(N)] + [pqc.CHAIN(entangler, N)]
    layers = [] 
    for i in range(p):
        layers.append(layer)
    return layers

def y_CPHASE(p, N):
    layers = []
    layer = [pqc.R_y(i, N) for i in range(N)] + [pqc.CHAIN(pqc.CPHASE, N)]
    for i in range(p):
        layers.append(layer)
    return layers

def double_y_CPHASE(p, N):
    layers = []
    layer = [pqc.R_y(i, N) for i in range(N)] + [pqc.R_y(i, N) for i in range(N)] + [pqc.CHAIN(pqc.CPHASE, N)]
    for i in range(p):
        layers.append(layer)
    return layers

# ============================== PROBLEM INPSIRED CIRCUITS ==============================


def gen_TFIM_layers(p, N):
    initial_layer = [pqc.H(i, N) for i in range(N)]
    layers = [initial_layer]
    for i in range(p):
        first_half = [pqc.RR_block(pqc.R_zz, N)]
        second_layer = [pqc.R_x(i, N) for i in range(N)]
        second_half = [pqc.shared_parameter(second_layer, N)]
        layer = first_half + second_half
        layers.append(layer)
    return layers


def gen_modified_TFIM_layers(p, N):
    layers = []
    for i in range(p):
        first_half = [pqc.RR_block(pqc.R_zz, N)]
        second_layer = [pqc.R_x(i, N) for i in range(N)]
        second_half = [pqc.shared_parameter(second_layer, N)]
        third_layer = [pqc.R_z(i, N) for i in range(N)]
        third_half = [pqc.shared_parameter(third_layer, N)]
        layer = first_half + second_half + third_half
        layers.append(layer)
    return layers


def TFIM_hamiltonian(N, g, h=0):
    H = 0
    for i in range(N):
        i_plus = (i + 1) % N
        H += genFockOp(qt.sigmaz(), i, N) * genFockOp(qt.sigmaz(), i_plus, N) + g * genFockOp(qt.sigmax(), i, N) + h * genFockOp(qt.sigmaz(), i, N)
    H = -1 * H
    return H


def gen_XXZ_layers(p, N, commute=False):
    even_indices = []
    odd_indices = []
    for i in range(1, (N // 2) + 1): # -1 to convert from 1 indexed paper defn to 0 indexed qubit lists
        even_bot, even_top = (2 * i - 1) - 1, (2 * i) - 1
        even_indices.append((even_bot, even_top))
        odd_bot, odd_top = (2 * i) - 1, ((2 * i + 1) - 1) % N #need mod N here
        odd_indices.append((odd_bot, odd_top))

    init_x_bits = [pqc.X(i, N) for i in range(N)]
    init_H = [pqc.H(i, N) for i in range(N) if i % 2 == 0] #H on even links
    init_CNOT = [pqc.CNOT([i, j], N) for i, j in even_indices]
    #layers = [init_x_bits, init_H, init_CNOT]
    layers = []
    for l in range(p):
        layer = []
        ZZ_1 = [pqc.R_zz((i, j), N) for i, j in odd_indices]
        YY_XX_1 = [pqc.R_yy((i, j), N) for i, j in odd_indices] + [pqc.R_xx((i, j), N) for i, j in odd_indices]
        ZZ_2 = [pqc.R_zz((i, j), N) for i, j in even_indices]
        YY_XX_2 = [pqc.R_yy((i, j), N) for i, j in even_indices] + [pqc.R_xx((i, j), N) for i, j in even_indices]
        theta = [pqc.shared_parameter(ZZ_1, N)]
        phi = [pqc.shared_parameter(YY_XX_1, N, commute=commute)]
        beta = [pqc.shared_parameter(ZZ_2, N)]
        gamma = [pqc.shared_parameter(YY_XX_2, N, commute=commute)]
        layer = theta + phi + beta + gamma
        layers.append(layer)
    return layers


def gen_theta_block(q1, q2, N):
    block = []
    block.append(pqc.sqrtiSWAP([q1, q2], N))

    minus_rot = pqc.negative_R_z(q1, N)
    offset_rot = pqc.offset_R_z(q2, N, np.pi)
    shared_param = pqc.shared_parameter([minus_rot, offset_rot], N)
    block.append(shared_param)

    block.append(pqc.sqrtiSWAP([q1, q2], N))

    fixed = pqc.fixed_R_z(q2, N, np.pi)
    block.append(fixed)
    return block


def list_to_pairs(x):
    pairs = [(x[i], x[i + 1]) for i in range(0, len(x) - 1, 2)]
    return pairs


def gen_fermionic_circuit(p, N):
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


def gen_fSim_circuit(p, N, rotator='y', fixed=False):
    r = rotator.lower()
    if r == 'y':
        rot_gate = pqc.R_y
    elif r == 'x':
        rot_gate = pqc.R_x
    elif r == 'z':
        print("using z gate as rotator")
        rot_gate = pqc.R_z
    else:
        raise Exception("Please supply a valid single qubit rotator")
    layers = []
    for l in range(p):
        layer = [rot_gate(i, N) for i in range(N)]
        if N % 2 == 0:
            offset = l % 2
            for i in range(offset, N, 2):
                if fixed is True:
                    gate = pqc.fixed_fSim([i, (i+1)%N], N)
                else:
                    gate = pqc.fSim([i, (i+1)%N], N)
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
                    gate = pqc.fixed_fSim([pair[0], pair[1]], N)
                else:
                    gate = pqc.fSim([pair[0], pair[1]], N)
                layer.append(gate)
            layer.append(rot_gate(offset, N))
        layers.append(layer)
    return layers
