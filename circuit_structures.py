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
            qs_on = random.sample(qs, k=N//2)
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


def gen_TFIM_layers(p, N):
    layers = []
    init_layer = [pqc.H(i, N) for i in range(N)]
    layers.append(init_layer)
    for i in range(p):
        first_half = [pqc.RR_block(pqc.R_zz, N)]
        second_layer = [pqc.R_x(i, N) for i in range(N)]
        second_half = [pqc.shared_parameter(second_layer, N)]
        layer = first_half + second_half
        layers.append(layer)
    return layers


def TFIM_hamiltonian(N, g):
    H = 0
    for i in range(N):
        i_plus = (i + 1) % N
        H += genFockOp(qt.sigmaz(), i, N) * genFockOp(qt.sigmaz(), i_plus, N) + g * genFockOp(qt.sigmax(), i, N)
    H = -1 * H
    return H