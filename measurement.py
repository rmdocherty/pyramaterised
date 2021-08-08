#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 21:20:44 2021

@author: ronan
"""
import qutip as qp
import numpy as np
import scipy


class Measurements():
    def __init__(self, QC):
        self._QC = QC

    def _get_QFI(self):
        """
        Given the input QC and it's gradient state list, calculate the assoicated
        QFI matrix by finding F_i,j = Re{<d_i psi| d_j psi>} - <d_i psi|psi><psi|d_j psi>
        for each i,j in n_params.

        Returns:
            qfi_matrix : np.array
            A n_param * n_param matrix of the QFI matrix for the VQC.
        """
        n_params = self._QC._n_params #these should both probably be getter methods but still
        grad_state_list = self._QC._gradient_state_list

        #get all single elements first
        single_qfi_elements = np.zeros(n_params, dtype=np.complex128)
        for param in range(n_params):
            overlap = self._QC._quantum_state.overlap(grad_state_list[param])
            single_qfi_elements[param] = overlap

        qfi_matrix = np.zeros([n_params, n_params])
        for p in range(n_params):
            for q in range(p, n_params):
                deriv_overlap = grad_state_list[p].overlap(grad_state_list[q])
                #single_qfi_elements[i] is <d_i psi | psi>
                RHS = np.conjugate(single_qfi_elements[p]) * single_qfi_elements[q]
                #assign p, qth elem of QFI, c.f eq (B3) in NIST review
                qfi_matrix[p, q] = np.real(deriv_overlap - RHS)

        for p in range(n_params): #use fact QFI mat. real, hermitian and therefore symmetric
            for q in range(p + 1, n_params):
                qfi_matrix[q, p] = qfi_matrix[p, q]
        return qfi_matrix

    def get_effective_quantum_dimension(self, cutoff_eigvals):
        """
        Get EFD by counting the # of non-zero eigenvalues of the QFI matrix.
        Returns:
            eff_quant_dim = Int
        """
        QFI = self._get_QFI()
        eigvals, eigvecs = scipy.linalg.eigh(QFI)
        nonzero_eigvals = eigvals[eigvals > cutoff_eigvals]
        eff_quant_dim = len(nonzero_eigvals)
        return eff_quant_dim

    def expressibility(self, sample_N):
        F_samples = []
        N = 2 ** self._QC._n_qubits
        for i in range(sample_N):
            #print(i)
            self._QC.gen_quantum_state()
            state1 = self._QC._quantum_state #psi theta
            self._QC.gen_quantum_state()
            state2 = self._QC._quantum_state #psi phi
            sqrt_F = state1.overlap(state2)
            F = np.conjugate(sqrt_F) * sqrt_F
            #print(type(F))
            F_samples.append(np.real(F))
        prob, F = np.histogram(F_samples, bins="fd")
        print(prob)
        expr = 0
        for index, P_pqc in enumerate(prob):
            #print(index)
            P_haar = (N - 1) * (1 - F[index]) ** (N - 2)
            expr += P_pqc * np.log(P_pqc / P_haar)
        print(f"Expressibility is {expr}")
        return expr

