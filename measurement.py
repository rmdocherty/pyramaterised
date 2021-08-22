#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 21:20:44 2021

@author: ronan
"""
import qutip as qp
import numpy as np
import scipy
import matplotlib.pyplot as plt


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

    def _gen_f_samples(self, sample_N):
        """
        Generate random psi_theta and psi_pi $sample_N times for given PQC, then calculate
        |<psi_theta | psi_phi>|^2 which is F (1st moment of frame potential).
        Returns:
            F_samples = List of floats
        """
        F_samples = []
        for i in range(sample_N):
            self._QC.gen_quantum_state(energy_out=False)
            state1 = self._QC._quantum_state #psi theta
            self._QC.gen_quantum_state(energy_out=False)
            state2 = self._QC._quantum_state #psi phi
            sqrt_F = state1.overlap(state2)
            F = sqrt_F * sqrt_F
            F_samples.append(np.real(F))
        return F_samples

    def _gen_histo(self, F_samples):
        """
        Generate the probability mass histogram (i.e sum(P_pqc) = 1) for the F_samples
        from the PQC.
        Returns:
            prob: List of floats, 0 < p < 1 that are probabilities of state pair with Fidelity F
            F: List of floats, 0 < f < 1 that are fidelity midpoints,
        """
        #bin no. = 75 from paper
        prob, edges = np.histogram(F_samples, bins=75, range=(0, 1))
        prob = prob / sum(prob) #normalise by sum of prob or length?
        #this F assumes bins go from 0 to 1. Calculate midpoints of bins from np.hist
        F = np.array([(edges[0] + edges[1]) / 2] + [(edges[i - 1] + edges[i]) / 2 for i in range(2, len(edges))])
        return prob, F

    def expressibility(self, sample_N, graphs=False):
        N = 2 ** self._QC._n_qubits
        F_samples = self._gen_f_samples(sample_N)
        prob, F = self._gen_histo(F_samples, graphs=True)
        expr = 0
        haar = (N - 1) * ((1 - F) ** (N - 2))
        P_haar = haar / sum(haar) #do i need to normalise this?
        for index, P_pqc in enumerate(prob): #f = 1 at last bin so this is div by 0. fix by doing bin middles instead
            log = np.log(P_pqc / P_haar[index]) if P_pqc > 0 else 0
            expr += P_pqc * log
        if graphs is True:
            plt.plot(F, P_haar, label="Haar", color="C0", alpha=0.7)
            plt.plot(F, prob, label="Quantum state", color="C1", alpha=0.7)
            plt.xlabel("Fidelity")
            plt.ylabel("Probability")
            plt.legend(fontsize=20)
        print(f"Expressibility is {expr}")
        return expr
