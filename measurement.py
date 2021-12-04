#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 21:20:44 2021

@author: ronan
"""
import qutip as qt
import numpy as np
import scipy
import matplotlib.pyplot as plt
from itertools import product, combinations
from helper_functions import pretty_subplot
import random



class Measurements():
    def __init__(self, QC):
        self._QC = QC
        self.minimize_function = QC.cost
    
    def set_minimise_function(self, function):
        self.minimize_function = function

    def _get_QFI(self, grad_list=[]):
        """
        Given the input QC and it's gradient state list, calculate the assoicated
        QFI matrix by finding F_i,j = Re{<d_i psi| d_j psi>} - <d_i psi|psi><psi|d_j psi>
        for each i,j in n_params.

        Returns:
            qfi_matrix : np.array
            A n_param * n_param matrix of the QFI matrix for the VQC.
        """
        n_params = len([i for i in self._QC._parameterised if i > -1]) #these should both probably be getter methods but still
        #print(f"Number of params is {n_params}")
        if grad_list == []:
            grad_state_list = self._QC.get_gradients()
        else:
            grad_state_list = grad_list

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
                qfi_matrix[p, q] = 4 * np.real(deriv_overlap - RHS) #factor of 4 as otherwise is fubini-study metric

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
        #print(QFI)
        eigvals, eigvecs = scipy.linalg.eigh(QFI)
        print(eigvals)
        nonzero_eigvals = eigvals[eigvals > cutoff_eigvals]
        eff_quant_dim = len(nonzero_eigvals)
        return eff_quant_dim

    def new_measure(self):
        QFI = self._get_QFI()
        eigvals, eigvecs = scipy.linalg.eigh(QFI)
        capped = [1 if v > 1 else v for v in eigvals]
        return sum(capped)

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
            F = np.abs(state1.overlap(state2))**2
            F_samples.append(F)
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
        F = np.array([(edges[i - 1] + edges[i]) / 2 for i in range(1, len(edges))])
        return prob, F

    def _expr(self, F_samples, N):
        P_pqc, F = self._gen_histo(F_samples)

        haar = (N - 1) * ((1 - F) ** (N - 2)) #from definition in expr paper
        P_haar = haar / sum(haar) #do i need to normalise this?

        expr = np.sum(scipy.special.rel_entr(P_pqc, P_haar))
        return expr

    def expressibility(self, sample_N, graphs=False):
        """
        Expressibility.

        Given a PQC circuit, calculate $sample_N state pairs with randomised
        parameters and their overlap integral. Use that to generate a Fidelity
        distribution and also generate the fidelity of the Haar state using the
        analytic expression. From both of those calculate the KL diveregence =
        relative entropy = Expressibility and return it.

        Parameters:
            sample_N: int
                Number of random state sample pairs to generate.
            graphs: bool, default = False
                Whether or not to plot a graph of PQC fidelity distribution vs
                Haar distribution.
        Returns:
            expr: float
                The D_KL divergence of the fidelity distribution of the PQC
                vs the distribution from the Haar expression.
        """
        N = 2 ** self._QC._n_qubits

        F_samples = self._gen_f_samples(sample_N)
        expr = self._expr(F_samples, N)

        # if graphs is True:
        #     plt.figure("Expressibility")
        #     plt.plot(F, P_haar, label="Haar", color="C0", alpha=0.7, marker="x")
        #     plt.plot(F, P_pqc, label="Quantum state", color="C1", alpha=0.7, marker=".")
        #     pretty_subplot(plt.gca(), "Fidelity", "Probability", "Fidelity vs probability", 20)
        #print(f"Expressibility is {expr}")
        return expr

    def _gen_entanglement_samples(self, sample_N):
        """Generate sample_N quantum states to be used for entanglement calculations"""
        samples = []
        for i in range(sample_N):
            self._QC.gen_quantum_state(energy_out=False)
            samples.append(self._QC._quantum_state)
        return samples

    def _single_Q(self, system, n):
        """Calcuate Q value for single |psi> using average qubit purity."""
        summand = 0
        for k in range(n):
            density_matrix = system.ptrace(k)
            density_matrix *= density_matrix
            summand += density_matrix.tr()
        Q = 2 * (1 - (1 / n) * summand)
        return Q

    def entanglement(self, sample_N, graphs=False):
        """
        Entanglement.

        Given a PQC circuit $sample_N states and calculate the entanglement using
        the partial trace of the system.

        Parameters:
            sample_N: int
                Number of random state sample pairs to generate.
            graphs: bool, default = False
                Whether or not to plot a graph of PQC fidelity distribution vs
                Haar distribution.
        Returns:
            ent: list of floats
                List of $sample_N entanglement Q values for the PQC.
        """
        n = self._QC._n_qubits
        samples = self._gen_entanglement_samples(sample_N)
        ent = [self._single_Q(s, n) for s in samples]

        if graphs is True:
            plt.figure("Entanglement")
            plt.hist(ent, bins="fd")
            pretty_subplot(plt.gca(), "Entanglement (Q)", "Count", "Entanglement (Q) histogram", 20)
        return ent

    def _gen_pauli_group(self):
        N = self._QC._n_qubits
        pauli_list = [qt.qeye(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]
        strings = product(pauli_list, repeat=N)
        P_n = [qt.tensor(list(s)) for s in strings]
        return P_n

    def entropy_of_magic(self, psi=None, P_n=[]): #do we need to do this over many states?
        if P_n == []:
            P_n = self._gen_pauli_group()
        if psi == None:
            psi = self._QC._quantum_state
        N = self._QC._n_qubits
        d = 2**N
        xi_p = []
        for P in P_n:
            xi_p.append((d**-1) * qt.expect(P, psi)**2)
        norm = np.linalg.norm(xi_p, ord=2)
        magic = -1 * np.log(d*norm**2) #should we use log10, ln or log
        if magic > np.log(d + 1) - np.log(2):
            raise Exception("Magic max exceeded!")
        return magic

    def theta_to_magic(self, theta, P_n=[]):
        if P_n == []:
            P_n = self._gen_pauli_group()
        QC = self._QC
        QC._quantum_state = QC.run(angles=theta)
        eom = self.entropy_of_magic(QC._quantum_state, P_n)
        return -1 * eom # -1 so we can minize easily

    def _gen_F_n_2(self):
        N = self._QC._n_qubits
        strings = list(product([0,1], repeat=N))
        dims = [2 for i in range(2**N)]
        states = [qt.state_number_qobj(dims, list(s)) for s in strings]
        #print(
        return states, strings

    def _get_coeffs(self, psi=None, F=[]):
        if F == []:
            F, S = self._gen_F_n_2()
        if psi is None:
            psi = self._QC._quantum_state
        coeffs = []
        print(F)
        for state in F:
            c_i = state.overlap(psi)
            coeffs.append(c_i)
        return coeffs

    def GKP_magic(self, psi=None, F=[]):
        if F == []:
            F, S = self._gen_F_n_2()
        N = self._QC._n_qubits
        if psi is None:
            psi = self._QC._quantum_state
        coeffs = []
        for state in F:
            c_i = psi.overlap(state)
            #print(c_i)
            coeffs.append(c_i)
        GKP = 0
        mag = len(F)
        #print(S)
        for i_count in range(mag):
            for j_count in range(mag):
                for k_count in range(mag):
                    i = S[i_count]
                    j = S[j_count]
                    k = S[k_count]
                    inner_prod = [i[n] * k[n] for n in range(len(i))]
                    binary_product = sum(inner_prod) % 2
                    k_plus_j = tuple([(k[m] + j[m]) % 2 for m in range(len(k))])
                    k_plus_j_index = S.index(k_plus_j)
                    summand = coeffs[k_count] * coeffs[k_plus_j_index] * ((-1)**binary_product)/(2**N)
                    GKP += summand
                GKP = np.abs(GKP)
        GKP = np.log2(GKP)
        return GKP

    def efficient_measurements(self, sample_N, expr=True, ent=True, eom=True):
        n = self._QC._n_qubits
        states = [self._QC.gen_quantum_state() for i in range(sample_N)]
        #need combinations to avoid (psi,psi) pairs and (psi, phi), (phi,psi) duplicates which mess up expr
        state_pairs = list(combinations(states, r=2))
        overlaps = []

        if expr:
            for psi, phi in state_pairs:
                F = np.abs(psi.overlap(phi))**2
                overlaps.append(F)
            expr = self._expr(overlaps, 2**n)
        else:
            expr = -1

        P_n = self._gen_pauli_group()
        magics = []
        q_vals = []

        if ent:
            for psi in states:
                Q = self._single_Q(psi, n)
                q_vals.append(Q)
            q, std = np.mean(q_vals), np.std(q_vals)
        else:
            q, std = -1, -1

        if eom:
            for psi in states:
                entropy_of_magic = self.entropy_of_magic(psi, P_n)
                magics.append(entropy_of_magic)
            magic_bar, magic_std = np.mean(magics), np.std(magics)
        else:
            magic_bar, magic_std = 0, 0
        return {"Expr": expr, "Ent": [q, std], "Magic": [magic_bar, magic_std]}

    def meyer_wallach(self, sample_N): 
        N = self._QC._n_qubits
        
        def iota(j, b): 
            iotabras = []
            iotakets = []
            stringstates = [list(i) for i in product([0, 1], repeat = N)]
            for state in stringstates: 
                if state[j] == b: 
                    iotabras.append(state)
                    newstate = state[:j] + state[j+1:]
                    iotakets.append(newstate)
            projector = sum(qt.qip.qubits.qubit_states(N=N-1, states = iotakets[i])*qt.qip.qubits.qubit_states(N=N, states = iotabras[i]).dag() for i in range(len(iotakets)))
            return projector
        
        def Distance(state1, state2): 
            distance = 0.5*sum(sum((state1[i][0]*state2[j][0] - state1[j][0]*state2[i][0])*np.conj(state1[i][0]*state2[j][0] - state1[j][0]*state2[i][0]) for i in range(2**(N-1))) for j in range(2**(N-1)))
            return distance    
        
        def Q(state): 
            Q = (4/N)*sum(Distance(iota(i,0)*state,iota(i,1)*state) for i in range(N))
            return Q

        
        entanglements = []
        samples = self._gen_entanglement_samples(sample_N)
        for system in samples: 
            entanglements.append(Q(system))
        mwexpr = np.mean(entanglements)
        mwstd = np.std(entanglements)
        print(f"Meyer-Wallach entanglement: {mwexpr} +/- {mwstd}")
        return mwexpr, mwstd 

    def get_gradient_vector(self, theta):
        psi = self._QC._quantum_state
        self.gradient_list = self._QC.get_gradients()
        gradients = []
        for i in self.gradient_list:
            deriv = i
            H_di_psi = self._QC.H * deriv
            d_i_f_theta = 2 * np.real(psi.overlap(H_di_psi))
            gradients.append(d_i_f_theta)
        return gradients

    def train(self, epsilon=1e-6, rate=0.001, method="gradient", angles=[], trajectory=False, magic=False):
        quit_iterations = 100000
        count = 0
        diff = 1
        traj = []
        magics = []

        P_n = self._gen_pauli_group()

        def trajmaj(Xi):
            eom = self.entropy_of_magic(psi=self._QC._quantum_state, P_n=P_n)
            magics.append(eom)
            trajectory = self.minimize_function(Xi)
            traj.append(trajectory)

        self._QC._quantum_state = self._QC.run(angles=angles)
        trajmaj(angles)

        if method.lower() in ["gradient", "qng"]:
            self._QC._quantum_state = self._QC.run(angles=angles)
            prev_energy = self.minimize_function(angles)
            while diff > epsilon and count < quit_iterations:

                if magic is True:
                    eom = self.entropy_of_magic(psi=self._QC._quantum_state, P_n=P_n)
                    magics.append(eom)

                theta = self._QC.get_params()
                gradients = self.get_gradient_vector(theta)

                if method == "gradient":
                    theta_update = list(np.array(theta) - rate * np.array(gradients))
                elif method == "QNG": #some serious problems here, think we need renormalizaiton
                    QFI = self._get_QFI(grad_list=self.gradient_list)
                    inverse = np.linalg.pinv(QFI)
                    f_inv_grad_psi = inverse.dot(np.array(gradients))
                    theta_update = list(np.array(theta) - rate * f_inv_grad_psi)

                if count % 100 == 0:
                    print(f"On iteration {count}, energy = {prev_energy}, diff is {diff}")
                    #print(theta_update)

                energy = self.minimize_function(theta_update)
                diff = np.abs(energy - prev_energy)
                count += 1
                prev_energy = energy
                traj.append(energy)
            print(f"Finished after {count} iterations with cost function = {energy}")
        else:
            if self.minimize_function == self.theta_to_magic:
                op_out = scipy.optimize.minimize(self.minimize_function, x0=angles, 
                                                method=method, args=[P_n], callback=trajmaj, tol=epsilon)
            else:
                op_out = scipy.optimize.minimize(self.minimize_function, x0=angles, 
                                                method=method, callback=trajmaj, tol=epsilon, jac=self.get_gradient_vector)
            energy = op_out.fun
        return [energy, traj, magics]
