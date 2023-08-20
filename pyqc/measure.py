#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 21:20:44 2021

@author: ronan
"""

import qutip as qt
import numpy as np
import scipy
import random
from itertools import combinations
from typing import Callable, Tuple, Literal, Dict, Union
from .circuit import PQC, Gradient, QubitNumber
from .gates import Angle


class Measurements:
    def __init__(self, QC: PQC):
        self.QC: PQC = QC
        self.minimize_function: Callable
        try:
            self.minimize_function = QC.cost
        except AttributeError:
            # identity function lambda
            self.minimize_function = lambda x: x

    def set_minimise_function(self, function: Callable) -> None:
        """Change the minimize function used in self.train"""
        self.minimize_function = function

    def get_QFI(self, grad_list: list[Gradient] = []) -> np.ndarray:
        """Given the input QC and it's gradient state list, calculate the assoicated
        QFI matrix by finding F_i,j = Re{<d_i psi| d_j psi>} - <d_i psi|psi><psi|d_j psi>
        for each i,j in n_params.

        Returns:
            qfi_matrix : np.array
            A n_param * n_param matrix of the QFI matrix for the VQC.
        """
        n_params: int = len([i for i in self.QC.parameterised if i > -1])
        grad_state_list: list[Gradient]
        if grad_list == []:
            grad_state_list = self.QC.get_gradients()
        else:
            grad_state_list = grad_list

        # get all single elements first
        single_qfi_elements: np.ndarray = np.zeros(n_params, dtype=np.complex128)
        for param in range(n_params):
            overlap: float = self.QC.state.overlap(grad_state_list[param])
            single_qfi_elements[param] = overlap

        qfi_matrix: np.ndarray = np.zeros([n_params, n_params])
        for p in range(n_params):
            for q in range(p, n_params):
                deriv_overlap: float = grad_state_list[p].overlap(grad_state_list[q])
                # single_qfi_elements[i] is <d_i psi | psi>
                RHS = np.conjugate(single_qfi_elements[p]) * single_qfi_elements[q]
                # assign p, qth elem of QFI, c.f eq (B3) in NIST review
                qfi_matrix[p, q] = 4 * np.real(
                    deriv_overlap - RHS
                )  # factor of 4 as otherwise is fubini-study metric

        for p in range(
            n_params
        ):  # use fact QFI mat. real, hermitian and therefore symmetric
            for q in range(p + 1, n_params):
                qfi_matrix[q, p] = qfi_matrix[p, q]
        return qfi_matrix

    def get_eigenvalues(self, QFI: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        eigvals, eigvecs = scipy.linalg.eigh(QFI)
        return eigvals, eigvecs

    def get_effective_quantum_dimension(self, cutoff_eigvals: float) -> int:
        """
        Get EFD by counting the # of non-zero eigenvalues of the QFI matrix.
        Returns:
            eff_quant_dim = int
        """
        QFI: np.ndarray = self.get_QFI()
        eigvals, eigvecs = scipy.linalg.eigh(QFI)
        nonzero_eigvals: np.ndarray = eigvals[eigvals > cutoff_eigvals]
        eff_quant_dim: int = len(nonzero_eigvals)
        return eff_quant_dim

    def new_measure(self, QFI: np.ndarray = None) -> int:
        """Sum of the eigenvalues of QFI where e-vals > 1 are capped at 1. This is
        to stop large eigenvalues dominating for certain circuits.
        Returns:
            sum(capped): int
        """
        if QFI is None:
            QFI = self.get_QFI()
        eigvals, eigvecs = self.get_eigenvalues(QFI)
        capped = [1 if v > 1 else v for v in eigvals]
        return sum(capped)

    def find_overparam_point(self, layer_index_list, epsilon: float = 1e-3) -> int:
        """Find the overparameterisation point of a circuit, which is the number
        of layers need to saturate the rank of the QFIM.
        Returns:
            count: number of layers needed.
        """
        layers_to_add = [self.QC.get_layer(i) for i in layer_index_list]
        prev_rank: int = 0
        rank_diff: int = 1
        count: int = 0
        while rank_diff > epsilon and count < 1e6:
            for l in layers_to_add:
                self.QC.add_layer(l)
            self.QC.update_state("random")
            QFI = self.get_QFI()
            rank: int = np.linalg.matrix_rank(QFI)
            rank_diff = np.abs(rank - prev_rank)
            print(f"Iteration {count}, r0={prev_rank}, r1={rank}, delta = {rank_diff}")
            prev_rank = rank
            count += 1
        return count

    def _gen_f_samples(self, sample_N: int) -> list[float]:
        """
        Generate random psi_theta $sample_N$ times for given PQC, then get
        make all possible combinations of states to form psi, phi pairs to
        calculate |<psi_theta | psi_phi>|^2 which is F (1st moment of frame potential).
        Returns:
            F_samples = List of floats
        """
        F_samples: list[float] = []
        states: list[qt.Qobj] = [self.QC.run("random") for i in range(sample_N)]
        state_pairs = list(combinations(states, r=2))
        for psi, phi in state_pairs:
            F: float = np.abs(psi.overlap(phi)) ** 2
            F_samples.append(F)
        return F_samples

    def _gen_histo(
        self, F_samples: list[float], filt: float = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate the probability mass histogram (i.e sum(P_pqc) = 1) for the F_samples
        from the PQC.
        Returns:
            prob: List of floats, 0 < p < 1 that are probabilities of state pair with Fidelity F
            F: List of floats, 0 < f < 1 that are fidelity midpoints,
        """
        F_sample_arr: np.ndarray = np.array(F_samples)
        if filt > 0:
            F_sample_arr = F_sample_arr[F_sample_arr < filt]
        # bin no. = 75 from paper
        prob, edges = np.histogram(
            F_samples, bins=int((75 / 10000) * len(F_samples)), range=(0, 1)
        )  # , range=(0, 1) #used to be 1, could be np.amax(F_samples)
        prob = prob / sum(prob)  # normalise by sum of prob or length?
        # this F assumes bins go from 0 to 1. Calculate midpoints of bins from np.hist
        F = np.array([(edges[i - 1] + edges[i]) / 2 for i in range(1, len(edges))])
        return prob, F

    def expr(self, F_samples: list[float], N, filt: int = 0) -> float:
        """Same as expressibility except must supply F_samples and can vary the
        N of the Hilbert space from which the kl_diveregence is calculated from.
        This allows the computation of the effective Hilbert space later.
        Returns:
            expr: float represeneting the kl_divergence between the histogrammed
                  f_sample distribution and N qubit Haar distribution.
        """
        if len(F_samples) == 0:
            return 0
        P_pqc, F = self._gen_histo(F_samples, filt)

        haar: np.ndarray = (N - 1) * (
            (1 - F) ** (N - 2)
        )  # from definition in expr paper
        P_haar: np.ndarray = haar / sum(haar)
        expr: float = np.sum(
            scipy.special.kl_div(P_pqc, P_haar)
        )  # expr = np.sum(scipy.special.rel_entr(P_pqc, P_haar))
        return expr

    def expressibility(self, sample_N: int) -> float:
        """Given a PQC circuit, calculate $sample_N$ state pairs with randomised
        parameters and their overlap integral. Use that to generate a Fidelity
        distribution and also generate the fidelity of the Haar state using the
        analytic expression. From both of those calculate the KL diveregence =
        relative entropy = Expressibility and return it.

        Returns:
            expr: float
                The D_KL divergence of the fidelity distribution of the PQC
                vs the distribution from the Haar expression.
        """
        N: int = 2**self.QC.n_qubits
        F_samples = self._gen_f_samples(sample_N)
        expr = self.expr(F_samples, N)
        return expr

    def find_eff_H(self, circuit_f_samples: list[float], n: QubitNumber) -> float:
        """Find the effective Hilbert Space by minimizing the dimension of the Hilbert
        space that the expressibility of the measured F_samples is calaculated with.
        This tells us which subspace the circuit acts in given its measured F_samples.
        Returns:
            out.x[0]: float - result of the minimization procedure.
        """

        def wrapper(n, F_samples=[]):
            return self.expr(F_samples, n, filt=0.2)

        def log_wrapper(n, F_samples=[]):
            return self.log_expr(F_samples, n)

        if n > 10:  # change this later
            wrap_fn = log_wrapper
        else:
            wrap_fn = wrapper
        out = scipy.optimize.minimize(
            wrap_fn, [4], args=(circuit_f_samples), method="BFGS"
        )  # Nelder-Mead
        if out.success is True:
            return out.x[0]
        else:
            print(out)
            return 0

    def single_Q(self, system: qt.Qobj, n: QubitNumber) -> float:
        """Calcuate Q value for single |psi> using average qubit purity.
        Returns
            Q: float - entanglement of state, varies between 0 and 1 (maximially entangled)
        """
        summand: float = 0
        for k in range(n):
            density_matrix: qt.Qobj = system.ptrace(k)
            density_matrix *= density_matrix
            summand += density_matrix.tr()
        Q = 2 * (1 - (1 / n) * summand)
        return Q

    def entanglement(self, sample_N: int) -> list[float]:
        """Given a PQC circuit $sample_N states and calculate the entanglement using
        the partial trace of the system.
        Returns:
            ent: list of floats
                List of $sample_N$ entanglement Q values for the PQC.
        """
        n: QubitNumber = self.QC.n_qubits
        samples: list[qt.Qobj] = [self.QC.run("random") for i in range(sample_N)]
        ent: list[float] = [self.single_Q(s, n) for s in samples]
        return ent

    def theta_to_magic(self, angles: list[Angle]) -> float:
        """Given a set of angles, calculate circuit state then measure the Reyni entropy of
        magic for the circuit. Multiplies by -1 s.t minimising this function maximises magic.
        Returns:
            -1 * eom: float - -1 * the magic.
        """
        state: qt.Qobj = self.QC.run(angles=angles)
        eom: float = self.renyi_entropy_fast(state)
        return -1 * eom  # -1 so we can minimize easily

    def theta_to_gkp(self, angles: list[Angle]):
        """Same as theta to magic but for GKP instead."""
        state: qt.Qobj = self.QC.run(angles=angles)
        gkp: float = self.gkp_fast(state)
        return -1 * gkp

    # The following fast reyni code is courtesy of txhaug
    def numberToBase(self, n: float, b: float, n_qubits: QubitNumber) -> np.ndarray:
        if n == 0:
            return np.zeros(n_qubits, dtype=int)
        digits: np.ndarray = np.zeros(n_qubits, dtype=int)
        counter: int = 0
        while n:
            digits[counter] = int(n % b)
            n //= b
            counter += 1
        return digits[::-1]

    def get_conversion_matrix_mod_add_index(
        self, base_states: list[np.ndarray]
    ) -> np.ndarray:
        n_qubits: QubitNumber = len(base_states[0])
        mag: int = len(base_states)
        to_index: np.ndarray = 2 ** np.arange(n_qubits)[::-1]
        conversion_matrix: np.ndarray = np.zeros([mag, mag], dtype=int)
        for j_count in range(mag):
            base_j = base_states[j_count]
            k_plus_j = np.mod(base_states + base_j, 2)
            k_plus_j_index = np.sum(k_plus_j * to_index, axis=1)
            conversion_matrix[j_count, :] = k_plus_j_index
        return conversion_matrix

    def get_conversion_matrix_binary_prod(
        self, base_states: list[np.ndarray]
    ) -> np.ndarray:
        mag: int = len(base_states)
        conversion_matrix: np.ndarray = np.zeros([mag, mag], dtype=int)
        for i_count in range(mag):
            base_i = base_states[i_count]
            binary_product: float = np.mod(np.dot(base_states, base_i), 2)
            conversion_matrix[i_count, :] = (-1) ** binary_product
        return conversion_matrix

    def get_conversion_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        n = self.QC.n_qubits
        base_states = [self.numberToBase(i, 2, n) for i in range(2**n)]
        conversion_matrix_binary_prod = self.get_conversion_matrix_binary_prod(
            base_states
        )
        conversion_matrix_mod_add_index = self.get_conversion_matrix_mod_add_index(
            base_states
        )
        return (conversion_matrix_mod_add_index, conversion_matrix_binary_prod)

    def set_converstion_matrices(self, conv_mats: np.ndarray) -> None:
        self.conversion_matrices = conv_mats

    def renyi_entropy_fast(
        self,
        state: qt.Qobj,
        conversion_matrices: Tuple[np.ndarray, np.ndarray] = None,
        alpha: float = 2,
    ) -> float:
        if conversion_matrices is None:
            (
                conversion_matrix_mod_add_index,
                conversion_matrix_binary_prod,
            ) = self.get_conversion_matrices()
        else:
            (
                conversion_matrix_mod_add_index,
                conversion_matrix_binary_prod,
            ) = conversion_matrices

        coeffs = state.data.toarray()[:, 0]
        n_qubits: QubitNumber = len(state.dims[0])

        renyi_fast: float = np.sum(
            np.abs(
                2 ** (-n_qubits / 2)
                * np.dot(
                    np.conjugate(coeffs) * conversion_matrix_binary_prod,
                    coeffs[conversion_matrix_mod_add_index],
                )
            )
            ** (2 * alpha)
        )
        renyi_fast = 1 / (1 - alpha) * np.log(renyi_fast) - np.log(2**n_qubits)
        return renyi_fast

    def entropy_of_magic(self, sample_N: int):
        """Calculate the Reyni entropy of magic of $sample_N$ states from the circuit.
        Returns:
            np.mean(magics): average magic of the state samples.
        """
        states: list[qt.Qobj] = [self.QC.run("random") for i in range(sample_N)]
        conv_mats = self.get_conversion_matrices()
        magics: list[float] = [self.renyi_entropy_fast(i, conv_mats) for i in states]
        return np.mean(magics)

    def gkp_fast(
        self, state: qt.Qobj, conversion_matrices: Tuple[np.ndarray, np.ndarray] = None
    ) -> float:
        return (
            1
            / (2 * np.log(2))
            * self.renyi_entropy_fast(state, conversion_matrices, alpha=1 / 2)
        )

    def efficient_measurements(
        self,
        sample_N: int,
        measure_expr: bool = True,
        measure_ent: bool = True,
        measure_eom: bool = True,
        measure_GKP: bool = True,
        full_data: bool = False,
        angles: Literal["random", "clifford"] = "random",
    ) -> Dict:
        """Generate lots of states and reuse them for a variety of measurements. Expressibility
        limited by qubit number as they are too memory intensize/don't work at high qubit number.
        Returns:
            dict of all the data recorded. If the measure hasn't been recorded, it has a value of -1.
        """
        n: QubitNumber = self.QC.n_qubits

        if sample_N == 0:
            measure_expr = False
            measure_ent = False
            measure_eom = False
            measure_GKP = False

        if angles == "clifford":
            clifford_angles: Tuple[Angle, Angle, Angle, Angle, Angle] = (
                0,
                np.pi / 2,
                np.pi,
                3 * np.pi / 2,
                2 * np.pi,
            )
            init_angles: list[list[Angle]] = [
                [random.choice(clifford_angles) for i in range(self.QC.n_params)]
                for i in range(sample_N)
            ]
            states: list[qt.Qobj] = [self.QC.run(angles=init) for init in init_angles]
        else:
            states = [self.QC.run(angles="random") for i in range(sample_N)]
        # need combinations to avoid (psi,psi) pairs and (psi, phi), (phi,psi) duplicates which mess up expr
        state_pairs = list(combinations(states, r=2))
        overlaps: list[float] = []
        magics: list[float] = []
        gkps: list[float] = []
        q_vals: list[float] = []
        conv_mats = self.get_conversion_matrices()

        if measure_expr and n < 12:
            for psi, phi in state_pairs:
                F = np.abs(psi.overlap(phi)) ** 2
                overlaps.append(F)
            if n < 7:  # if we're rnning to large N we only want the overlaps
                expr = self.expr(overlaps, 2**n)
            else:
                expr = -1
        else:
            expr = -1

        if measure_ent:
            for psi in states:
                Q = self.single_Q(psi, n)
                q_vals.append(Q)
            q, std = np.mean(q_vals), np.std(q_vals)
        else:
            q, std = -1, -1

        if measure_eom:
            for psi in states:
                entropy_of_magic = self.renyi_entropy_fast(psi, conv_mats)
                magics.append(entropy_of_magic)
            magic_bar, magic_std = np.mean(magics), np.std(magics)
        else:
            magic_bar, magic_std = -1, -1

        if measure_GKP:
            for psi in states:
                gkp = self.gkp_fast(psi, conv_mats)
                gkps.append(gkp)
            gkp_bar, gkp_std = np.mean(gkps), np.std(gkps)
        else:
            gkp_bar, gkp_std = -1, -1

        if full_data is True:
            return {"Expr": overlaps, "Ent": q_vals, "Magic": magics, "GKP": gkps}
        else:
            return {
                "Expr": expr,
                "Ent": [q, std],
                "Magic": [magic_bar, magic_std],
                "GKP": [gkp_bar, gkp_std],
            }

    def get_gradient_vector(self, theta: list[Angle]) -> list[Gradient]:
        self.QC.state = self.QC.run(angles=theta)
        psi = self.QC.state
        self.gradient_list = self.QC.get_gradients()
        gradients: list[Gradient] = []
        for i in self.gradient_list:
            deriv: Gradient = i
            H_di_psi: qt.Qobj = self.QC.H * deriv
            d_i_f_theta: Gradient = 2 * np.real(psi.overlap(H_di_psi))
            gradients.append(d_i_f_theta)
        return gradients

    def train(
        self,
        epsilon: float = 1e-6,
        rate: float = 0.001,
        method: str = "gradient",
        angles: Union[list[Angle], Literal["random"]] = "random",
        verbose: bool = False,
    ) -> Tuple[float, list[float], list[float], list[float], list[float]]:
        """Optimise parameters of the PQC to minimise the current minimize function (usually
        a Hamiltonian). Optimisation can be done using gradient descent or any of scipy's
        inbuilt optimization routienes (NB these are always much better than gradient-based,
        but as they converge so quickly they may be less useful for studying capacity measures during
        training).
        Returns:
        (energy, traj, magics, ents, gkps): tuple of various quantities. energy is the final value of
        the cost function and trajectory is the values of the cost function at each iteration.
        magics, ents, gkps are values of those capacity measures at each iteration.
        """
        quit_iterations: int = 100000
        count: int = 0
        diff: float = 1
        traj: list[float] = []
        magics: list[float] = []
        gkps: list[float] = []
        ents: list[float] = []

        def trajmaj(Xi):
            """Callback to track capacity measures (trajectory+magic+entanglement) during training."""
            eom = self.renyi_entropy_fast(self.QC.state)
            magics.append(eom)
            trajectory = self.minimize_function(Xi)
            traj.append(trajectory)
            entanglement = self.single_Q(self.QC.state, self.QC.n_qubits)
            ents.append(entanglement)
            gkp = self.gkp_fast(self.QC.state)
            gkps.append(gkp)

        self.QC.state = self.QC.run(angles=angles)
        trajmaj(angles)

        if method.lower() in ["gradient", "qng"]:
            prev_energy: float = self.minimize_function(angles)
            while diff > epsilon and count < quit_iterations:
                theta: list[Angle] = self.QC.get_params()
                gradients: list[Gradient] = self.get_gradient_vector(theta)

                if method == "gradient":
                    theta_update: list[Angle] = list(
                        np.array(theta) - rate * np.array(gradients)
                    )
                elif (
                    method == "QNG"
                ):  # some serious problems here, think we need renormalization
                    QFI = self.get_QFI(grad_list=self.gradient_list)
                    inverse = np.linalg.pinv(QFI)
                    f_inv_grad_psi = inverse.dot(np.array(gradients))
                    theta_update = list(np.array(theta) - rate * f_inv_grad_psi)

                if count % 100 == 0 and verbose is True:
                    print(
                        f"On iteration {count}, energy = {prev_energy}, diff is {diff}"
                    )

                energy: float = self.minimize_function(
                    theta_update
                )  # NB: PQC status update occurs here
                diff = np.abs(energy - prev_energy)

                trajmaj(theta_update)
                count += 1
                prev_energy = energy
        else:
            op_out = scipy.optimize.minimize(
                self.minimize_function,
                x0=angles,
                method=method,
                callback=trajmaj,
                tol=epsilon,
            )
            energy = op_out.fun
        return (energy, traj, magics, ents, gkps)
