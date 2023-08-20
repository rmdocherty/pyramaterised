import unittest

from math import isclose
import numpy as np
import qutip as qt
import scipy
import random

import pyramaterised as pyqc


class RandUnitary(pyqc.PQC):
    def __init__(self):
        self.initial_state = qt.basis(2, 0)
        self.n_qubits = 1

    def run(self, angles):
        # self._quantum_state = qt.random_objects.rand_unitary(2) * self._initial_state
        self.state = qt.random_objects.rand_ket_haar(2)
        return self.state


class Bell(pyqc.PQC):
    def __init__(self):
        self.n_qubits = 2
        self.state = qt.states.bell_state("11")

    def run(self, angles):
        return qt.states.bell_state("11")


class One:
    def __init__(self):
        self.n_qubits = 1
        self.state = qt.basis(1, 1)

    def run(self, angles):
        return qt.basis(1, 1)


class Haar(pyqc.PQC):
    def __init__(self, N):
        self.n_qubits = N
        circuit_state = qt.tensor([qt.basis(2, 0) for i in range(N)])
        haar = qt.random_objects.rand_unitary_haar(
            2**N, [[2 for i in range(N)], [2 for i in range(N)]]
        )
        self.state = haar * circuit_state

    def run(self, angles):
        N = self.n_qubits
        circuit_state = qt.tensor([qt.basis(2, 0) for i in range(N)])
        haar = qt.random_objects.rand_unitary_haar(
            2**N, [[2 for i in range(N)], [2 for i in range(N)]]
        )
        self.state = haar * circuit_state
        return self.state

    def cost(self):
        return None


class TestGates(unittest.TestCase):
    def test_basic_gates(self) -> None:
        """Single Hadamard gate acting on |0> basis state should just be (sqrt(2), sqrt(2))"""
        print("Testing basic gates:")
        test_H = pyqc.PQC(1)
        layer: list[pyqc.Gate] = [pyqc.fixed_R_y(0, 1, np.pi / 2)]
        test_H.add_layer(layer)
        out = test_H.run("random")
        x_component = np.real(out[1][0][0])
        y_component = np.real(out[0][0][0])
        over_sqrt2 = 1 / np.sqrt(2)
        assert isclose(
            x_component, over_sqrt2
        )  # isclose() checks if 2 numbers within an epsilon of each other
        print(
            f"Output of 1 Hadamard circuit on |0> basis is ({x_component}, {y_component}) and expected output is {over_sqrt2}\n"
        )

        """Apply 2 Hadamard gates using the layer structure to test if it works."""
        test_2H = pyqc.PQC(1)
        layer = [pyqc.H(0, 1)]
        test_2H.add_layer(layer, n=2)
        out = test_2H.run("random")
        assert out == qt.basis(2, 0)  # assert throws exception if conditional not true
        print("Action of 2 Hadamards on |0> is |0> again\n")


def check_iden(A):
    diag_entries = []
    non_zero_off_diags = []
    for i, row in enumerate(A):
        for j, column in enumerate(row):
            if i == j:
                diag_entries.append(column)
            elif column != 0 and i != j:
                non_zero_off_diags.append(column)
    assert len(non_zero_off_diags) == 0


class TestCircuits(unittest.TestCase):
    def test_NPQC(self):
        """
        Testing using an NPQC as defined in https://arxiv.org/pdf/2107.14063.pdf
        Defining property of NPQC is that QFI(theta_r) = Identity, where
        theta_r_i =  0 for R_x gates, pi/2 for R_y gates. This speeds up training when
        theta_r used as initial parameter vector. To check our QFI code is working,
        generate each NPQC with N qubits and P <= 2**(N/2) layers and check it's QFI
        is the identity when initialised with theta_r - this means every off diagonal
        element of the QFI should be 0, which is checked for in check_iden().
        """
        print("Testing NPQC and QFI code:")
        for N in range(4, 9, 2):  # step=2 for even N
            for P in range(
                1, 2 ** (N // 2) + 1
            ):  # works iff P < 2^(N/2) so agrees with paper.
                layers, theta_ref = pyqc.templates.NPQC_layers(P, N)
                NPQC = pyqc.PQC(N)
                for l in layers:
                    NPQC.add_layer(l)
                # need to set |psi> before making QFI measurements
                NPQC.state = qt.Qobj(NPQC.run(angles=theta_ref))
                NPQC_m = pyqc.measure.Measurements(NPQC)
                QFI = np.array(NPQC_m.get_QFI())
                masked = np.where(QFI < 10**-12, 0, QFI)
                check_iden(masked)
        print("QFIM of NPQC is identity for all qubit numbers from 4-8\n")

    def test_TFIM(self):
        print("Testing TFIM circuit and minimisation code:")
        N, p = 4, 4
        g_0, h_0 = 1, 0

        TFIM = pyqc.PQC(N)
        plus_state = (1 / np.sqrt(2)) * (qt.basis(2, 0) + qt.basis(2, 1))
        final_state = qt.tensor([plus_state for i in range(N)])

        hamiltonian = pyqc.templates.TFIM_hamiltonian(N, g=g_0, h=h_0)
        groundstate_energy, groundstate = hamiltonian.groundstate()
        a = hamiltonian.eigenenergies()
        TFIM.set_H(hamiltonian)

        TFIM_layers = pyqc.templates.TFIM_layers(p, N)
        for l in TFIM_layers:
            TFIM.add_layer(l)

        random_angles = [random.random() * np.pi for i in range(2 * p)]
        TFIM_m = pyqc.measure.Measurements(TFIM)
        out = TFIM_m.train(method="BFGS", angles=random_angles)
        print(f"Finished optimising TFIM circuit in {len(out[1])} iterations.\n")
        F = TFIM.fidelity(groundstate)
        assert isclose(F, 1)
        print(
            f"Overlap between QuTIP groundstate of TFIM Hamiltonian and circuit minimised state is {F:.4f}\n"
        )

    def test_qg_circuit(self):
        """Tests based on default circuit in arXiv:2102.01659v1 github"""
        print("Testing quantum geometry circuit:")
        qg_circuit = pyqc.PQC(4)
        init_layer: list[pyqc.Gate] = [
            pyqc.fixed_R_y(i, 4, np.pi / 4) for i in range(4)
        ]
        layer1: list[pyqc.Gate] = [
            pyqc.R_z(0, 4),
            pyqc.R_x(1, 4),
            pyqc.R_y(2, 4),
            pyqc.R_z(3, 4),
            pyqc.CHAIN(pyqc.CNOT, 4),
        ]
        layer2: list[pyqc.Gate] = [
            pyqc.R_x(0, 4),
            pyqc.R_x(1, 4),
            pyqc.R_x(2, 4),
            pyqc.R_y(3, 4),
            pyqc.CHAIN(pyqc.CNOT, 4),
        ]
        layer3: list[pyqc.Gate] = [
            pyqc.R_z(0, 4),
            pyqc.R_x(1, 4),
            pyqc.R_y(2, 4),
            pyqc.R_y(3, 4),
            pyqc.CHAIN(pyqc.CNOT, 4),
        ]

        qg_circuit.add_layer(init_layer)
        qg_circuit.add_layer(layer1)
        qg_circuit.add_layer(layer2)
        qg_circuit.add_layer(layer3, n=1)

        default_angles = [
            3.21587011,
            5.97193953,
            0.90578156,
            5.96054027,
            1.9592948,
            2.65983852,
            5.20060878,
            2.571074,
            3.45319898,
            0.17315902,
            4.73446249,
            3.38125416,
        ]

        energy = qg_circuit.cost(default_angles)
        print(f"Energy is {energy}, should be 0.46135870050914374\n")
        assert isclose(energy, 0.46135870050914374, abs_tol=1e-5)
        qg_m = pyqc.measure.Measurements(qg_circuit)
        efd = qg_m.get_effective_quantum_dimension(10**-12)
        assert isclose(efd, 12, abs_tol=1e-5)
        print(f"Effective quantum dimension is {efd}, should be 12\n")


class TestMeasurements(unittest.TestCase):
    def test_expressibility(self) -> None:
        """Tests based on Expr baselines from Fig 1 arXiv:1905.10876v1"""
        print("Testing expressibility code:")
        idle = pyqc.PQC(1)
        layer = [pyqc.I(0, 1)]
        idle.add_layer(layer)
        idle_m = pyqc.measure.Measurements(idle)
        i_e = idle_m.efficient_measurements(
            1200, measure_ent=False, measure_eom=False, measure_GKP=False
        )
        print(f"Idle circuit expr is {i_e['Expr']}, should be O(10) (4.317 in paper)\n")

        circuit_A = pyqc.PQC(1)
        layer = [pyqc.H(0, 1), pyqc.R_z(0, 1)]
        circuit_A.add_layer(layer)
        circuit_A_m = pyqc.measure.Measurements(circuit_A)
        a_e = circuit_A_m.efficient_measurements(
            1200, measure_ent=False, measure_eom=False, measure_GKP=False
        )
        print(f"Circuit_A expr is {a_e['Expr']}, should be O(0.1) (0.2 in paper)\n")

        circuit_B = pyqc.PQC(1)
        layer = [pyqc.H(0, 1), pyqc.R_z(0, 1), pyqc.R_x(0, 1)]
        circuit_B.add_layer(layer)
        circuit_B_m = pyqc.measure.Measurements(circuit_B)
        b_e = circuit_B_m.efficient_measurements(
            1200, measure_ent=False, measure_eom=False, measure_GKP=False
        )
        print(
            f"Circuit_B expr is {b_e['Expr']}, should be O(0.1) - O(0.01) (0.02 in paper)\n"
        )

        U: pyqc.PQC = RandUnitary()
        rum = pyqc.measure.Measurements(U)
        rum_expr = rum.expressibility(150)
        print(
            f"Random unitary expr is {rum_expr}, should be O(0.01) - O(0.001) (0.07 in paper)\n"
        )

    def test_entropy_of_magic(self) -> None:
        print("Testing entropy of magic of Haar states:")
        eoms = []
        eom_stds = []
        analytic = []
        gkps: list[float] = []

        for N in range(1, 8):
            h = Haar(N)
            hm = pyqc.measure.Measurements(h)
            eom = hm.efficient_measurements(
                2000,
                measure_expr=False,
                measure_ent=False,
                measure_eom=True,
                measure_GKP=False,
            )["Magic"]
            eoms.append(eom[0])
            eom_stds.append(eom[1])

            upper_bound = np.log((2**N) + 1) - np.log(2)
            avg_magic = np.log(3 + 2**N) - np.log(4)
            analytic.append(avg_magic)
            frac_avg = avg_magic / upper_bound
            print(
                f"{N} qubit Haar state EoM = {eom[0]:.5f} +/- {eom[1]:.6f} vs analytic expr = {avg_magic:.6f}\n"
            )

    def test_on_bell_state(self) -> None:
        bell = Bell()
        bell_m = pyqc.measure.Measurements(bell)
        gkp = bell_m.gkp_fast(bell.run("random"))
        eom = bell_m.renyi_entropy_fast(bell.run("random"))
        ent = bell_m.single_Q(bell.run("random"), bell.n_qubits)
        print(f"Reyni Entropy of Magic is {eom}, should be 0 for stabiliser state\n")
        assert isclose(eom, 0, abs_tol=1e-10)
        print(f"GKP Magic is {gkp}, should be 0 for stabiliser state\n")
        assert isclose(gkp, 0, abs_tol=1e-10)
        print(f"Entanglement is {ent}, should be 1 for Bell state\n")
        assert isclose(ent, 1, abs_tol=1e-10)

    def test_effective_hilbert_space(self) -> None:
        print(
            "Test finding |H_eff| for an n-qubit hardware efficient circuit - scales as 2**N: "
        )
        for n in range(2, 9, 2):  # 2, 9, 2
            generic_HE = pyqc.templates.generate_circuit("generic_HE", n, 2 * n)
            generic_HE_m = pyqc.measure.Measurements(generic_HE)
            f_samples = generic_HE_m._gen_f_samples(150)
            eff_H = generic_HE_m.find_eff_H(f_samples, n)
            print(
                f"|H_eff| for generic HE circuit for {n} qubits is {eff_H:.6f}, should be around {2**n}\n"
            )
            assert isclose(eff_H, 2**n, rel_tol=0.1)
        print(
            "Test finding |H_eff| for an n-qubit, half-filled zfsim circuit - scales as N choose N//2: "
        )
        for n in range(2, 9, 2):  # 2, 9, 2
            zfsim = pyqc.templates.generate_circuit("zfsim", n, n)
            zfsim_m = pyqc.measure.Measurements(zfsim)
            f_samples = zfsim_m._gen_f_samples(150)
            eff_H = zfsim_m.find_eff_H(f_samples, n)
            n_c_k = scipy.special.comb(n, n // 2)
            print(
                f"|H_eff| for zfsim circuit for {n} qubits is {eff_H:.6f}, should be around {n_c_k}\n"
            )
            assert isclose(eff_H, n_c_k, rel_tol=0.4)
        print(
            "Test finding |H_eff| for an 8-qubit, K filled zfsim circuit - scales as 8 choose K: "
        )
        N = 8
        for k in range(1, 5):
            zfsim = pyqc.templates.generate_circuit("zfsim", N, N)
            init_state = [qt.basis(2, 1) for i in range(k)] + [
                qt.basis(2, 0) for i in range(N - k)
            ]
            random.shuffle(init_state)
            tensored = qt.tensor(init_state)
            zfsim.initial_state = tensored
            zfsim_m = pyqc.measure.Measurements(zfsim)
            f_samples = zfsim_m._gen_f_samples(150)
            eff_H = zfsim_m.find_eff_H(f_samples, N)
            n_c_k = scipy.special.comb(N, k)
            print(
                f"|H_eff| for zfsim circuit for {k} spin-ups, {N} qubits is {eff_H:.6f}, should be around {n_c_k}\n"
            )
            assert isclose(eff_H, n_c_k, rel_tol=0.4)


if __name__ == "__main__":
    unittest.main()
