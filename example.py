import pyramaterised as pyqc
import random
import numpy as np

"""Define a 4-qubit, 6-layer PQC with X rotations on each 
   qubit and a chain of CNOT gates as entanglers"""
N: pyqc.QubitNumber = 4
rotations: list[pyqc.Gate] = [pyqc.R_x(i, N) for i in range(N)]
entanglers: list[pyqc.Gate] = [pyqc.CHAIN(pyqc.CNOT, N)]
# Generic HE PQC structure: a layer comprised of rotations then entanglements
layer = rotations + entanglers
circuit: pyqc.PQC = pyqc.PQC(N)
circuit.add_layer(layer, n=3)  # Add layer 3 times
print(circuit)

"""Make various capacity measurements of the circuit, and 
   train it using BFGS method."""
capacity = pyqc.measure.Measurements(circuit)
expr: float = capacity.expressibility(150)
eom: float = capacity.entropy_of_magic(150)
print(f"Expressibility is {expr}, Entropy of Magic is {eom}")
random_angles = [random.random() * np.pi for i in range(6 * N)]
# Minimise cost
out = capacity.train(method="BFGS", angles=random_angles)
# Maximise magic:
capacity.set_minimise_function(capacity.theta_to_magic)
out_magic = capacity.train(method="BFGS", angles=random_angles)
print(f"Magic when circuit optimised relative to ZZ hamiltonian: {out[2][-1]}")
print(f"Magic when circuit optimised for magic: {out_magic[2][-1]}")
