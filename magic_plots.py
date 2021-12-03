#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 12:15:15 2021

@author: ronan
"""
import PQC_lib as pqc
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import random
from measurement import Measurements
from circuit_structures import *
from helper_functions import pretty_graph, extend

#%%
p = 10
N = 4
all_entropies = []
exprs = []

N_repeats = 20 #100

for n in range(N_repeats):
    clifford_layers = gen_clifford_circuit(p, N)
    entropies = []
    if n % 100 == 0:
        print(f"{n} circuits computed")
    for i in range(p // 2): #want to insert in middle of circuit
        insertion_point = i + p // 4 #insert from the quarter upwards
        clifford_layers[insertion_point] = clifford_layers[insertion_point] + [pqc.T(i, N) for i in range(N)]
        insert_circuit = pqc.PQC(N)
        for l in clifford_layers:
            insert_circuit.add_layer(l)
        insert_circuit._quantum_state = qt.Qobj(insert_circuit.run())
        i_c_m = Measurements(insert_circuit)
        e = i_c_m.efficient_measurements(100, expr=True, ent=False, eom=True)
        entropies.append(e['Magic'][0])
        exprs.append(e['Expr'])

    entropies = np.array(entropies)
    max_magic = np.log((2**N) + 1) - np.log(2)
    entropies = entropies / max_magic
    all_entropies.append(entropies)

#%%
stderr = True
magics = np.mean(all_entropies, axis=0)
if stderr:
    magic_stds = np.std(all_entropies, axis=0) / np.sqrt(len(all_entropies))
    label_str = f"Standard error of the mean over {len(all_entropies)} repeats"
else:
    magic_stds = np.std(all_entropies, axis=0)
    label_str = f"Standard deviation over {len(all_entropies)} repeats"
    
def plot_magic(N, p, magics, magic_stds, color, std_color, label_str, title_str):
    n_t_gates = np.array([(i + 1) for i in range(p // 2)])
    plt.errorbar(n_t_gates, magics, yerr=magic_stds, ls="", marker=".", lw=2, ms=0, color=std_color)
    plt.plot(n_t_gates, magics, ls="-", marker=".", lw=4, ms=15, color=color, label=label_str)
    pretty_graph("Number of T-Gates layers", "Fractional Reyni Entropy of Magic", title_str, 20)
    plt.legend(fontsize=18)


default_title = f"T-Gate injection on {N}-qubit, {p}-layer Clifford Circuit"
plot_magic(N, p, magics, magic_stds, "cornflowerblue", "darkblue", label_str, default_title)
plt.hlines(exprs[0], 0, np.array([(i + 1) for i in range(p // 2)])[-1])

#%%
files = [f"magic_t_gate_1000r_{n}q_40l.npy" for n in range(3, 6)] #"magic_t_gate_1000r_4q_4l.npy", "magic_t_gate_1000r_5q_40l.npy"
stds = [f[:5] + "_std" + f[5:] for f in files]
magic_paths = ["data/" + f for f in files]
std_paths = ["data/" + f for f in stds]
magic_colors = ["lightgreen", "cornflowerblue", "orange"]
std_colors = ["darkgreen", "darkblue", "darkorange"]

title_str = "T-Gate injection on N-qubit, 40-layer random Clifford Circuit"
p = 40
for i in range(len(files)):
    N = i + 3
    label = f"{N} qubits"
    magic = np.load(magic_paths[i])
    magic_std = np.load(std_paths[i])
    plot_magic(N, p, magic, magic_std, magic_colors[i], std_colors[i], label, title_str)

#%%

random.seed(1)

p = 150
N = 4
d = 2**N
N_gates = 50
N_repeats = 100

all_entropies = []
all_stds = []
all_exprs = []
max_magic = np.log(d + 1) - np.log(2)

#%%
N_repeats = 10#50

for i in range(N_repeats):
    print(f"Iteration {i}")
    entropies = [0]
    stds = [0]
    exprs = [1]
    clifford_layers = gen_clifford_circuit(p, N, method='fixed')
    for g in range(N_gates):
        insertion_point = random.randint(p//4, (3 * p) //4)
        q_on = random.randint(0, N - 1)
        clifford_layers[insertion_point] = clifford_layers[insertion_point] + [pqc.T(q_on, N)]
        insert_circuit = pqc.PQC(N)
        for l in clifford_layers:
            insert_circuit.add_layer(l)
        insert_circuit._quantum_state = qt.Qobj(insert_circuit.run())
        i_c_m = Measurements(insert_circuit)
        e = i_c_m.efficient_measurements(1, expr=True, ent=False, eom=True)
        entropies.append(e['Magic'][0])
        stds.append(e['Magic'][1])
        exprs.append(e['Expr'])
    all_exprs.append(exprs)
    all_entropies.append(entropies)
    all_stds.append(stds)

#%%

magics = np.mean(all_entropies, axis=0) / max_magic
stds = np.std(all_entropies, axis=0) / max_magic
stderrs = stds / np.sqrt(len(stds))
n_t_gates = np.array(range(len(magics)))
default_title = f"T-Gate injection on {N}-qubit, {p}-layer Clifford Circuit"
plt.figure(default_title.strip(","))

haar_random_magic = np.log(3 + d) - np.log(4)
normalised_haar = haar_random_magic / max_magic
plt.hlines(normalised_haar, 0, n_t_gates[-1], lw=2, ls='dotted', label="Haar random magic", color='red')
plt.errorbar(n_t_gates, magics, yerr=stderrs, lw=4, label=f"Clifford circuit magic, {len(all_entropies)} repeats")


def f(theta, d):
    return (7*d**2 - 3*d + d*(d+3)*np.cos(4*theta) - 8) / (8*(d**2 - 1))
k_doped_linear_magic = 1 - ((3 + d)**(-1)) * (4 + (d - 1) * f(np.pi / 4, d) **n_t_gates)
k_doped_reyni_maigc = -1 * np.log(1 - k_doped_linear_magic)
k_doped_reyni_maigc_normalised = k_doped_reyni_maigc / max_magic

plt.plot(n_t_gates, k_doped_reyni_maigc_normalised, label="Theoretical model", lw=4, ls='--')
pretty_graph("Number of T gates", "Fractional magic", default_title, 20)
plt.legend(fontsize=18)

#%%
P, N = 4, 4

layers, theta_ref = NPQC_layers(P, N)
train_NPQC = pqc.PQC(N)
train_NPQC.set_H(TFIM_hamiltonian(N, 1, 0))
train_NPQC.set_initial_state(plus_state)
for l in layers:
    train_NPQC.add_layer(l)

train_NPQC_m = Measurements(train_NPQC)
ener, traj, magics = train_NPQC_m.train(method="gradient", magic=True, angles=theta_ref) #theta_ref
#%% Really interesting results - when initialised with theta_ref, training always seems to take 1762 iterations and gives really nice gaussian
#magics = np.load('data/magics_npqc_training_4l_4q.npy')
max_magic = np.log((2**N) + 1) - np.log(2)
magics = np.array(magics) / max_magic
iterations = range(len(magics))
plt.figure("NPC training magic")
plt.plot(iterations, magics, lw=4, color="green")
pretty_graph("Training Iteration", "Fractional Reyni Entropy of Magic", "Magic during NPQC BFGS training initialised with reference param", 20)

plt.figure("NPQC training trajectory")
plt.plot(iterations, traj, lw=4, color="orange")
pretty_graph("Iterations", "Cost function", "Cost function vs iterations for NPQC initialised with reference param", 20)


#%%
random.seed(1000)
N = 2
p = 2
trainer = "BFGS"

init_layer = [pqc.fixed_R_y(i, N, np.pi / 4) for i in range(N)]
layer1 =  [pqc.R_y(i, N) for i in range(N)] + [pqc.R_z(i, N) for i in range(N)] + [pqc.CHAIN(pqc.CNOT, N)] #+ [pqc.CHAIN(pqc.CNOT, N)]  #[pqc.R_y(i, N) for i in range(N)] +


all_entropies = []
all_trajectories = []
all_states = []
clifford_angles = [np.pi/2 for i in range(2*4*p)]
hamiltonian = TFIM_hamiltonian(N, g=1)
groundstate = hamiltonian.groundstate()[1]


#%%
for i in range(10):
    print(i)
    qg_circuit = pqc.PQC(N)
    qg_circuit.add_layer(layer1, n=p)
    qg_circuit.set_H(hamiltonian)
    qg_circuit_m = Measurements(qg_circuit)
    random_angles = [random.random()*2*np.pi for i in range(2*4*p)] #should this be haar random?
    out = qg_circuit_m.train(method=trainer, trajectory=True, magic=True, angles=random_angles, rate=0.001, epsilon=1e-6)
    all_entropies.append(out[2])
    all_trajectories.append(out[1])
    print(qg_circuit_m._QC._fidelity(groundstate))

#%%
count = 0
for state_list in all_states:
    print(f"Initilization {count}")
    for state in state_list:
        if state == state_list[-1]:
            print(state)
            print(state.ptrace(0))
            print(state.ptrace(1))
            print(qt.expect(qg_circuit.H, state))
            print("\n")
    count += 1

#%%
magics = extend(all_entropies)
trajectories = extend(all_trajectories)

n_avg = len(magics)

max_magic = np.log((2**N) + 1) - np.log(2)

title=f"{N} qubit {p} layer HE PQC BFGS Ising Hamiltonian"
plt.figure(title + "_"+ trainer)
for m in magics:
    iterations = range(len(m))
    magic = np.array(m) / max_magic
    if m.all() == magics[-1].all():
        plt.plot(iterations, magic, lw=4, color="green", label=trainer)
    else:
        plt.plot(iterations, magic, lw=4, color="green")
pretty_graph("Training Iteration", "Fractional Reyni Entropy of Magic", title, 20)
#plt.legend(fontsize=18)

plt.figure("Trajectories")
for t in trajectories:
    iterations = range(len(t))
    if t.all() == trajectories[-1].all():
        plt.plot(iterations, t, lw=4, color="orange", label=trainer)
    else:
        plt.plot(iterations, t, lw=4, color="orange")
pretty_graph("Iterations", "Cost function", title, 20)
#plt.legend(fontsize=18)


plt.figure("Average magic")
avg_magic = np.mean(magics, axis=0) / max_magic
iterations = range(len(avg_magic))
std_magic = np.std(magics, axis=0) / max_magic
stderr_magic = std_magic / np.sqrt(n_avg)
plt.errorbar(iterations, avg_magic, yerr=stderr_magic, color='lightgreen', lw=3, marker="", alpha=0.5, label=f"Standard error over {n_avg} repeats")
plt.plot(iterations, avg_magic, lw=4, color="green", label=trainer)
pretty_graph("Training Iteration", "Fractional Reyni Entropy of Magic", title, 20)
plt.legend(fontsize=18)

plt.figure("Average cost")
avg_traj = np.mean(trajectories, axis=0)
iterations = range(len(avg_traj))
std_traj = np.std(trajectories, axis=0)
std_err_traj = std_traj / np.sqrt(n_avg)
plt.errorbar(iterations, avg_traj, yerr=std_err_traj, color='wheat', lw=3, marker="", alpha=0.5, label=f"Standard error over {n_avg} repeats")
plt.plot(iterations, avg_traj, color="orange", lw=4, label=trainer)
pretty_graph("Iterations", "Cost function", title, 20)

plt.legend(fontsize=18)

#%%
"""
Measure TFIM magic for a range of different initilizations
"""

N, p = 4, 4
N_repeats = 10
g, h = 1, 0
random.seed(3)
all_entropies = []
all_trajectories = []

#%%
for i in range(5):
    print(i)
    TFIM = pqc.PQC(N)
    #need to use |+> as initial state for TFIM model
    plus_state = (1/np.sqrt(2)) * (qt.basis(2,0) + qt.basis(2,1))
    #TFIM.set_initial_state(plus_state)
    
    hamiltonian = TFIM_hamiltonian(N, g=g, h=h)
    groundstate = hamiltonian.groundstate()[1]
    TFIM.set_H(hamiltonian)
    TFIM.set_initial_state(plus_state)
    TFIM_layers = gen_TFIM_layers(p, N)
    for l in TFIM_layers:
        TFIM.add_layer(l)
    
    random_angles = [random.random()*2*np.pi for i in range(2*p)]
    
    TFIM_m = Measurements(TFIM)
    out = TFIM_m.train(method='gradient', rate=0.001, epsilon=1e-6, angles=random_angles, magic=True, trajectory=True)
    all_entropies.append(out[2])
    all_trajectories.append(out[1])

#%%
plt.figure("raw magic of TFIM")
N = N
max_magic = np.log((2**N) + 1) - np.log(2)
for m in all_entropies:
    iterations = range(len(m))
    magics = np.array(m) / max_magic
    plt.plot(iterations, magics, lw=4, color="green")
pretty_graph("Training Iteration", "Fractional Reyni Entropy of Magic", f"Magic during {N} qubit {p} layer TFIM training initialised with random angles", 20)

plt.figure("average magic of TFIM")

padded_magics = extend(all_entropies)
n_avg = len(padded_magics)

avg_magic = np.mean(padded_magics, axis=0) / max_magic
std_magic = np.std(padded_magics, axis=0) / max_magic
iterations = range(len(avg_magic))
stderr_magic = std_magic / np.sqrt(n_avg)
plt.errorbar(iterations, avg_magic, yerr=stderr_magic, color='lightgreen', lw=3, marker="", alpha=0.5, label=f"Standard error")
plt.plot(iterations, avg_magic, lw=4, color="green", label=f"Average over {n_avg} repeats")
pretty_graph("Training Iteration", "Fractional Reyni Entropy of Magic", f"Average Magic during {N} qubit {p} layer TFIM training initialised with random angles", 20)
plt.legend(fontsize=18)

#%%
plt.figure("Average cost")
padded_traj = extend(all_trajectories)
avg_traj = np.mean(padded_traj, axis=0)
std_traj = np.std(padded_traj, axis=0)
std_err_traj = std_traj / np.sqrt(n_avg)
plt.errorbar(iterations, avg_traj, yerr=std_err_traj, color='wheat', lw=3, marker="", alpha=0.5, label=f"Standard error")
plt.plot(iterations, avg_traj, color="orange", lw=4, label=f"Average over {n_avg} repeats")
#plt.hlines(BFGS_min, 0, iterations[-1], lw=2, ls='dotted', label="BFGS min", color='red')
pretty_graph("Iterations", "Cost function", f"Cost function vs iterations for {N} qubit {p} layer TFIM training initialised with random angles", 20)

plt.legend(fontsize=18)

#%%
start = "data/"
string = "_2q_5l_HE_"
end = ".npy"

magic_grad = np.load(start+"magic"+string+"grad"+end)
traj_grad = np.load(start+"cost"+string+"gradient"+end)

magic_bfgs = np.load(start+"magic"+string+"bfgs"+end)
traj_bfgs = np.load(start+"cost"+string+"bfgs"+end)

converge = False
for i in range(1, len(traj_grad)):
    if np.abs(traj_grad[i] - traj_grad[i-1]) <= 1e-6 and converge is False:
        print(i)
        grad_conv = i - 1
        converge=True

converge = False
for i in range(1, len(traj_bfgs)):
    if np.abs(traj_bfgs[i] - traj_bfgs[i-1]) <= 1e-6 and converge is False:
        print(i)
        bfgs_conv = i -1 
        converge=True
#%%
scale_factor = grad_conv / 8
iter_grad = range(len(magic_grad))
iter_bfgs = np.array(range(len(magic_bfgs)))
plt.plot(iter_grad, magic_grad)
plt.plot(iter_bfgs * scale_factor, magic_bfgs)