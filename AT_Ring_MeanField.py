from numba import jit, njit
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
import numpy as np
import scipy as sp
from scipy.constants import hbar, Boltzmann
from scipy.interpolate import UnivariateSpline
from quspin.operators import hamiltonian
from quspin.basis import boson_basis_1d
from quspin.tools.evolution import evolve
import datetime
import pytz
from tqdm import tqdm
import h5py
from multiprocessing import Pool
import os
import quspin

########################################################################################################################
# Set the initial plotting parameters
########################################################################################################################
# Colorblind friendly color scheme reordered for maximum efficiency
ok = {
    'blue': "#56B4E9",
    'orange': "#E69F00",
    'green': "#009E73",
    'amber': "#F5C710",
    'purple': "#CC79A7",
    'navy': "#0072B2",
    'red': "#D55E00",
    'black': "#000000",
    'yellow': "#F0E442",
    'grey': "#999999",
}

plt_params = {
    'figure.figsize': (4, 3),
    'figure.dpi': 300,
    'legend.fontsize': 8,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'axes.prop_cycle': plt.cycler('color', (ok[_] for _ in ok)),
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'lines.linewidth': 2,
}
plt.rcParams.update(plt_params)

########################################################################################################################
# Define Model Parameters
# Using Quspin, we will be solving the Gross-Pitaevskii equation (GPE)
# i d/dt ψ_j(t) = -J[ψ_{j-1}(t) + ψ_{j+1}(t)] + V(x) ψ_j(t) + g|ψ_j(t)|² ψ_j(t)
# For more information, see: https://github.com/weinbe58/QuSpin/blob/master/examples/notebooks/GPE.ipynb
########################################################################################################################
T = 20
L = 12
# calculate centre of chain
if L % 2 == 0:
    j0 = L // 2 - 0.5           # center of chain
else:
    j0 = L // 2               # center of chain
sites = np.arange(L) - j0   # centered chain sites around zero
degrees = np.array(sites) * 180 / sites[-1]

# Model Parameters
J = 1.0                     # Hopping Parameter
g = 2.0                     # Bose-Hubbard interaction strength

cooling_potential_width = 5
cooling_potential_offset = 0

n_ramps = 2                 # Define the number of ramps in the periodic potential
ramp_width = int(L/(n_ramps ** 2))   # Give the length of the ramp potential in degrees
assert n_ramps * ramp_width < L,\
    (f"A system of grid size {L} cannot accommodate {n_ramps} barriers of width {ramp_width},\n"
     f"as the total width of all barriers {n_ramps * ramp_width}!")
ramp_spacing = int(L / n_ramps)  # Give the spacing between centers of ramps



ramp_height = 2 * (J + g)  # Determine the maximum height of the
# Set the initial ramp center, and also grab the ends. The center is necessary for even grid sizes
ramp_center = [sites[int(0.5 * ramp_spacing)] + (1 - ramp_width % 2) / 2]
ramp_start = [ramp_center[0] - int(0.5 * ramp_width) + (1 - ramp_width % 2) / 2]
ramp_end = [np.floor(ramp_center[0] + 0.5 * ramp_width) - (1 - L % 2) / 2]
# Use a loop to set the remaining number of ramps
for _ in range(1, n_ramps):
    ramp_center.append(sites[int(0.5 * ramp_spacing) + _ * ramp_spacing] + (1 - ramp_width % 2) / 2)
    ramp_start.append(ramp_center[_] - int(0.5 * ramp_width) + (1 - ramp_width % 2) / 2)
    ramp_end.append(ramp_center[_] + int(0.5 * ramp_width) - (1 - ramp_width % 2) / 2)



def cooling_potential(j):
    v_cooling = cooling_potential_width * (j - cooling_potential_offset) ** 2
    return v_cooling - v_cooling.min()


def ring_with_ramp(j):
    """
    This is absolutely the MOST over-engineered function I have ever written.
    I would have saved literal days if I had just created the list manually.
    :param j:
    :return:
    """
    ring = np.zeros(len(j))
    for _ in range(n_ramps):
        ramp = np.where((j >= ramp_start[_]) & (j <= ramp_end[_]))[0]
        for i in ramp:
            ring[i] = ramp_height * ((j[i] - ramp_end[_]) / (ramp_start[_] - ramp_end[_]))
    return ring


def GPE_imag_time(tau, ϕ, init_hamiltonian, u):
    """
    This function solves the real-valued GPE in imaginary time:
    $$ -\dot\phi(\tau) = Hsp(t=0)\phi(\tau) + U |\phi(\tau)|^2 \phi(\tau) $$
    """
    return -(init_hamiltonian.dot(ϕ, time=0) + u * np.abs(ϕ) ** 2 * ϕ)

def GPE(time, psi):
    """
    This function solves the complex-valued time-dependent GPE:
    $$ i\dot\psi(t) = Hsp(t)\psi(t) + U |\psi(t)|^2 \psi(t) $$
    """
    # solve static part of GPE
    psi_dot = Hsp.static.dot(psi) + g * np.abs(psi) ** 2 * psi
    # We don't have a dynamic portion of the hamiltonian
    return -1j * psi_dot


########################################################################################################################
# Quickly define the archive
########################################################################################################################
tag = f'Ring_L{L}-T{T}-J{J}-g{g:.2f}'
savesfolder = tag.replace('.', ',')
parent_dir = "./Archive_Data/QuspinGPE"
try:
    os.mkdir(parent_dir)
    print(f'Parent Directory created, saved to: {parent_dir}')
except:
    FileExistsError
    print(f'Parent directory check passed! \nResults will be saved to the path {parent_dir}\n')
path = os.path.join(parent_dir, savesfolder)
savespath = 'Archive_Data/QuspinGPE/' + str(savesfolder) + '/'

try:
    os.mkdir(path)
    print(f'Simulation Directory "{savesfolder}" created')
except:
    FileExistsError
    print('WARNING: The directory you are saving to already exists!!! \nYou may be overwriting previous data (; n;)\n')

########################################################################################################################
# Plot the potential and save before proceeding
########################################################################################################################
fig = 0
plt.figure(fig)
fig += 1
potential = ring_with_ramp(sites)
init_potential = cooling_potential(sites)
plt.plot(degrees, potential, '*-', label='Ring Potential')
plt.plot(degrees, init_potential, label='Cooling Potential')
plt.xlim(degrees[0], degrees[-1])
plt.ylim(-0.25, 1.1*potential.max())
plt.xticks(np.arange(-180, 180, step=60))
plt.legend(loc='upper center')
plt.tight_layout()
plt.savefig(savespath + 'Potential.pdf')
plt.savefig(savespath + 'Potential.png')
plt.show()


########################################################################################################################
# Set up the models
########################################################################################################################
# Define the site coupling lists
hopping = [[-J, _, (_+1) % L] for _ in range(L-1)]
trap = [[init_potential[_], _] for _ in range(L)]
v = [[potential[_], _] for _ in range(L)]
# Define the static and dynamic lists
static = [['+-', hopping], ['-+', hopping], ['n', trap]]  # 'n' needs to be potential, changed to trap for testing
static_gs = [['+-', hopping], ['-+', hopping], ['n', trap]]

# define basis
basis = boson_basis_1d(L, Nb=1, sps=2)
# build Hamiltonian
Hsp = hamiltonian(static, [], basis=basis, dtype=np.float64)
init_H = hamiltonian(static_gs, [], basis=basis, dtype=np.float64)

E, V = init_H.eigsh(time=0.0, k=1, which='SA')
tau = np.linspace(0, 15, 21)
# WARNING, the code below only works because there is only 1 boson
init_GPE_params = (init_H, g)
GPE_params = (Hsp, g)
ψ_0 = np.zeros(L)
if L % 2 == 0:
    init_sites = [int(L / 2 - 1), int(L / 2)]
else:
    init_sites = [int(L / 2 - 1), int(L / 2), int(L / 2 + 1)]

for _ in init_sites:
    ψ_0[_] += E * np.sqrt(L) / len(init_sites)

#print(f'The shape of V is{np.shape(V)}')
#print(f'The shape of ψ_0 is {np.shape(ψ_0)}')
#print(f'The shape of the hamiltonian is {np.shape(init_H)}')
#print(f'The shape the function wanted is {np.shape(V[:, 0]*np.sqrt(L))}')
#ψ_0 = V[:, 0]*np.sqrt(L)
gs_density = np.abs(np.array(ψ_0)) ** 2
#gs_density *= (1 / np.sum(gs_density))
#ψ_0 *= np.sqrt(1 / np.sum(gs_density))

#ψ_0 = np.array(ψ_0) * (1/np.sqrt(np.abs(np.sum(ψ_0))))
ψ_τ = evolve(ψ_0, tau[0], tau, GPE_imag_time, f_params=init_GPE_params, imag_time=True, real=True, iterate=True)
for _, psi0 in enumerate(ψ_τ):
    E_GS = (init_H.matrix_ele(psi0, psi0, time=0) + 0.5 * g * np.sum(np.abs(psi0) ** 4)).real
    plt.plot(degrees, abs(ψ_0) ** 2, marker='s', alpha=0.2,
             label='$|\\phi_j(0)|^2$')
    plt.plot(degrees, abs(psi0) ** 2, marker='o',
             label='$|\\phi_j(\\tau)|^2$')
    plt.plot()
    plt.xlabel('$\\mathrm{lattice\\ sites}$', fontsize=14)
    plt.title('$J\\tau=%0.2f,\\ E_\\mathrm{GS}(\\tau)=%0.4fJ$' % (tau[_], E_GS)
              , fontsize=14)
    plt.ylim([-0.01, np.max(np.abs(ψ_0) ** 2) + 0.01])
    plt.legend(fontsize=14)
    plt.draw()  # draw frame
    plt.pause(0.005)  # pause frame
    plt.clf()  # clear figure
print(f'Final ground state energy: {E_GS}')

t = np.linspace(0.0, T, 101)
density = []
energies = []
ψ_t = evolve(psi0, t[0], t, GPE, iterate=True, atol=1E-12, rtol=1E-12)
for _, ψ in enumerate(ψ_t):
    E_t = (Hsp.matrix_ele(ψ, ψ, time=t[_]) + 0.5 * g * np.sum(np.abs(ψ)) ** 4).real
    energies.append(E_t)
    density.append(np.abs(ψ) ** 2)
    plt.plot(degrees, abs(psi0) ** 2,  marker='s', alpha=0.2
             , label='$|\\psi_{\\mathrm{GS},j}|^2$')
    plt.plot(degrees, abs(ψ) ** 2,  marker='o', label='$|\\psi_j(t)|^2$')
    plt.plot(degrees, init_potential, '--', label='$\\mathrm{trap}$')
    plt.xlim([degrees[0], degrees[-1]])
    plt.ylim([-0.01, 10])
    plt.xlabel('$\\mathrm{lattice\\ sites}$', fontsize=14)
    plt.title('$Jt=%0.2f,\\ E(t)-E_\\mathrm{GS}=%0.4fJ$' % (t[_], E_t - E_GS), fontsize=14)
    plt.legend(loc='upper right', fontsize=14)
    plt.draw()  # draw frame
    plt.pause(0.00005)  # pause frame
    plt.clf()  # clear figure

plt.figure(fig)
fig += 1
plt.imshow(np.flip(density, 1),
           extent=[degrees[0], degrees[-1], t[0], t[-1]],
           aspect=(degrees[-1] - degrees[0]) / (t[-1] - t[0])
           )
plt.xlabel('x (degrees)')
plt.ylabel('time')
plt.tight_layout()
plt.show()
