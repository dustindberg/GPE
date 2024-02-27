from collections import namedtuple
import numpy as np
from numba import jit, njit
from scipy.integrate import ode
import matplotlib.pyplot as plt
import h5py
import os

########################################################################################################################
# Define functions to be used
########################################################################################################################
quantum_system = namedtuple('quantum_system', ['J', 'g', 'V', 'ψ', 'is_open_boundary'])


def Hamiltonian(t: float, ψ: np.ndarray, qsys: quantum_system):
    hop = qsys.J

    Hψ = np.zeros_like(ψ)

    Hψ[1:-1] = -hop * (ψ[:-2] + ψ[2:])

    if qsys.is_open_boundary:
        # imposing the open boundary conditions
        Hψ[0] = - hop * ψ[1]
        Hψ[-1] = - hop * ψ[-2]
    else:
        # imposing the periodic boundary conditions
        Hψ[0] = -hop * (ψ[-1] + ψ[1])
        Hψ[-1] = -hop * (ψ[-2] + ψ[0])

    Hψ += (qsys.V + qsys.g * np.abs(ψ) ** 2) * ψ

    return Hψ


def img_propagator(τ: float, n: int, qsys: quantum_system):
    time, dtimes = np.linspace(0, τ, n + 1, retstep=True)

    solver = ode(
        lambda current_time, Ψ: -Hamiltonian(current_time, Ψ, qsys)
    ).set_integrator('zvode')

    ψ = np.copy(qsys.ψ)

    for t in time[1:-1]:
        solver.set_initial_value(ψ, t)
        ψ = solver.integrate(t + dtimes)
        ψ /= np.linalg.norm(ψ)

    # update the wavefunction of the system
    qsys.ψ[:] = ψ

    return qsys


def propagator(time: np.ndarray, qsys: quantum_system):
    solver = ode(
        lambda t, ψ: -1j * Hamiltonian(t, ψ, qsys)
    ).set_integrator('zvode')

    solver.set_initial_value(qsys.ψ, time[0])

    wavefunctions = [qsys.ψ.copy(), ]
    wavefunctions.extend(
        solver.integrate(t) for t in time[1:]
    )

    qsys.ψ[:] = wavefunctions[-1]

    return np.array(wavefunctions)


def get_current(wavefunction: np.ndarray, qsys: quantum_system):
    ψ = np.copy(wavefunction)
    n_sites = len(ψ)
    hop = qsys.J
    ψ_star = ψ.conj()
    current = []
    for ψt in ψ:
        ψt_plus = np.roll(ψt, -1, axis=0)
        ψt_minus = np.roll(ψt, 1, axis=0)
        ρ = 1j * hop * (
            ψt.conj().T * (ψt_minus + ψt_plus)
            - (ψt_minus + ψt_plus).conj().T * ψt
            )
        current.append(np.sum(ρ).real)
    return current


########################################################################################################################
# Define Model Parameters
# Using scipy's ODE solver, we will be solving the Gross-Pitaevskii equation (GPE)
# i d/dt ψ_j(t) = -J[ψ_{j-1}(t) + ψ_{j+1}(t)] + V(x) ψ_j(t) + g|ψ_j(t)|² ψ_j(t)
########################################################################################################################
# Physical System Parameters
L = 18              # Number of sites
J = 1.0             # hopping strength
g = 10.0            # Bose-Hubbard interaction strength
τ_imag = 10         # Imaginary time propagation
ni_steps = 1000     # Number of steps for imaginary time propagation
t_prop = 10         # Time of propagation
n_steps = t_prop * 200  # Number of steps for real-time propagation
times = np.linspace(0, t_prop, n_steps)
# Set params for the potential
cooling_potential_width = 0.5
cooling_potential_offset = 0
# Set params for
n_ramps = 2                 # Define the number of ramps in the periodic potential
ramp_height = 1 * J

# calculate center of chain
if L % 2 == 0:
    j0 = L // 2 - 0.5           # center of chain
else:
    j0 = L // 2               # center of chain
sites = np.arange(L) - j0
degrees = np.array(sites) * 180 / sites[-1]


def cooling_potential(j):
    v_cooling = cooling_potential_width * (j - cooling_potential_offset) ** 2
    return v_cooling - v_cooling.min()


ramp_width = int(L/(n_ramps ** 2))   # Give the length of the ramp potential in degrees
assert n_ramps * ramp_width < L,\
    (f"A system of grid size {L} cannot accommodate {n_ramps} barriers of width {ramp_width},\n"
     f"as the total width of all barriers {n_ramps * ramp_width}!")
ramp_spacing = int(L / n_ramps)  # Give the spacing between centers of ramps
# Set the initial ramp center, and also grab the ends. The center is necessary for even grid sizes
ramp_center = [sites[int(0.5 * ramp_spacing)] + (1 - ramp_width % 2) / 2]
ramp_start = [ramp_center[0] - int(0.5 * ramp_width) + (1 - ramp_width % 2) / 2]
ramp_end = [np.floor(ramp_center[0] + 0.5 * ramp_width) - (1 - L % 2) / 2]

# Use a loop to set the remaining number of ramps
for _ in range(1, n_ramps):
    ramp_center.append(sites[int(0.5 * ramp_spacing) + _ * ramp_spacing] + (1 - ramp_width % 2) / 2)
    ramp_start.append(ramp_center[_] - int(0.5 * ramp_width) + (1 - ramp_width % 2) / 2)
    ramp_end.append(ramp_center[_] + int(0.5 * ramp_width) - (1 - ramp_width % 2) / 2)


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


########################################################################################################################
# Set the initial plotting parameters and the saving architecture
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

fnum = 0
extent = [degrees[0], degrees[-1], times[0], times[-1]]
aspect = (degrees[-1] - degrees[0]) / (times[-1] - times[0])

# Save file architecture
tag = f'Ring_L{L}-T{t_prop}-J{J}-g{g:.2f}-n_ramps{n_ramps}'
savesfolder = tag.replace('.', ',')
parent_dir = "./Archive_Data/ODE_Solver"
try:
    os.mkdir(parent_dir)
    print(f'Parent Directory created, saved to: {parent_dir}')
except:
    FileExistsError
    print(f'Parent directory check passed! \nResults will be saved to the path {parent_dir}\n')
path = os.path.join(parent_dir, savesfolder)
savespath = 'Archive_Data/ODE_Solver/' + str(savesfolder) + '/'

try:
    os.mkdir(path)
    print(f'Simulation Directory "{savesfolder}" created')
except:
    FileExistsError
    print('WARNING: The directory you are saving to already exists!!! \nYou may be overwriting previous data (; n;)\n')

# Store params in a retrievable
params = {
    'L': L,
    'J': J,
    'g': g,
    't_prop': t_prop,
    'n_steps': n_steps,
    'cooling_potential_width': cooling_potential_width,
    'cooling_potential_offset': cooling_potential_offset,
    'n_ramps': n_ramps,
    'ramp_height': ramp_height
}

"""with h5py.File(savespath+tag+'_data.hdf5', "w") as file:
    parameters_save = file.create_group('params')
    file.create_group('GPE')
    file.create_group('SE')
    for _, __ in params.items():
        parameters_save[_] = __"""

########################################################################################################################
# Establish systems and evolve
########################################################################################################################
V_trapping = cooling_potential(sites)
V_propagation = ring_with_ramp(sites)

# Define the quantum systems for the Schrödinger Gross-Pitaevskii equations
qsys_gpe = quantum_system(J=J, g=g, V=V_trapping, ψ=np.ones(L, complex), is_open_boundary=False)
qsys_se = quantum_system(J=J, g=0, V=V_trapping, ψ=np.ones(L, complex), is_open_boundary=False)
qsys_gpe = img_propagator(τ_imag, ni_steps, qsys_gpe)
qsys_se = img_propagator(τ_imag, ni_steps, qsys_se)

"""with h5py.File(savespath+tag+'_data.hdf5', "a") as file:
    gpe = file.group['GPE']
    se = file.group['SE']
    gpe.create_dataset('ground_state', data=qsys_gpe.ψ)
    se.create_dataset('ground_state', data=qsys_se.ψ)"""

# Set the propagation potential
qsys_gpe.V[:] = V_propagation
qsys_se.V[:] = V_propagation
gpe_wavefunction = propagator(times, qsys_gpe)
se_wavefunction = propagator(times, qsys_se)

gpe_current = get_current(gpe_wavefunction, qsys_gpe)
se_current = get_current(se_wavefunction,qsys_se)

plt.figure(fnum, figsize=(6, 3))
fnum += 1
plt.subplot(121)
plt.title("$g=%0.2f$" % g)
plt.imshow(
    np.abs(gpe_wavefunction) ** 2,
    extent=extent,
    aspect=aspect,
    interpolation='nearest',
    origin='lower'
)
plt.xticks(np.arange(degrees[0], degrees[-1], step=60))
plt.ylabel("time")
plt.colorbar()

plt.subplot(122)
plt.title(r"$g = 0$")
plt.imshow(
    np.abs(se_wavefunction) ** 2,
    extent=extent,
    aspect=aspect,
    interpolation='nearest',
    origin='lower'
)
plt.xticks(np.arange(degrees[0], degrees[-1], step=60))
plt.ylabel("time")
plt.colorbar()

plt.figure(fnum)
fnum += 1
plt.plot(times, gpe_current, label=r'$g=%0.2f$'%g)
plt.plot(times, se_current, '--', label=r'$g=0$')
plt.xlabel('Time')
plt.ylabel('Current')
plt.show()

