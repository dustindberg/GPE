from collections import namedtuple
import numpy as np
import scipy.ndimage
from numba import jit, njit
from scipy.integrate import ode
from copy import deepcopy
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


def get_energy(t: float, ψ: np.ndarray, qsys: quantum_system):
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

    Hψ += (qsys.V + 0.5 * qsys.g * np.abs(ψ) ** 2) * ψ

    return np.vdot(Hψ, ψ)


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

#    return qsys


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
    ψ_plus = np.roll(ψ, -1, axis=1)
    ψ_minus = np.roll(ψ, 1, axis=1)
    current = -1j * qsys.J * (ψ_minus.conj() * ψ - ψ_minus * ψ.conj())
    return current.sum(axis=1)


def get_moving_avg(x):
    return np.cumsum(x) / np.arange(1, x.size + 1)
    #return scipy.ndimage.uniform_filter1d(x, N, mode='nearest')


########################################################################################################################
# Define Model Parameters
# Using scipy's ODE solver, we will be solving the Gross-Pitaevskii equation (GPE)
# i d/dt ψ_j(t) = -J[ψ_{j-1}(t) + ψ_{j+1}(t)] + V(x) ψ_j(t) + g|ψ_j(t)|² ψ_j(t)
########################################################################################################################
# Physical System Parameters
L = 20              # Number of sites
J = 1.0             # hopping strength
g = 20.0           # Bose-Hubbard interaction strength
τ_imag = 10         # Imaginary time propagation
ni_steps = 1000     # Number of steps for imaginary time propagation
t_prop = 300         # Time of propagation
n_steps = t_prop * 500  # Number of steps for real-time propagation
times = np.linspace(0, t_prop, n_steps)
# Set params for the potential
cooling_potential_width = 0.5
cooling_potential_offset = 0
# Set params for
n_ramps = 2                 # Define the number of ramps in the periodic potential
ramp_height = 1

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


V_trapping = cooling_potential(sites)   # 5 * (np.arange(L) - j0) ** 2
V_propagation = np.array([0, 0, 1, .8, .6, .4, 0.2, 0, 0, 0, 0, 0, 1, .8, .6, .4, .2, 0, 0, 0])  # ring_with_ramp(sites)
r_width = 5
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
tag = f'Ring_L{L}-T{t_prop}-J{J}-g{g:.2f}-ramp_width{r_width}'.replace('.', ',')
savesfolder = tag
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
    'ramp_height': ramp_height,
    'init_potential': V_trapping,
    'potential': V_propagation
}

with h5py.File(savespath+tag+'_data.hdf5', "w") as file:
    parameters_save = file.create_group('params')
    for _, __ in params.items():
        parameters_save.create_dataset(_, data=__)

########################################################################################################################
# Establish systems and evolve
########################################################################################################################

plt.figure(fnum)
fnum+=1
#plt.title('Potentials after Propagation')
plt.plot(degrees, V_propagation, '*-', label='Ring Potential')
plt.plot(degrees, V_trapping, '*-', label='Cooling Potential')
plt.xlim(degrees[0], degrees[-1])
plt.ylim(-0.01, 1.2*V_propagation.max())
plt.legend()
plt.tight_layout()
plt.savefig(savespath+tag+'_Potentials.pdf')
plt.savefig(savespath+tag+'_Potentials.png')

# Define the quantum systems for the Schrödinger Gross-Pitaevskii equations
init_qsys_gpe = quantum_system(J=J, g=g, V=deepcopy(V_trapping), ψ=np.ones(L, complex), is_open_boundary=False)
init_qsys_se = quantum_system(J=J, g=0, V=deepcopy(V_trapping), ψ=np.ones(L, complex), is_open_boundary=False)
img_propagator(τ_imag, ni_steps, init_qsys_gpe)
img_propagator(τ_imag, ni_steps, init_qsys_se)


"""plt.figure(fnum)
fnum+=1
plt.title(r"Initial state with")
plt.plot(np.abs(init_qsys_gpe.ψ) ** 2, label=r"$g = %0.2f$"%g)
plt.plot(np.abs(init_qsys_se.ψ) ** 2, label=r"$g = 0$")
plt.ylabel("probability")
plt.xlabel("site")
plt.legend()
plt.tight_layout()"""


# Set the propagation potential
qsys_gpe = quantum_system(J=J, g=g, V=V_propagation, ψ=deepcopy(init_qsys_gpe.ψ), is_open_boundary=False)
qsys_se = quantum_system(J=J, g=0, V=V_propagation, ψ=deepcopy(init_qsys_se.ψ), is_open_boundary=False)
gpe_wavefunction = propagator(times, qsys_gpe)
se_wavefunction = propagator(times, qsys_se)

gpe_current = get_current(gpe_wavefunction, qsys_gpe)
gpe_vel = get_moving_avg(gpe_current)
gpe_en = [get_energy(0, ψ, qsys_gpe) for ψ in gpe_wavefunction]
se_current = get_current(se_wavefunction, qsys_se)
se_vel = get_moving_avg(se_current)
se_en = [get_energy(0, ψ, qsys_se) for ψ in se_wavefunction]

print(f'Average current over propagation for \n Gpe: {gpe_current.mean()} \n Schrodinger: {se_current.mean()}')

with h5py.File(savespath+tag+'_data.hdf5', "a") as file:
    gpe_save = file.create_group('GPE')
    se_save = file.create_group('SE')
    gpe_save.create_dataset('ground_state', data=init_qsys_gpe.ψ)
    gpe_save.create_dataset('wavefunction', data=gpe_wavefunction)
    gpe_save.create_dataset('current', data=gpe_current)
    gpe_save.create_dataset('current_avgs', data=gpe_vel)
    se_save.create_dataset('ground_state', data=init_qsys_se.ψ)
    se_save.create_dataset('wavefunction', data=se_wavefunction)
    se_save.create_dataset('current', data=se_current)
    se_save.create_dataset('current_avgs', data=se_vel)
########################################################################################################################
# Plot the results
########################################################################################################################
"""plt.figure(fnum)
fnum+=1
plt.title('Potentials after Propagation')
plt.plot(degrees, V_propagation, label='Ring Declared')
plt.plot(degrees, qsys_gpe.V, '-.', label='Potential')
plt.plot(degrees, V_trapping, '--', label='Cooling Potential')
plt.xlim(degrees[0], degrees[-1])
plt.ylim(-0.01, 1.2*V_propagation.max())
plt.legend()
plt.tight_layout()"""

plt.figure(fnum, figsize=(6, 3))
fnum += 1
plt.title('Energy')
plt.plot(times,
    [get_energy(0, ψ, qsys_gpe) for ψ in gpe_wavefunction],
    label=r"$g=%0.2f$" % g
)
plt.tight_layout()
plt.savefig(savespath+tag+'_Energy.pdf')
plt.savefig(savespath+tag+'_Energy.png')

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
plt.savefig(savespath+tag+'_Density.pdf')
plt.savefig(savespath+tag+'_Density.png')

plt.figure(fnum)
fnum += 1
plt.plot(times, gpe_current, label=r'$g=%0.2f$'%g)
plt.plot(times, se_current, '--', label=r'$g=0$')
plt.xlabel('Time')
plt.ylabel('Current')
plt.legend()
plt.tight_layout()
plt.savefig(savespath+tag+'_Current.pdf')
plt.savefig(savespath+tag+'_Current.png')

plt.figure(fnum)
fnum+=1
plt.plot(times, gpe_vel, label=r'$g={}$'.format(g))
plt.plot(times, se_vel, '--', label=r'$g={}$'.format(0))
plt.xlabel('Time')
plt.ylabel('Cumulative Time Average')
plt.legend()
plt.tight_layout()
plt.savefig(savespath+tag+'_Current_Average.pdf')
plt.savefig(savespath+tag+'_Current_Average.png')

"""plt.figure(fnum)
fnum+=1
plt.plot(np.abs(test).max(axis=1))

plt.figure(fnum)
fnum+=1
plt.imshow(np.abs(test), aspect=18/2000)
plt.colorbar()
"""

#plt.show()
plt.close('all')
