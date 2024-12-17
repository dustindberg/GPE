from collections import namedtuple
import numpy as np
import scipy.ndimage
from scipy.integrate import ode
from scipy.signal.windows import blackman
from numba import jit, njit
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
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
    ).set_integrator('zvode', atol=1E-12, rtol=1E-12)

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


def get_cumulative_avg(x):
    return np.cumsum(x) / np.arange(1, x.size + 1)
    #return scipy.ndimage.uniform_filter1d(x, N, mode='nearest')


########################################################################################################################
# Define Model Parameters
# Using scipy's ODE solver, we will be solving the Gross-Pitaevskii equation (GPE)
# i d/dt ψ_j(t) = -J[ψ_{j-1}(t) + ψ_{j+1}(t)] + V(x) ψ_j(t) + g|ψ_j(t)|² ψ_j(t)
########################################################################################################################
# Physical System Parameters
L = 36              # Number of sites
J = 1            # hopping strength
obc = True #False
# g = 0.00 * J        # Bose-Hubbard interaction strength
g_list = np.linspace(0, 5, 6)
τ_imag = 25        # Imaginary time propagation
ni_steps = τ_imag * L ** 2        # Number of steps for imaginary time propagation
t_prop = 5       # Time of propagation
n_steps = t_prop * L ** 2    # Number of steps for real-time propagation
times = np.linspace(0, t_prop, n_steps)
# Set params for the potential
cooling_potential_width = 0.5
cooling_potential_offset = 0 # -int(L/2 - 7)

# Set params for
n_ramps = 1                 # Define the number of ramps in the periodic potential
h = 2.0 * J
ϵ = 1e-7

# calculate center of chain
if L % 2 == 0:
    j0 = L // 2 - 0.5           # center of chain
else:
    j0 = L // 2               # center of chain
sites = np.arange(L) - j0
degrees = np.arange(L) #np.array(sites) * 180 / sites[-1]


def cooling_potential(j):
    v_cooling = cooling_potential_width * (j - cooling_potential_offset) ** 2
    return v_cooling - v_cooling.min()


ramp_width = int(L/(n_ramps ** 2))   # Give the length of the ramp potential in degrees

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
            ring[i] = h * ((j[i] - ramp_end[_]) / (ramp_start[_] - ramp_end[_]))
    return ring



V_trapping = np.hstack((np.zeros(12),10*h*np.ones(L-12))) #cooling_potential(np.arange(L))
#np.hstack((h*np.ones(4), np.linspace(h, 0, 9), np.zeros(7), h * np.ones(12)))#
#cooling_potential(sites)   # 5 * (np.arange(L) - j0) ** 2
w = int(L/3)
left_edge = 0
right_edge = 12 #int(L/(n_ramps * 2)) - left_edge
shift = 0
V_propagation = np.flip(np.hstack((np.zeros(12), np.linspace(h, 0, 13), np.zeros(11))))
#np.hstack((np.zeros(4), np.linspace(h, 0, 9), np.zeros(7), np.linspace(h, 0, 9), np.zeros(3))))
# np.hstack((np.zeros(int(L/3)), np.linspace(h, 0, int(L/6)), np.array(0),
                #           np.linspace(h, (h / int(L/6 - 1)), int(L/6)-1), np.zeros(int(L/3))))

"""np.roll(np.hstack((np.zeros(w),
                                   np.linspace(h/(left_edge+1), (left_edge*h)/(left_edge+1), left_edge),
                                   np.linspace(h, (h/right_edge), right_edge), np.zeros(w),
                                   np.linspace(h/(left_edge+1), (left_edge*h)/(left_edge+1), left_edge),
                                   np.linspace(h, (h/right_edge), right_edge))), -(int(w/2) + shift))"""

# np.array([0, 0, h/3, 2*h/3])
# np.zeros(w), np.linspace(h, (h/(w+1)), w), np.zeros(w), np.linspace(h, (h/(w+1)), w)
# V_propagation = np.roll(np.linspace(w, 0, L), shift) * (h / w)  # ring_with_ramp(sites)

assert n_ramps * w <= L,\
    (f"A system of grid size {L} cannot accommodate {n_ramps} barriers of width {w},\n"
     f"as the total width of all barriers {n_ramps * w}!")
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
    'figure.dpi': 150,
    'legend.fontsize': 8,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'axes.prop_cycle': plt.cycler('color', (ok[_] for _ in ok)),
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'lines.linewidth': 2,
}
plt.rcParams.update(plt_params)


extent = [degrees[0], degrees[-1], times[0], times[-1]]
aspect = (degrees[-1] - degrees[0]) / (times[-1] - times[0])

#chosen_gs_index = 100
#g_save = str(g_list[chosen_gs_index])
parent_dir = "./Archive_Data/ODE_Solver/"
#parent_dir += f'SameGS-g={g_list[chosen_gs_index]}_'.replace('.', ',')
parent_dir += f'Looped_SingleBarrier_'
parent_dir += f'L{L}-J{J:.1f}-t{t_prop}-g_range{g_list[1]:.2f}to{g_list[-1]:.2f}-obc{obc}'.replace(
    '.', ',')
parent_dir += f'-{n_ramps}Ramp-W{w}-REdge{right_edge}-H{h}'.replace('.', ',')
#parent_dir += f'-V0{cooling_potential_width}at{cooling_potential_offset}'.replace('.', ',')
#parent_dir +='-SYMMETRIC'
parent_dir += '-FLIPPED'

if shift != 0:
    parent_dir += f'-ShiftedBy{shift}'
try:
    os.mkdir(parent_dir)
    print(f'Parent Directory created, saving to: {parent_dir}')
except:
    FileExistsError
    print(f'CAREFUL! The parent directory exists. \nResults will overwrite everything in {parent_dir}\n')


fnum = 0
plt.figure(fnum)
fnum += 1
#plt.title('Potentials after Propagation')
plt.plot(degrees, V_propagation, '*-', label='Prop Potential')
plt.plot(degrees, V_trapping, '-.', label='Cooling Potential')
plt.xlim(degrees[0], degrees[-1])
plt.ylim(-0.01, 1.2*V_propagation.max())
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig(parent_dir + '/Potentials.pdf')
#plt.savefig(parent_dir+'/Potentials.png')
plt.show()

# Quickly get the Schrodinger ground state so that it only has to be run once
init_qsys_se = quantum_system(J=J, g=0, V=deepcopy(V_trapping), ψ=np.ones(L, complex), is_open_boundary=obc)
img_propagator(τ_imag, ni_steps, init_qsys_se)

# If running with single ground state, uncomment this.
#init_qsys_gpe = quantum_system(J=J, g=chosen_gs_index, V=deepcopy(V_trapping), ψ=np.ones(L, complex), is_open_boundary=obc)
#img_propagator(τ_imag, ni_steps, init_qsys_gpe)

results_dict = dict()
print('Beginning simulations. Progress is tracked with TQDM')
for g in tqdm(g_list):
    #print(f'Beginning simulation with g={g:.2f}')
    # Save file architecture
    tag = f'Ring_L{L}-T{t_prop}-J{J}-g{g:.2f}-ramp_width{w}-ramp_height{h}'.replace('.', ',')
    savesfolder = tag
    path = os.path.join(parent_dir, savesfolder)
    savespath = parent_dir + '/' + str(savesfolder) + '/'

    try:
        os.mkdir(path)
        #print(f'Simulation Directory "{savesfolder}" created')
    except:
        FileExistsError
        #print('WARNING: The directory you are saving to already exists!!! \nYou may be overwriting previous data (; n;)\n')

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
        'ramp_height': h,
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



    # Define the quantum systems for the Schrödinger Gross-Pitaevskii equations
    init_qsys_gpe = quantum_system(J=J, g=g, V=deepcopy(V_trapping), ψ=np.ones(L, complex), is_open_boundary=obc)
    img_propagator(τ_imag, ni_steps, init_qsys_gpe)


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
    qsys_gpe = quantum_system(J=J, g=g, V=V_propagation, ψ=deepcopy(init_qsys_se.ψ), is_open_boundary=obc)
    qsys_se = quantum_system(J=J, g=0, V=V_propagation, ψ=deepcopy(init_qsys_gpe.ψ), is_open_boundary=obc)
    gpe_wavefunction = propagator(times, qsys_gpe)
    se_wavefunction = propagator(times, qsys_se)
    gpe_current = get_current(gpe_wavefunction, qsys_gpe)
    gpe_vel = get_cumulative_avg(gpe_current)
    gpe_en = np.array([get_energy(0, ψ, qsys_gpe) for ψ in gpe_wavefunction])
    se_current = get_current(se_wavefunction, qsys_se)
    se_vel = get_cumulative_avg(se_current)
    se_en = np.array([get_energy(0, ψ, qsys_se) for ψ in se_wavefunction])

    err = np.abs(np.real(gpe_en).max() - np.real(gpe_en).min())
    if err > ϵ:
        print(f'\nWARNING! The error for run g={g:.2f} was {err:.3e}, which is greater than your declared tolerance {ϵ:.1e}')

    if np.real(gpe_en).max() > h:
        print(f'\nATTENTION! For g={g}, the BEC energy, E={np.real(gpe_en).max()}, is above the barrier height, {h}.')

    with h5py.File(savespath+tag+'_data.hdf5', "a") as file:
        gpe_save = file.create_group('GPE')
        se_save = file.create_group('SE')
        gpe_save.create_dataset('ground_state', data=init_qsys_gpe.ψ)
        gpe_save.create_dataset('wavefunction', data=gpe_wavefunction)
        gpe_save.create_dataset('current', data=gpe_current)
        gpe_save.create_dataset('current_avgs', data=gpe_vel)
        gpe_save.create_dataset('energy', data=gpe_en)
        se_save.create_dataset('ground_state', data=init_qsys_se.ψ)
        se_save.create_dataset('wavefunction', data=se_wavefunction)
        se_save.create_dataset('current', data=se_current)
        se_save.create_dataset('current_avgs', data=se_vel)
        se_save.create_dataset('energy', data=se_en)

        results_dict[f'{g:.2f}'] = {
            'params': params,
            'GPE': {
                'ground_state': init_qsys_gpe.ψ,
                'current': np.real(gpe_current),
                'current_avgs': np.real(gpe_vel),
                'energy': np.real(gpe_en)
            },
            'SE': {
                'ground_state': init_qsys_se.ψ,
                'current': np.real(se_current),
                'current_avgs': np.real(se_vel),
                'energy': np.real(se_en)
            }
        }


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
        np.real(gpe_en),
        label=r"$g=%0.2f$" % g
    )
    plt.tight_layout()
    plt.savefig(savespath+tag+'_Energy.pdf')
    #plt.savefig(savespath+tag+'_Energy.png')

    plt.figure(fnum, figsize=(6, 3))
    fnum += 1
    plt.subplot(121)
    plt.title("$g=%0.2f$" % g)
    plt.imshow(
        np.abs(gpe_wavefunction) ** 2,
        extent=extent,
        aspect=aspect,
        origin='lower',
        interpolation='none',   # 'nearest',
        norm=SymLogNorm(vmin=1e-7, vmax=1., linthresh=1e-9),
    )
    plt.xticks(np.arange(degrees[0], degrees[-1], step=60))
    plt.ylabel("time")

    plt.subplot(122)
    plt.title(r"$g = 0$")
    plt.imshow(
        np.abs(se_wavefunction) ** 2,
        extent=extent,
        aspect=aspect,
        origin='lower',
        interpolation='none',   # 'nearest',
        norm=SymLogNorm(vmin=1e-7, vmax=1., linthresh=1e-9),
    )
    plt.xticks(np.arange(degrees[0], degrees[-1], step=60))
    plt.ylabel("time")
    plt.colorbar()
    #plt.savefig(savespath+tag+'_Density.pdf')
    plt.savefig(savespath+tag+'_Density.png')

    plt.figure(fnum)
    fnum += 1
    plt.plot(times, np.real(gpe_current), label=r'$g=%0.2f$'%g)
    plt.plot(times, np.real(se_current), '--', label=r'$g=0$')
    plt.xlabel('Time')
    plt.ylabel('Current')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(savespath+tag+'_Current.pdf')
    #plt.savefig(savespath+tag+'_Current.png')

    plt.figure(fnum)
    fnum+=1
    plt.plot(times, np.real(gpe_vel), label=r'$g={}$'.format(g))
    plt.plot(times, np.real(se_vel), '--', label=r'$g={}$'.format(0))
    plt.xlabel('Time')
    plt.ylabel('Cumulative Time Average')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(savespath+tag+'_Current_Average.pdf')
    #plt.savefig(savespath+tag+'_Current_Average.png')

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
    fnum = 1


schro_current_avg = results_dict['0.00']['GPE']['current_avgs']
half = int(0.5 * len(schro_current_avg))
throw = [max(schro_current_avg[half:]), min(schro_current_avg[half:])]

final_current_avgs = np.empty(len(results_dict))
thrown_current_avgs = np.empty(len(results_dict))
current_upper_bound = np.empty(len(results_dict))
current_lower_bound = np.empty(len(results_dict))
current_avgs_upper_thrown = np.empty(len(results_dict))
current_avgs_lower_thrown = np.empty(len(results_dict))
avg_energy_tracker = np.empty(len(results_dict))
order_tracker = np.empty(len(results_dict))
keys = list(results_dict.keys())
for _ in range(len(results_dict)):
    thrown_data = get_cumulative_avg(results_dict[keys[_]]['GPE']['current'][half:])
    # thrown_data *= blackman(len(thrown_data))
    final_current_avgs[_] = results_dict[keys[_]]['GPE']['current_avgs'][-1]
    thrown_current_avgs[_] = thrown_data[-1]
    current_upper_bound[_] = max(results_dict[keys[_]]['GPE']['current_avgs'][half:])
    current_lower_bound[_] = min(results_dict[keys[_]]['GPE']['current_avgs'][half:])
    current_avgs_upper_thrown[_] = max(thrown_data)
    current_avgs_lower_thrown[_] = min(thrown_data)
    avg_energy_tracker[_] = np.array(results_dict[keys[_]]['GPE']['energy']).mean()
    order_tracker[_] = keys[_]


plt.figure(fnum, figsize=(16, 4))
fnum += 1
plt.fill_between(g_list, current_upper_bound, current_lower_bound, color=ok['blue'], alpha=0.7,
                 label='Oscillation Range')
plt.plot(g_list, final_current_avgs, color=ok['navy'], label='Final Current')
plt.xlim(g_list[1], g_list[-1])
plt.hlines(throw[0], g_list[0], g_list[-1], linestyles='--', color=ok['red'], label='Baseline Oscillations')
plt.hlines(throw[-1], g_list[0], g_list[-1], linestyles='--', color=ok['red'])
plt.hlines((throw[0] + throw[-1])/2, g_list[0], g_list[-1], linestyles='--', color=ok['orange'], label='"Zero" point')
plt.xlabel('$g$')
plt.ylabel('Average Current')
plt.legend(loc='upper left')
plt.tight_layout()

plt.savefig(parent_dir+'/CurrentVSg.pdf')
#plt.savefig(parent_dir+'/CurrentVSg.png')

plt.figure(fnum, figsize=(16, 4))
fnum += 1
plt.fill_between(g_list, current_avgs_upper_thrown, current_avgs_lower_thrown, color=ok['blue'], alpha=0.7,
                 label='Current Avgs Oscillation Range after half time')
plt.plot(g_list, thrown_current_avgs,
         color=ok['navy'], label='Final Current Avg after half time')
plt.xlim(g_list[1], g_list[-1])
plt.hlines(throw[0], g_list[0], g_list[-1], linestyles='--', color=ok['red'], label='Baseline Oscillations')
plt.hlines(throw[-1], g_list[0], g_list[-1], linestyles='--', color=ok['red'])
plt.hlines((throw[0] + throw[-1])/2, g_list[0], g_list[-1], linestyles='--', color=ok['orange'], label='"Zero" point')
plt.xlabel('$g$')
plt.ylabel('Thrown Cumulative Average Current')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig(parent_dir+'/CurrentVSg_ThrownHalf.pdf')

plt.figure(fnum, figsize=(16, 4))
fnum += 1
plt.plot(g_list, avg_energy_tracker)
plt.xlim(g_list[1], g_list[-1])
plt.xlabel('$g$')
plt.ylabel('Average Energy')
plt.tight_layout()
#plt.savefig(parent_dir+'/AvgEnergy.pdf')
plt.savefig(parent_dir+'/AvgEnergy.png')

osc_avg = (current_upper_bound + current_lower_bound) * 0.5
plt.figure(fnum, figsize=(16, 4))
fnum += 1
plt.fill_between(g_list, current_upper_bound, current_lower_bound, color=ok['blue'], alpha=0.7,
                 label='Oscillation Range')
plt.plot(g_list, osc_avg, color=ok['navy'], label='Average Current Oscillations')
plt.xlim(g_list[1], g_list[-1])
plt.hlines(throw[0], g_list[0], g_list[-1], linestyles='--', color=ok['red'], label='Baseline Oscillations')
plt.hlines(throw[-1], g_list[0], g_list[-1], linestyles='--', color=ok['red'])
plt.hlines((throw[0] + throw[-1])/2, g_list[0], g_list[-1], linestyles='--', color=ok['orange'], label='"Zero" point')
plt.xlabel('$g$')
plt.ylabel('Average Current')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(parent_dir+'/CurrentOscillationsVSg.pdf')


plt.close('all')

