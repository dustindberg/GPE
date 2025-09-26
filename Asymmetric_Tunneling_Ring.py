from numba import njit
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
from scipy.constants import hbar
from scipy.interpolate import UnivariateSpline
from split_op_gpe1D import SplitOpGPE1D, imag_time_gpe1D            # class for the split operator propagation
import datetime
import pytz
import pickle as pickle
# from multiprocessing import Pool
import os


"""
Define function for propagation
"""


def replace(str1):
    """
    Used for creation of tagging system for better organization
    :param str1: A float with decimal places to be converted
    :return: the string with no '.' for better archive practices
    """
    str1 = str1.replace('.', ',')
    return str1


def get_gs(params):
    """
    Find the initial state and then propagate
    :param params: dict with parameters for propagation
    :return: dict containing results
    """
    # get the initial state
    init_state, mu = imag_time_gpe1D(
        v=params['initial_trap'],
        g=g,
        dt=1e-3,
        epsilon=1e-8,
        **params
    )

    init_state, mu = imag_time_gpe1D(
        v=params['initial_trap'],
        g=g,
        init_wavefunction=init_state,
        dt=1e-5,
        epsilon=1e-10,
        **params
    )

    return {
        'init_state': init_state,
        'mu': mu,
        'parameters': params
    }


def propagate_looping_gpe(params):
    """
    Propagates the GPE separately so that final states can be tracked without issue while looping
    """

    print("\nPropagate GPE equation")

    gpe_propagator = SplitOpGPE1D(
        v=v,
        g=g,
        dt=propagation_dt,
        epsilon=eps,
        **params
    ).set_wavefunction(params['init_state'])

    # propagate till time T and for each time step save a probability density
    gpe_wavefunctions = [
        gpe_propagator.propagate(t).copy() for t in params['times']
    ]

    # bundle results into a dictionary
    return {
        'init_state': params['init_state'],
        'mu': params['mu'],

        'wavefunctions': gpe_wavefunctions,
        'extent': [gpe_propagator.x.min(), gpe_propagator.x.max(), min(gpe_propagator.times), max(gpe_propagator.times)],
        'times': gpe_propagator.times,

        'x_average': gpe_propagator.x_average,
        'x_average_rhs': gpe_propagator.x_average_rhs,

        'p_average': gpe_propagator.p_average,
        'p_average_rhs': gpe_propagator.p_average_rhs,
        'hamiltonian_average': gpe_propagator.hamiltonian_average,

        'time_increments': gpe_propagator.time_increments,

        'dx': gpe_propagator.dx,
        'x': gpe_propagator.x,

        't': gpe_propagator.t,

        'parameters': params
    }


def propagate_looping_schrodinger(params):
    """
    Propagates the Schrodinger separately so that final states can be tracked without issue while looping
    """

    print("\nPropagate Schrodinger equation")

    schrodinger_propagator = SplitOpGPE1D(
        v=v,
        g=0.,
        dt=propagation_dt,
        epsilon=eps,
        **params
    ).set_wavefunction(params['init_state'])

    # Propagate till time T and for each time step save a probability density
    schrodinger_wavefunctions = [
        schrodinger_propagator.propagate(t).copy() for t in params['times']
    ]

    return {
        'init_state': params['init_state'],
        'mu': params['mu'],

        'wavefunctions': schrodinger_wavefunctions,
        'extent': [schrodinger_propagator.x.min(), schrodinger_propagator.x.max(), min(schrodinger_propagator.times),
                   max(schrodinger_propagator.times)],
        'times': schrodinger_propagator.times,

        'x_average': schrodinger_propagator.x_average,
        'x_average_rhs': schrodinger_propagator.x_average_rhs,

        'p_average': schrodinger_propagator.p_average,
        'p_average_rhs': schrodinger_propagator.p_average_rhs,
        'hamiltonian_average': schrodinger_propagator.hamiltonian_average,

        'time_increments': schrodinger_propagator.time_increments,

        't': schrodinger_propagator.t,

        'parameters': params
    }


def analyze_propagation(qsys_dict, title, figure_number):
    """
    Make plots to check the quality of propagation
    :param qsys_dict: dict with parameters
    :param title: str
    :param figure_number: tracking figure number across plots
    :return: an updated figure number
    """

    """ Plot the density over time """
    plt.figure(figure_number, figsize=(8, 6))
    figure_number += 1
    plt.title(title)
    plot_title = title
    # plot the time dependent density
    extent = qsys_dict['extent']
    plt.imshow(
        np.abs(qsys_dict['wavefunctions']) ** 2,
        # some plotting parameters
        origin='lower',
        extent=extent,
        aspect=(extent[1] - extent[0]) / (extent[-1] - extent[-2]),
        norm=SymLogNorm(vmin=1e-15, vmax=1., linthresh=1e-15)
    )
    plt.xlabel('Coordinate $x$ (a.u.)')
    plt.ylabel('Time $t$ (a.u.)')
    plt.colorbar()
    plt.savefig(savespath + title + '.pdf')

    # Save the density for further testing
    density = np.abs(qsys_dict['wavefunctions']) ** 2
    np.save(savespath + 'Density_' + title, density)

    # Plot tests of the Ehrenfest theorems
    figefr = plt.figure(figure_number, figsize=(24, 6))
    figure_number += 1
    time = qsys_dict['times']
    plt.subplot(141)
    plt.title("Verify the first Ehrenfest theorem", pad=15)
    # Calculate the derivative using the spline interpolation because times is not a linearly spaced array
    #plt.plot(time, UnivariateSpline(time, qsys_dict['x_average'], s=0).derivative()(time),
    #         '-r', label='$d\\langle\\hat{x}\\rangle / dt$')
    plt.plot(time, qsys_dict['x_average_rhs'], '--b', label='$\\langle\\hat{p}\\rangle$')
    plt.legend(loc='lower left')
    plt.ylabel('momentum')
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(142)
    plt.title("Verify the second Ehrenfest theorem", pad=15)
    # Calculate the derivative using the spline interpolation because times is not a linearly spaced array
    #plt.plot(time, UnivariateSpline(time, qsys_dict['p_average'], s=0).derivative()(time),
    #         '-r', label='$d\\langle\\hat{p}\\rangle / dt$')
    plt.plot(time, qsys_dict['p_average_rhs'], '--b', label='$\\langle -U\'(\\hat{x})\\rangle$')
    plt.legend(loc='lower left')
    plt.ylabel('force')
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(143)
    plt.title("The expectation value of the Hamiltonian", pad=15)

    # Analyze how well the energy was preserved
    h = np.array(qsys_dict['hamiltonian_average'])
    print(
        "\nHamiltonian is preserved within the accuracy of {:.1e} percent".format(
            100. * (1. - h.min() / h.max())
        )
    )
    print("Initial Energy {:.4e}".format(h[0]))

    plt.plot(time, h)
    plt.ylabel('Energy (au)')
    plt.xlabel('Time $t$ (au)')

    plt.subplot(144)
    plt.title('Time Increments $dt$', pad=15)
    plt.plot(qsys_dict['time_increments'])
    plt.ylabel('$dt$')
    plt.xlabel('Time Step')
    figefr.suptitle(plot_title)
    plt.savefig(savespath + 'EFT_' + plot_title + '.png')

    return figure_number


"""
Functions for computation
"""


#@njit(parallel=True)
def v(x):
    """
    Potential energy
    """
    """return height_asymmetric * (
        np.exp(-(x + 71.) ** 2 / delta)
        + 0.55 * np.exp(-(x + 79.) ** 2 / delta)
        + np.exp(-(x + 21.) ** 2 / delta)
        + 0.55 * np.exp(-(x + 29.) ** 2 / delta)
        + np.exp(-(x - 29.) ** 2 / delta)
        + 0.55 * np.exp(-(x - 21.) ** 2 / delta)
        + np.exp(-(x - 79.) ** 2 / delta)
        + 0.55 * np.exp(-(x - 71.) ** 2 / delta)
    )"""
#    return height_asymmetric * (
#        np.exp(-(x + 58.) ** 2 / delta)
#        + 0.55 * np.exp(-(x + 42.) ** 2 / delta)
#        + np.exp(-(x - 42.) ** 2 / delta)
#        + 0.55 * np.exp(-(x - 58.) ** 2 / delta)
#    )
    return 0


#@njit(parallel=True)
def diff_v(x):
    """
    the derivative of the potential energy for Ehrenfest theorem evaluation
    """
    """return (-2 * height_asymmetric / delta) * (
        + (x + 71.) * np.exp(-(x + 71.) ** 2 / delta)
        + (x + 79.) * 0.55 * np.exp(-(x + 79.) ** 2 / delta)
        + (x + 21.) * np.exp(-(x + 21.) ** 2 / delta)
        + (x + 29.) * 0.55 * np.exp(-(x + 29.) ** 2 / delta)
        + (x - 29.) * np.exp(-(x - 29.) ** 2 / delta)
        + (x - 21.) * 0.55 * np.exp(-(x - 21.) ** 2 / delta)
        + (x - 79.) * np.exp(-(x - 79.) ** 2 / delta)
        + (x - 71.) * 0.55 * np.exp(-(x - 71.) ** 2 / delta)
    )"""
#    return (-2 * height_asymmetric / delta) * (
#        + (x + 58.) * np.exp(-(x + 58.) ** 2 / delta)
#        + (x + 42.) * 0.55 * np.exp(-(x + 42.) ** 2 / delta)
#        + (x - 42.) * np.exp(-(x - 42.) ** 2 / delta)
#        + (x - 58.) * 0.55 * np.exp(-(x - 58.) ** 2 / delta)
#    )
    return 0


@njit
def diff_k(p):
    """
    the derivative of the kinetic energy for Ehrenfest theorem evaluation
    """
    return p


@njit
def k(p):
    """
    Non-relativistic kinetic energy
    """
    return 0.5 * p ** 2


@njit
def initial_trap(x):
    """
    Trapping potential to get the initial state
    :param x:
    :return:
    """
    return v_0 * (x + cooling_offset) ** 2


# Start timing for optimizing runs
Start_time = datetime.datetime.now(pytz.timezone('US/Central'))

########################################################################################################################
# IMPORTANT: Are you continuing a previous run or starting a new one?
########################################################################################################################

Previous_run = False    # True if importing from previous run, false if starting fresh

# Paste the folder name you are loading from
previous_file = "Ring_Sigma8,5_Height45,0_Vo0,5"
T_prev_start = 100.  # The starting time of the previous run
T_prev_end = 110.    # Ending time of the previous run

"""
Define the initial parameters for interaction and potential
"""

# Define physical parameters
a_0 = 5.291772109e-11                           # Bohr Radius in meters

# Rubidium-87 properties
m = 1.4431609e-25                               # Calculated mass of 87Rb in kg
a_s = 100 * a_0                                  # Background scattering length of 87Rb in meters

# Experiment parameters
N = 1e4                                         # Number of particles
omeg_x = 50 * 2 * np.pi                         # Harmonic oscillation in the x-axis in Hz
omeg_y = 500 * 2 * np.pi                        # Harmonic oscillation in the y-axis in Hz
omeg_z = 500 * 2 * np.pi                        # Harmonic oscillation in the z-axis in Hz
omeg_cooling = 450 * 2 * np.pi                  # Harmonic oscillation for the trapping potential in Hz
scale = 1.0                                     # Scaling factor for the interaction parameter

# Parameters calculated by Python
L_x = np.sqrt(hbar / (m * omeg_x))              # Characteristic length in the x-direction in meters
L_y = np.sqrt(hbar / (m * omeg_y))              # Characteristic length in the y-direction in meters
L_z = np.sqrt(hbar / (m * omeg_z))              # Characteristic length in the z-direction in meters

# Dimensionless interaction parameter
g = 2 * N * L_x * m * scale * a_s * np.sqrt(omeg_y * omeg_z) / hbar

"""
Parameters needed for transfer to graphing file
"""

kick = 0.                      # initial momentum kick
propagation_dt = 3e-3           # dt for adaptive step
eps = 5e-4                      # Error tolerance for adaptive step
height_asymmetric = 0.         # Height parameter of asymmetric barrier
sigma = 4.25 #8.5                     # For estimating realistic barrier configurations
delta = 2. * (sigma ** 2)       # Width parameter for realistic barrier
v_0 = 0.25                       # Coefficient for the trapping potential
cooling_offset = 0.             # Center offset for cooling potential
prob_region = 0.64              # For calculating probability
prob_region_flipped = 0.36      # For calculating probability of the flipped case
T_start = 0.0                   # Keep track of the starting time
T_interval = 10.0               # Time interval for updated looping
T_end = T_start + T_interval    # Get the ending time

x_amplitude = 100.              # Set the range for calculation
x_grid_dim = 32 * 1024          # For faster testing: 8*1024, more accuracy: 32*1024, best blend: 16x32

########################################################################################################################
# Create a tag using date and time to save and archive data
########################################################################################################################

parent_dir = "/home/dustin/PycharmProjects/GPE/Archive_Data/Ring_runs/"

if Previous_run is False:
    times = np.linspace(T_start, T_end, 500)  # Time grid

    savesfolder = 'Ring_NoPotential'
        #'Ring' + '_Sigma' + replace(str(sigma)) \
        #+ '_Height' + replace(str(height_asymmetric)) + '_Vo' + replace(str(v_0))
    path = os.path.join(parent_dir, savesfolder)
    os.mkdir(path)
    print("Directory '%s' created" % savesfolder)
elif Previous_run is True:
    times = np.linspace(T_prev_end, T_prev_end + T_interval, 500)
    print("Continuing Run from T=" + replace(str(T_prev_end)))
    savesfolder = previous_file
else:
    print('ERROR: Please specify if you are starting a new run')
    exit()
savespath = 'Archive_Data/Ring_runs/' + str(savesfolder) + '/'


"""
Save final dictionary of parameters
"""


# save parameters as a separate bundle
sys_params = dict(
    x_amplitude=x_amplitude,
    x_grid_dim=x_grid_dim,
    N=N,
    k=k,
    initial_trap=initial_trap,
    diff_v=diff_v,
    diff_k=diff_k,
    times=times,
)


""" 
Get the ground state if starting fresh OR load the final state of a previous run
"""
if Previous_run is False:
    ground_state = get_gs(sys_params)
    with open(savespath + savesfolder + "_GroundState.pickle", "wb") as f:
        pickle.dump(ground_state, f)
    with open(savespath + savesfolder + "_GroundState.pickle", "rb") as f:
        ground_state = pickle.load(f)

elif Previous_run is True:
    with open(savespath + savesfolder + "Schrodinger_Range_" + replace(str(T_prev_start)) + "to"
              + replace(str(T_prev_end)) + ".pickle", "rb") as f:
        looped_propagation_schr = pickle.load(f)
    with open(savespath + savesfolder + "GPE_Range_" + replace(str(T_prev_start)) + "to"
              + replace(str(T_prev_end)) + ".pickle", "rb") as f:
        looped_propagation_gpe = pickle.load(f)

else:
    print('ERROR: Please specify if you are starting a new run')
    exit()

"""
Collect Params for beginning looping code
"""

params_lib = {
    'gpe': sys_params.copy(),
    'schrodinger': sys_params.copy()
}

if Previous_run is False:
    params_lib['gpe']['init_state'] = ground_state['init_state']
    params_lib['gpe']['mu'] = ground_state['mu']
    params_lib['schrodinger']['init_state'] = ground_state['init_state']
    params_lib['schrodinger']['mu'] = ground_state['mu']

    params_lib['t'] = T_start
    params_lib['gpe']['t'] = T_start
    params_lib['schrodinger']['t'] = T_start

elif Previous_run is True:
    params_lib['gpe'].update(looped_propagation_gpe)
    params_lib['schrodinger'].update(looped_propagation_schr)

    params_lib['gpe']['init_state'] = looped_propagation_gpe['wavefunctions'][-1][:]
    params_lib['gpe']['mu'] = looped_propagation_gpe['mu']
    params_lib['gpe']['times'] = times
    params_lib['gpe']['t'] = T_prev_end
    params_lib['gpe']['initial_trap'] = v
    params_lib['schrodinger']['init_state'] = looped_propagation_schr['wavefunctions'][-1][:]
    params_lib['schrodinger']['mu'] = looped_propagation_schr['mu']
    params_lib['schrodinger']['times'] = times
    params_lib['schrodinger']['t'] = T_prev_end
    params_lib['schrodinger']['initial_trap'] = v

else:
    print('ERROR: Please specify if you are starting a new run')
    exit()


continue_loop = True
fignum = 0


"""
Run a single case looping until Keyboard interrupt
"""
while continue_loop is True:
    try:
        # with Pool() as pool:
        #     looped_propagation_gpe, looped_propagation_schr = pool.map(run_parallel, ['gpe','schrodinger'])
        print('Starting run with starting time t = ' + str(params_lib['schrodinger']['t']))
        looped_propagation_gpe = propagate_looping_gpe(params_lib['gpe'])
        looped_propagation_schr = propagate_looping_schrodinger(params_lib['schrodinger'])

        with open(savespath + savesfolder + "GPE_Range_" + replace(str(T_start)) + "to"
                  + replace(str(T_end)) + ".pickle", "wb") as f:
            pickle.dump(looped_propagation_gpe, f)

        with open(savespath + savesfolder + "GPE_Range_" + replace(str(T_start)) + "to"
                  + replace(str(T_end)) + ".pickle", "rb") as f:
            looped_propagation_gpe = pickle.load(f)

        with open(savespath + savesfolder + "Schrodinger_Range_" + replace(str(T_start)) + "to"
                  + replace(str(T_end)) + ".pickle", "wb") as f:
            pickle.dump(looped_propagation_schr, f)

        with open(savespath + savesfolder + "Schrodinger_Range_" + replace(str(T_start)) + "to"
                  + replace(str(T_end)) + ".pickle", "rb") as f:
            looped_propagation_schr = pickle.load(f)

        fignum = analyze_propagation(looped_propagation_gpe, 'GPE' + replace(str(T_end)), fignum)
        fignum = analyze_propagation(looped_propagation_schr, 'Schrodinger' + replace(str(T_end)), fignum)

        """ Begin redefining initial parameters for """
        params_lib['gpe'].update(looped_propagation_gpe)
        params_lib['schrodinger'].update(looped_propagation_schr)
        T_start += T_interval
        T_end = T_start + T_interval    # Get the ending time
        times = np.linspace(T_start, T_end, 500)  # New time grid
        params_lib['times'] = times
        params_lib['t'] = times[0]
        params_lib['gpe']['init_state'] = looped_propagation_gpe['wavefunctions'][-1][:]
        params_lib['gpe']['mu'] = looped_propagation_gpe['mu']
        params_lib['gpe']['times'] = times
        params_lib['gpe']['t'] = times[0]
        params_lib['gpe']['parameters']['t'] = T_start
        params_lib['gpe']['initial_trap'] = v
        params_lib['schrodinger']['init_state'] = looped_propagation_schr['wavefunctions'][-1][:]
        params_lib['schrodinger']['mu'] = looped_propagation_schr['mu']
        params_lib['schrodinger']['times'] = times
        params_lib['schrodinger']['t'] = times[0]
        params_lib['schrodinger']['parameters']['t'] = T_start
        params_lib['schrodinger']['initial_trap'] = v
        print('Attempting next run with a starting time of t = ' + str(params_lib['schrodinger']['t']))
    except KeyboardInterrupt:
        continue_loop = False

plt.show()
