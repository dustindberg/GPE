from numba import njit
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
import numpy as np
from scipy.constants import hbar, Boltzmann
from scipy.interpolate import UnivariateSpline
from split_op_gpe1D import SplitOpGPE1D, imag_time_gpe1D                                                                # class for the split operator propagation
import datetime
from datetime import date
import pytz
import pickle
from multiprocessing import Pool


Start_time = datetime.datetime.now(pytz.timezone('US/Central'))                                                         # Start timing for optimizing runs

########################################################################################################################
# Define the initial parameters for interaction and potential
########################################################################################################################

# Define physical parameters
a_0 = 5.291772109e-11                                                                                                   # Bohr Radius in meters

# Rubidium-87 properties
m = 1.4431609e-25                                                                                                       # Calculated mass of 87Rb in kg
a_s = 100 * a_0                                                                                                         # Background scattering length of 87Rb in meters

# Potassium-41 properties
# m= 6.80187119e-26                                                                                                      # Calculated mass of 41K in kg
# a_s = 65.42 * a_0                                                                                                      # Background scattering length of 41K in meters

# Experiment parameters
N = 1e4                                                                                                                 # Number of particles
omeg_x = 50 * 2 * np.pi                                                                                                 # Harmonic oscillation in the x-axis in Hz
omeg_y = 500 * 2 * np.pi                                                                                                # Harmonic oscillation in the y-axis in Hz
omeg_z = 500 * 2 * np.pi                                                                                                # Harmonic oscillation in the z-axis in Hz
omeg_cooling = 450 * 2 * np.pi                                                                                          # Harmonic oscillation for the trapping potential in Hz
scale = 1                                                                                                               # Scaling factor for the interaction parameter

# Parameters calculated by Python
# L_x = np.sqrt(hbar / (m * omeg_x))                                                                                      # Characteristic length in the x-direction in meters
L_x = 1.52295528474e-06                 # Value for testing against Patrick's data
L_y = np.sqrt(hbar / (m * omeg_y))                                                                                      # Characteristic length in the y-direction in meters
L_z = np.sqrt(hbar / (m * omeg_z))                                                                                      # Characteristic length in the z-direction in meters
g = 2 * N * L_x * m * scale * a_s * np.sqrt(omeg_y * omeg_z) / hbar                                                     # Dimensionless interaction parameter

# Conversion factors to plot in physical units
L_xmum = np.sqrt(hbar / (m * omeg_x)) * 1e6                                                                             # Characteristic length in the x-direction in meters
L_ymum = np.sqrt(hbar / (m * omeg_y)) * 1e6                                                                             # Characteristic length in the y-direction in meters
L_zmum = np.sqrt(hbar / (m * omeg_z)) * 1e6                                                                             # Characteristic length in the z-direction in meters
time_conv = 1. / omeg_x * 1e3                                                                                           # Converts characteristic time into milliseconds
energy_conv = hbar * omeg_x                                                                                             # Converts dimensionless energy terms to Joules
muK_conv = energy_conv * (1e6 / Boltzmann)                                                                              # Converts Joule terms to microKelvin
nK_conv = energy_conv * (1e9 / Boltzmann)                                                                               # Converts Joule terms to nanoKelvin
specvol_mum = (L_xmum * L_ymum * L_zmum) / N                                                                            # Converts dimensionless spacial terms into micrometers^3 per particle
dens_conv = 1. / (L_xmum * L_ymum * L_zmum)                                                                             # Calculated version of density unit conversion

# Parameters for computation
propagation_dt = 1e-4
height_asymmetric = 35                                                                                                  # Height parameter of asymmetric barrier
delta = 5                                                                                                               # Sharpness parameter of asymmetric barrier
# v_0 = 45.5                                                                                                              # Coefficient for the trapping potential
v_0 = 12.5                              # For testing Patrick's data
offset = 20.                                                                                                            # Center offset for cooling potential

today = date.today()                    # tag the date for unique file naming
filename = str(today.year) + str(today.month) + str(today.day) + "_" + str(Start_time.hour) + str(Start_time.minute)

# Functions for computation
@njit
def v(x, t=0.):
    """
    Potential energy
    """
    return 0.5 * x ** 2 + x ** 2 * height_asymmetric * np.exp(-(x / delta) ** 2) * (x < 0)
    #return 3000 - 2800 * np.exp(-((x + offset)/ 45) ** 2) - 2850 * np.exp(-((x - offset) / 47) ** 2) - 100 * np.exp(-((x - 5) / 10) ** 2)

@njit
def diff_v(x, t=0.):
    """
    the derivative of the potential energy for Ehrenfest theorem evaluation
    """
    return x + (2. * x - 2. * (1. / delta) ** 2 * x ** 3) * height_asymmetric * np.exp(-(x / delta) ** 2) * (x < 0)
    #return (224 / 81) * (x + offset) * np.exp(-((x + offset)/ 45) ** 2) + (5700 / 2209) * (x - offset) * np.exp(-((x - offset) / 47) ** 2) + 2 * (x - 5) * np.exp(-((x - 5) / 10) ** 2)

@njit
def diff_k(p, t=0.):
    """
    the derivative of the kinetic energy for Ehrenfest theorem evaluation
    """
    return p

@njit
def k(p, t=0.):
    """
    Non-relativistic kinetic energy
    """
    return 0.5 * p ** 2

@njit
def initial_trap(x, t=0):
    """
    Trapping potential to get the initial state
    :param x:
    :return:
    """
    return v_0 * (x + offset) ** 2

########################################################################################################################
# Declare parallel functions
########################################################################################################################

def run_single_case(params):
    """
    Find the initial state and then propagate
    :param params: dict with parameters for propagation
    :return: dict containing results
    """
    # get the initial state
    init_state, mu = imag_time_gpe1D(
        v=params['initial_trap'],
        g=g,
        dt=5e-5,
        epsilon=1e-9,
        **params
    )

    init_state, mu = imag_time_gpe1D(
        v=params['initial_trap'],
        g=g,
        init_wavefunction=init_state,
        dt=1e-6,
        epsilon=1e-11,
        **params
    )

    ####################################################################################################################
    # Propagate GPE equation
    ####################################################################################################################

    print("\nPropagate GPE equation")

    gpe_propagator = SplitOpGPE1D(
        v=v,
        g=g,
        dt=propagation_dt,
        **params
    ).set_wavefunction(init_state)

    # propagate till time T and for each time step save a probability density
    gpe_wavefunctions = [
        gpe_propagator.propagate(t).copy() for t in params['times']
    ]

    ####################################################################################################################
    # Propagate Schrodinger equation
    ####################################################################################################################

    print("\nPropagate Schrodinger equation")

    schrodinger_propagator = SplitOpGPE1D(
        v=v,
        g=0.,
        dt=propagation_dt,
        **params
    ).set_wavefunction(init_state)

    # Propagate till time T and for each time step save a probability density
    schrodinger_wavefunctions = [
        schrodinger_propagator.propagate(t).copy() for t in params['times']
    ]

    # bundle results into a dictionary
    return {
        'init_state': init_state,
        'mu': mu,

        # bundle separately GPE data
        'gpe': {
            'wavefunctions': gpe_wavefunctions,
            'extent': [gpe_propagator.x.min(), gpe_propagator.x.max(), 0., max(gpe_propagator.times)],
            'times': gpe_propagator.times,

            'x_average': gpe_propagator.x_average,
            'x_average_rhs': gpe_propagator.x_average_rhs,

            'p_average': gpe_propagator.p_average,
            'p_average_rhs': gpe_propagator.p_average_rhs,
            'hamiltonian_average': gpe_propagator.hamiltonian_average,

            'time_increments': gpe_propagator.time_increments,

            'dx': gpe_propagator.dx,
            'x': gpe_propagator.x,
        },

        # bundle separately Schrodinger data
        'schrodinger': {
            'wavefunctions': schrodinger_wavefunctions,
            'extent': [schrodinger_propagator.x.min(), schrodinger_propagator.x.max(), 0., max(schrodinger_propagator.times)],
            'times': schrodinger_propagator.times,

            'x_average': schrodinger_propagator.x_average,
            'x_average_rhs': schrodinger_propagator.x_average_rhs,

            'p_average': schrodinger_propagator.p_average,
            'p_average_rhs': schrodinger_propagator.p_average_rhs,
            'hamiltonian_average': schrodinger_propagator.hamiltonian_average,

            'time_increments': schrodinger_propagator.time_increments,
        },
    }

########################################################################################################################
# Serial code to launch parallel computations
########################################################################################################################

if __name__ == '__main__':

    fignum = 1                                                                                                          # Declare starting figure number
    # T = .5 * 2. * 2. * np.pi                                                                                            # Time duration for two periods
    T = 15.                             # For Testing Patrick's Data
    # times = np.linspace(0, T, 500)
    times = np.linspace(0, T, 1502)     # For testing Patrick's Data
    t_msplot = times * time_conv                                                                                        # Declare time with units of ms for plotting

    # save parameters as a separate bundle
    sys_params = dict(
        x_amplitude = 80.,                                                                                              # Set the range for calculation
        x_grid_dim = 32 * 1024,                                                                                          # For faster testing: 8*1024, more accuracy: 32*1024, best blend of speed and accuracy: 16x32
        N = N,
        k = k,
        initial_trap = initial_trap,
        diff_v = diff_v,
        diff_k = diff_k,
        times = times,
    )

    sys_params_flipped = sys_params.copy()                                                                              #copy to create parameters for the flipped case

    sys_params_flipped['initial_trap'] = njit(lambda x, t: initial_trap(-x, t))                                         #This is used to flip the initial trap about the offset

    ####################################################################################################################
    # Run calculations in parallel
    ####################################################################################################################

    # Get the unflip and flip simulations run in parallel;
    with Pool() as pool:
        qsys, qsys_flipped = pool.map(run_single_case, [sys_params, sys_params_flipped])
        # Results will be saved in qsys and qsys_flipped, respectively

    with open(filename + ".pickle", "wb") as f:
        pickle.dump([qsys, qsys_flipped], f)

    with open(filename + ".pickle", "rb") as f:
        qsys, qsys_flipped = pickle.load(f)

    ####################################################################################################################
    # Plot the potential in physical units before proceeding with simulation
    ####################################################################################################################

    dx = qsys['gpe']['dx']
    size = qsys['gpe']['x'].size
    x_cut = int(0.6 * size)                                                                                             #These are cuts such that we observe the behavior about the initial location of the wave
    x_cut_flipped = int(0.4 * size)

    @njit
    def v_muKelvin(v):
        """"
        The potential energy with corrected units microKelvin
        """
        return v * muK_conv

    figV = plt.figure(fignum, figsize=(8,6))
    fignum+=1
    plt.title('Potential')
    x = qsys['gpe']['x']
    x_mum = x * L_xmum
    v_muK = v(x) * muK_conv
    potential = v_muKelvin(v(x))
    plt.plot(x_mum, v_muK)

    plt.xlabel('$x$ ($\mu$m) ')
    plt.ylabel('$V(x)$ ($\mu$K)')
    plt.xlim([-80 * L_xmum, 80 * L_xmum])
    plt.savefig('Potential' + '.png')

    ####################################################################################################################
    # Generate plots to test the propagation
    ####################################################################################################################

    def analyze_propagation(qsys, title, fignum):
        """
        Make plots to check the quality of propagation
        :param qsys: dict with parameters
        :param title: str
        :param fignum: tracking figure number across plots
        :return: fignum
        """

        # Plot the density over time
        figdensplot = plt.figure(fignum, figsize=(8,6))
        fignum+=1
        plt.title(title)
        plot_title = title
        # plot the time dependent density
        extent = qsys['extent']
        plt.imshow(
            np.abs(qsys['wavefunctions']) ** 2,
            # some plotting parameters
            origin='lower',
            extent=extent,
            aspect=(extent[1] - extent[0]) / (extent[-1] - extent[-2]),
            norm=SymLogNorm(vmin=1e-13, vmax=1., linthresh=1e-15)
        )
        plt.xlabel('Coordinate $x$ (a.u.)')
        plt.ylabel('Time $t$ (a.u.)')
        plt.colorbar()
        plt.savefig(title + '.png')

        # Plot tests of the Ehrenfest theorems
        figefr = plt.figure(fignum, figsize=(24,6))
        fignum += 1
        times = qsys['times']
        t_ms = np.array(times) * time_conv
        plt.subplot(141)
        plt.title("Verify the first Ehrenfest theorem", pad = 15)
        # Calculate the derivative using the spline interpolation because times is not a linearly spaced array
        plt.plot(t_ms, UnivariateSpline(times, qsys['x_average'], s=0).derivative()(times),
                 '-r', label='$d\\langle\\hat{x}\\rangle / dt$')
        plt.plot(times, qsys['x_average_rhs'], '--b',label='$\\langle\\hat{p}\\rangle$')
        plt.legend()
        plt.ylabel('momentum')
        plt.xlabel('time $t$ (a.u.)')

        plt.subplot(142)
        plt.title("Verify the second Ehrenfest theorem", pad = 15)
        # Calculate the derivative using the spline interpolation because times is not a linearly spaced array
        plt.plot(times, UnivariateSpline(times, qsys['p_average'], s=0).derivative()(times),
                 '-r', label='$d\\langle\\hat{p}\\rangle / dt$')
        plt.plot(times, qsys['p_average_rhs'], '--b', label='$\\langle -U\'(\\hat{x})\\rangle$')
        plt.legend()
        plt.ylabel('force')
        plt.xlabel('time $t$ (a.u.)')

        plt.subplot(143)
        plt.title("The expectation value of the Hamiltonian", pad = 15)

        # Analyze how well the energy was preserved
        h = np.array(qsys['hamiltonian_average'])
        print(
            "\nHamiltonian is preserved within the accuracy of {:.1e} percent".format(
                100. * (1. - h.min() / h.max())
            )
        )
        print("Initial Energy {:.4e}".format(h[0]))

        plt.plot(t_ms, h * muK_conv)
        plt.ylabel('Energy ($\mu$K)')
        plt.xlabel('Time $t$ (ms)')

        plt.subplot(144)
        plt.title('Time Increments $dt$', pad = 15)
        plt.plot(qsys['time_increments'])
        plt.ylabel('$dt$')
        plt.xlabel('Time Step')
        figefr.suptitle(plot_title)
        plt.savefig('EFT_' + plot_title + '.png')

        return fignum

    ####################################################################################################################
    # Analyze the simulations
    ####################################################################################################################

    # Analyze the schrodinger propagation
    fignum = analyze_propagation(qsys['schrodinger'], "Schrodinger evolution", fignum)

    # Analyze the Flipped schrodinger propagation
    fignum = analyze_propagation(qsys_flipped['schrodinger'], "Flipped Schrodinger evolution", fignum)

    # Analyze the GPE propagation
    fignum = analyze_propagation(qsys['gpe'], "GPE evolution", fignum)

    # Analyze the Flipped GPE propagation
    fignum = analyze_propagation(qsys_flipped['gpe'], "Flipped GPE evolution", fignum)

    ########################################################################################################################
    # Calculate and plot the transmission probability
    ########################################################################################################################

    figTP = plt.figure(fignum,figsize=(18,6))
    fignum += 1
    plt.subplot(121)
    plt.plot(
        t_msplot,
        np.sum(np.abs(qsys['schrodinger']['wavefunctions'])[:, x_cut:] ** 2, axis=1) * dx,
        label='Schrodinger'
    )
    plt.plot(
        t_msplot,
        np.sum(np.abs(qsys_flipped['schrodinger']['wavefunctions'])[:, :x_cut_flipped] ** 2, axis=1) * dx,
        label='Flipped Schrodinger'
    )
    plt.legend()
    plt.xlabel('time $t$ (ms)')
    plt.ylabel("transmission probability")

    plt.subplot(122)
    plt.plot(
        t_msplot,
        np.sum(np.abs(qsys['gpe']['wavefunctions'])[:, x_cut:] ** 2, axis=1) * dx,
        label='GPE'
    )
    plt.plot(
        t_msplot,
        np.sum(np.abs(qsys_flipped['gpe']['wavefunctions'])[:, :x_cut_flipped] ** 2, axis=1) * dx,
        label='Flipped GPE'
    )
    plt.legend()
    plt.xlabel('Time $t$ (ms)')
    plt.ylabel("Transmission Probability")
    plt.savefig('Transmission Probability' + '.png')

    End_time = datetime.datetime.now(pytz.timezone('US/Central'))                                                       # Get current time to finish timing of program

    print ("Start time: {}:{}:{}".format(Start_time.hour,Start_time.minute,Start_time.second))                          # Print times for review
    print ("End time: {}:{}:{}".format(End_time.hour, End_time.minute, End_time.second))

    plt.show()                                                                                                          #generate all plots
