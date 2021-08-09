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
import _pickle as pickle
from multiprocessing import Pool
import os


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
scale = 1.0                                                                                                               # Scaling factor for the interaction parameter

# Parameters calculated by Python
L_x = np.sqrt(hbar / (m * omeg_x))                                                                                      # Characteristic length in the x-direction in meters
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
propagation_dt = 3e-5  #1e-5
eps = 5e-4
height_asymmetric = 1250.                                                                                                 # Height parameter of asymmetric barrier
trap_height = 2000.
sigma = 8.5
delta = 2. * (sigma ** 2)
v_0 = 10.0                                           # Coefficient for the trapping potential
fwhm = 2. * sigma*np.sqrt(2*np.log(2))
peak_offset = fwhm    # 0.5*FWHM = sigma*sqrt(2log(2))
cooling_offset = 43.5                                # Center offset for cooling potential


# Create a tag using date and time to save and archive data
today = date.today()
def Replace(str1):
    str1 = str1.replace('.', ',')
    return str1
today = date.today()
filename = 'TEST3_Trap_TrapHeight' + Replace(str(trap_height)) + '_Sigma' + Replace(str(sigma)) + '_Height' \
           + Replace(str(height_asymmetric)) + '_Vo' + Replace(str(v_0)) + '_Offset'+Replace(str(cooling_offset))
savesfolder = filename
parent_dir = "/home/skref/PycharmProjects/GPE/Archive_Data"
path = os.path.join(parent_dir, savesfolder)
os.mkdir(path)
savespath = 'Archive_Data/' + str(savesfolder) + '/'

print("Directory '%s' created" % savesfolder)


# Functions for computation
@njit(parallel=True)
def v(x):
    """
    Potential energy
    """
    return trap_height - height_asymmetric * (
        np.exp(-(x + 2.1 * fwhm) ** 2 / delta)
        + np.exp(-(x - 2.1 * fwhm) ** 2 / delta)
        + np.exp(-(x - 1.5 * fwhm) ** 2 / delta)
        + 0.80 * np.exp(-(x - 0.667 * fwhm) ** 2 / delta)
        + 0.75 * np.exp(-x ** 2 / delta)
        + 0.65 * np.exp(-(x + 0.667 * fwhm) ** 2 / delta)
        + np.exp(-(x + 1.5 * fwhm) ** 2 / delta)
    )
    # return trap_height - height_asymmetric * (
    #     0.325 * np.exp(-(x - 2.0 * peak_offset) ** 2 / delta)
    #     + 0.55 * np.exp(-(x - 1.0 * peak_offset) ** 2 / delta)
    #     + 0.60 * np.exp(-x ** 2 / delta)
    #     + 0.46 * np.exp(-(x + 1.0 * peak_offset) ** 2 / delta)
    #     + 0.21 * np.exp(-(x + 2.0 * peak_offset) ** 2 / delta)
    #     + np.exp(-(x - 5.8 * peak_offset) ** 2 / (4 * delta))
    #     + np.exp(-(x - 3.3 * peak_offset) ** 2 / (4 * delta))
    #     + 1.000082 * np.exp(-(x + 3.3 * peak_offset) ** 2 / (4 * delta))
    #     + 1.000082 * np.exp(-(x + 5.8 * peak_offset) ** 2 / (4 * delta))
    # )
#    return 0.5 * x ** 2 + height_asymmetric * (
#        0.04 * np.exp(-(x+9.2) ** 2 / 25)
#        + np.exp(-(x+4) ** 2 / 25)
#        + 0.6 * np.exp(-(x-1.5) ** 2 / 25)
#        + 0.2 * np.exp(-(x-7.415) ** 2 / 25)
#    )
#    return height_asymmetric * (
#        np.exp(-((x + 3. * peak_offset) / delta) ** 2)
#        + (2. / 3.) * np.exp(-((x + peak_offset) / delta) ** 2)
#        + 0.5 * np.exp(-((x - peak_offset) / delta) ** 2)
#        + 0.5 * np.exp(-((x - 3. * peak_offset) / delta) ** 2)
#        )
    #return 0.5 * x ** 2 + x ** 2 * height_asymmetric * np.exp(-(x / delta) ** 2) * (x < 0)


@njit(parallel=True)
def diff_v(x):
    """
    the derivative of the potential energy for Ehrenfest theorem evaluation
    """
    return (2 * height_asymmetric / delta) * (
        (x + 2.1 * fwhm) * np.exp(-(x + 2.1 * fwhm) ** 2 / delta)
        + (x - 2.1 * fwhm) * np.exp(-(x - 2.1 * fwhm) ** 2 / delta)
        + (x - 1.5 * fwhm) * np.exp(-(x - 1.5 * fwhm) ** 2 / delta)
        + (x - 0.667 * fwhm) * 0.80 * np.exp(-(x - 0.667 * fwhm) ** 2 / delta)
        + 0.75 * np.exp(-x ** 2 / delta)
        + (x + 0.667 * fwhm) * 0.65 * np.exp(-(x + 0.667 * fwhm) ** 2 / delta)
        + (x + 1.5 * fwhm) * np.exp(-(x + 1.5 * fwhm) ** 2 / delta)
    )
    # return (2 * height_asymmetric / delta) * (
    #     (x - 2.0 * peak_offset) * 0.325 * np.exp(-(x - 2.0 * peak_offset) ** 2 / delta)
    #     + (x - 1.0 * peak_offset) * 0.55 * np.exp(-(x - 1.0 * peak_offset) ** 2 / delta)
    #     + x * 0.60 * np.exp(-x ** 2 / delta)
    #     + (x + 1.0 * peak_offset) * 0.46 * np.exp(-(x + 1.0 * peak_offset) ** 2 / delta)
    #     + (x + 2.0 * peak_offset) * 0.21 * np.exp(-(x + 2.0 * peak_offset) ** 2 / delta)
    #     + (1. / 4.) * (x - 5.8 * peak_offset) * np.exp(-(x - 5.8 * peak_offset) ** 2 / (4 * delta))
    #     + (1. / 4.) * (x - 3.3 * peak_offset) * np.exp(-(x - 3.3 * peak_offset) ** 2 / (4 * delta))
    #     + (1. / 4.) * (x + 3.3 * peak_offset) * 1.000082 * np.exp(-(x + 3.3 * peak_offset) ** 2 / (4 * delta))
    #     + (1. / 4.) * (x + 5.8 * peak_offset) * 1.000082 * np.exp(-(x + 5.8 * peak_offset) ** 2 / (4 * delta))
    # )
#    return x + (-2 * height_asymmetric / 25) *(
#        (x+9.2) * 0.04 * np.exp(-(x+9.2) ** 2 / 25)
#        + (x+4) * np.exp(-(x+4) ** 2 / 25)
#        + (x-1.5) * 0.6 * np.exp(-(x-1.5) ** 2 / 25)
#        + (x-7.415) * 0.2 * np.exp(-(x-7.415) ** 2 / 25)
#    )
#    return (-2 * height_asymmetric / delta ** 2) * (
#            (x + 3 * peak_offset) * np.exp(-((x + 3 * peak_offset) / delta) ** 2)
#            + (x + peak_offset) * (2. / 3.) * np.exp(-((x + peak_offset) / delta) ** 2)
#            + (x - peak_offset) * 0.5 * np.exp(-((x - peak_offset) / delta) ** 2)
#            + (x - 3 * peak_offset) * 0.5 * np.exp(-((x - 3 * peak_offset) / delta) ** 2)
#        )
    # return x + (2. * x - 2. * (1. / delta) ** 2 * x ** 3) * height_asymmetric * np.exp(-(x / delta) ** 2) * (x < 0)


@njit
def diff_k(p):
    """
    the derivative of the kinetic energy for Ehrenfest theorem evaluation
    """
    return p

@njit(parallel=True)
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
        dt=1e-3,
        epsilon=1e-9,
        **params
    )

    init_state, mu = imag_time_gpe1D(
        v=params['initial_trap'],
        g=g,
        init_wavefunction=init_state,
        dt=1e-4,
        epsilon=1e-10,
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
        epsilon = eps,
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
        epsilon = eps,
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

    # Declare final parameters for dictionary
    # T = .5 * 2. * 2. * np.pi                                                                                          # Time length of two periods
    T = 4.0                                                                                                         # Time length of 1 period
    times = np.linspace(0, T, 500)
    x_amplitude = 75.                                                                                                  # Set the range for calculation
    x_grid_dim = 16 * 1024                                                                                              # For faster testing: 8*1024, more accuracy: 32*1024, best blend of speed and accuracy: 16x32

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

    sys_params_flipped = sys_params.copy()                                                                              #copy to create parameters for the flipped case

    sys_params_flipped['initial_trap'] = njit(lambda x: initial_trap(-x))                                         #This is used to flip the initial trap about the offset

    ####################################################################################################################
    # Run calculations in parallel
    ####################################################################################################################

    # Get the unflip and flip simulations run in parallel;
    with Pool() as pool:
        qsys, qsys_flipped = pool.map(run_single_case, [sys_params, sys_params_flipped])
        # Results will be saved in qsys and qsys_flipped, respectively

    with open(savespath + filename + ".pickle", "wb") as f:
        pickle.dump([qsys, qsys_flipped], f)

    with open(savespath + filename + ".pickle", "rb") as f:
        qsys, qsys_flipped = pickle.load(f)

    ####################################################################################################################
    # Plot the potential in physical units before proceeding with simulation
    ####################################################################################################################

    # Declare plotting specific terms
    fignum = 1                                                                                                          # Declare starting figure number
    t_msplot = times * time_conv                                                                                        # Declare time with units of ms for plotting
    dx = qsys['gpe']['dx']
    size = qsys['gpe']['x'].size
    x_cut = int(0.72 * size) # int(0.735 * size)
    x_cut_flipped = int(0.28 * size) # int(0.265 * size)
    # x_cut = int(0.65 * size)                                                                                          # These are cuts such that we observe the behavior about the initial location of the wave
    # x_cut_flipped = int(0.35 * size)

    outfilename = filename + ".txt"
    outpath = os.path.join(savespath, outfilename)
    out = open(outpath, "w")
    out.write(
        "Initial Parameters: \n"
        + "height_asymmetric = " + str(height_asymmetric) + "\n"
        + "v_0 = " + str(v_0) + "\n"
        + "cooling_offset = " + str(cooling_offset) + "\n"
        + "x_amplitude = " + str(x_amplitude) + "\n"
        + "x_grid_dim = " + str(x_grid_dim) + "\n"
        + "Propagation dt = " + str(propagation_dt) + "\n"
        + "Epsilon = " + str(eps) + "\n"
        + "Scale of g = " + str(scale) + "\n"
        + "Time run for = " + str(T) + "seconds \n"
    )



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
    plt.plot(x_mum, v_muK, color='k')
    plt.hlines((qsys['gpe']['hamiltonian_average'][0])*muK_conv, x_mum.min(), x_mum.max(),colors = 'r')
    plt.fill_between(
        x_mum[x_cut:],
        potential[x_cut:],
        potential.max(),
        facecolor="b",
             color='b',
          alpha=0.2
    )
    plt.fill_between(
        x_mum[:x_cut_flipped],
        potential[:x_cut_flipped],
        potential.max(),
        facecolor="orange",
             color='orange',
          alpha=0.2
    )
    plt.xlabel('$x$ ($\mu$m) ')
    plt.ylabel('$V(x)$ ($\mu$K)')
    plt.xlim(-x_amplitude * L_xmum, x_amplitude * L_xmum)
    plt.savefig(savespath + 'Potential' + '.png')

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
            norm=SymLogNorm(vmin=1e-7, vmax=1., linthresh=1e-15)
        )
        plt.xlabel('Coordinate $x$ (a.u.)')
        plt.ylabel('Time $t$ (a.u.)')
        plt.colorbar()
        plt.savefig(savespath + title + '.png')

        #save the density for further testing
        density = np.abs(qsys['wavefunctions']) ** 2
        np.save('Density_' + title, density)

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
        plt.plot(t_ms, qsys['x_average_rhs'], '--b', label='$\\langle\\hat{p}\\rangle$')
        plt.legend(loc='lower right')
        plt.ylabel('momentum')
        plt.xlabel('time $t$ (a.u.)')

        plt.subplot(142)
        plt.title("Verify the second Ehrenfest theorem", pad = 15)
        # Calculate the derivative using the spline interpolation because times is not a linearly spaced array
        plt.plot(t_ms, UnivariateSpline(times, qsys['p_average'], s=0).derivative()(times),
                 '-r', label='$d\\langle\\hat{p}\\rangle / dt$')
        plt.plot(t_ms, qsys['p_average_rhs'], '--b', label='$\\langle -U\'(\\hat{x})\\rangle$')
        plt.legend(loc='lower right')
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

        out.write("\n" + str(plot_title) + " Hamiltonian is preserved within the accuracy of {:.1e} percent".format(
                100. * (1. - h.min() / h.max())) + "\n"
                + str(plot_title) + " Initial Energy {:.4e}".format(h[0]) + "\n"
        )

        plt.plot(t_ms, h * muK_conv)
        plt.ylabel('Energy ($\mu$K)')
        plt.xlabel('Time $t$ (ms)')

        plt.subplot(144)
        plt.title('Time Increments $dt$', pad = 15)
        plt.plot(qsys['time_increments'])
        plt.ylabel('$dt$')
        plt.xlabel('Time Step')
        figefr.suptitle(plot_title)
        plt.savefig(savespath + 'EFT_' + plot_title + '.png')

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
    plt.legend(loc='upper left')
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
    plt.legend(loc='upper left')
    plt.xlabel('Time $t$ (ms)')
    plt.ylabel("Transmission Probability")
    plt.savefig(savespath + 'Transmission Probability' + '.png')

    End_time = datetime.datetime.now(pytz.timezone('US/Central'))                                                       # Get current time to finish timing of program

    print ("Start time: {}:{}:{}".format(Start_time.hour,Start_time.minute,Start_time.second))                          # Print times for review
    print ("End time: {}:{}:{}".format(End_time.hour, End_time.minute, End_time.second))

    print("Start time: {}:{}:{}".format(Start_time.hour,Start_time.minute,Start_time.second), file=out)
    print("End time: {}:{}:{}".format(End_time.hour, End_time.minute, End_time.second), file=out)

    out.close()
    plt.show()                                                                                                          # Generate all plots
