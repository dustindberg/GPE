from numba import njit
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
from scipy.constants import hbar
from scipy.interpolate import UnivariateSpline
from split_op_gpe1D import SplitOpGPE1D, imag_time_gpe1D    # class for the split operator propagation
import datetime
from datetime import date
import pytz
import pickle as pickle
from multiprocessing import Pool
import os

from tqdm import tqdm


# Start timing for optimizing runs
Start_time = datetime.datetime.now(pytz.timezone('US/Central'))

########################################################################################################################
# Define the initial parameters for interaction and potential
########################################################################################################################

# Define physical parameters
a_0 = 5.291772109e-11               # Bohr Radius in meters

# Rubidium-87 properties
m = 1.4431609e-25                   # Calculated mass of 87Rb in kg
a_s = 100 * a_0                     # Background scattering length of 87Rb in meters

# Potassium-41 properties
# m= 6.80187119e-26                  # Calculated mass of 41K in kg
# a_s = 65.42 * a_0                  # Background scattering length of 41K in meters

# Experiment parameters
N = 1e4                             # Number of particles
omeg_x = 50 * 2 * np.pi             # Harmonic oscillation in the x-axis in Hz
omeg_y = 500 * 2 * np.pi            # Harmonic oscillation in the y-axis in Hz
omeg_z = 500 * 2 * np.pi            # Harmonic oscillation in the z-axis in Hz
omeg_cooling = 450 * 2 * np.pi      # Harmonic oscillation for the trapping potential in Hz
scale = 1.0                           # Scaling factor for the interaction parameter

# Parameters calculated by Python
L_x = np.sqrt(hbar / (m * omeg_x))  # Characteristic length in the x-direction in meters
L_y = np.sqrt(hbar / (m * omeg_y))  # Characteristic length in the y-direction in meters
L_z = np.sqrt(hbar / (m * omeg_z))  # Characteristic length in the z-direction in meters
g = 2 * N * L_x * m * scale * a_s * np.sqrt(omeg_y * omeg_z) / hbar     # Dimensionless interaction parameter


def replace(str1):
    str1 = str1.replace('.', ',')
    return str1


########################################################################################################################
# Computational parameters which are needed for transfer to graphing file
########################################################################################################################
propagation_dt = 3e-5           # dt for adaptive step
eps = 5e-4                      # Error tolerance for adaptive step
height_asymmetric = 285         # Height parameter of asymmetric barrier
trap_height = 400.              # Used for trap height
sigma = 8.5                     # Width parameter for gaussian
delta = 2. * (sigma ** 2)       # Width parameter for realistic barrier
v_0 = 0.5                       # Coefficient for the trapping potential (was 10 for paper run)
cooling_offset = 38. # 37.0          # Center offset for cooling potential
prob_region = 0.7               # For calculating probability
prob_region_flipped = 0.3       # For calculating probability of the flipped case
T = 40.0                        # Total time
times = np.linspace(0, T, 500)  # Time grid
x_amplitude = 80.               # Set the range for calculation
x_grid_dim = 32 * 1024          # For faster testing: 8*1024, more accuracy: 32*1024, best blend: 16x32

# Create a tag using date and time to save and archive data
today = date.today()

filename = 'G_' + replace(str(g))
            # 'Trap_Height' + replace(str(height_asymmetric)) + '_Sigma' + replace(str(sigma)) + '_Vo' +\
            # replace(str(v_0)) + '_Offset' + replace(str(cooling_offset)) + '_T' + replace(str(T))
savesfolder = filename
parent_dir = "./Archive_Data"
path = os.path.join(parent_dir, savesfolder)
try:
    os.mkdir(path)
    print("Directory '%s' created" % savesfolder)
except:
    FileExistsError
    print('WARNING: The directory you are saving to already exists')

savespath = 'Archive_Data/' + str(savesfolder) + '/'



# Functions for computation
@njit(parallel=True)
def v(x):
    """
    Potential energy
    """
    return trap_height - height_asymmetric * (
        np.exp(-(x + 45.) ** 2 / delta)
        + np.exp(-(x + 30.) ** 2 / delta)
        + 0.85 * np.exp(-(x + 15.) ** 2 / delta)
        + 0.95 * np.exp(-(x - 0.3) ** 2 / delta)
        + 0.85 * np.exp(-(x - 15.) ** 2 / delta)
        + np.exp(-(x - 30.) ** 2 / delta)
        + np.exp(-(x - 45.) ** 2 / delta)
    )


@njit(parallel=True)
def diff_v(x):
    """
    the derivative of the potential energy for Ehrenfest theorem evaluation
    """
    return (2 * height_asymmetric / delta) * (
        (x + 45.) * np.exp(-(x + 45.) ** 2 / delta)
        + (x + 30.) * np.exp(-(x + 30.) ** 2 / delta)
        + (x + 15.) * 0.85 * np.exp(-(x + 15.) ** 2 / delta)
        + (x - 0.3) * 0.95 * np.exp(-(x - 0.3) ** 2 / delta)
        + (x - 15.) * 0.85 * np.exp(-(x - 15.) ** 2 / delta)
        + (x - 30.) * np.exp(-(x - 30.) ** 2 / delta)
        + (x - 45.) * np.exp(-(x - 45.) ** 2 / delta)
    )


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

    ####################################################################################################################
    # Propagate GPE equation
    ####################################################################################################################

    print("\nPropagate GPE equation")

    gpe_propagator = SplitOpGPE1D(
        v=v,
        g=g,
        dt=propagation_dt,
        epsilon=eps,
        **params
    ).set_wavefunction(init_state)

    # propagate till time T and for each time step save a probability density
    gpe_wavefunctions = [
        gpe_propagator.propagate(t).copy() for t in tqdm(params['times'])
    ]

    ####################################################################################################################
    # Propagate Schrodinger equation
    ####################################################################################################################

    print("\nPropagate Schrodinger equation")

    schrodinger_propagator = SplitOpGPE1D(
        v=v,
        g=0.,
        dt=propagation_dt,
        epsilon=eps,
        **params
    ).set_wavefunction(init_state)

    # Propagate till time T and for each time step save a probability density
    schrodinger_wavefunctions = [
        schrodinger_propagator.propagate(t).copy() for t in tqdm(params['times'])
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
            'extent': [schrodinger_propagator.x.min(), schrodinger_propagator.x.max(),
                       0., max(schrodinger_propagator.times)],
            'times': schrodinger_propagator.times,

            'x_average': schrodinger_propagator.x_average,
            'x_average_rhs': schrodinger_propagator.x_average_rhs,

            'p_average': schrodinger_propagator.p_average,
            'p_average_rhs': schrodinger_propagator.p_average_rhs,
            'hamiltonian_average': schrodinger_propagator.hamiltonian_average,

            'time_increments': schrodinger_propagator.time_increments,
        },

        # collect parameters for export
        'parameters': params
    }


########################################################################################################################
# Serial code to launch parallel computations
########################################################################################################################

if __name__ == '__main__':

    # Redeclare final parameters for dictionary because it breaks for some reason if this isn't here
    T = T
    times = times
    x_amplitude = x_amplitude
    x_grid_dim = x_grid_dim

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

    # copy to create parameters for the flipped case
    sys_params_flipped = sys_params.copy()
    # This is used to flip the initial trap about the offset
    sys_params_flipped['initial_trap'] = njit(lambda x: initial_trap(-x))

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
    fignum = 1                                          # Declare starting figure number
    pos_grid = qsys['gpe']['x']                         # Retrieve x grid from pickle file
    dx = qsys['gpe']['dx']                              # Retrieve dx spacing
    size = qsys['gpe']['x'].size                        # Create size parameter for probability evaluation
    x_cut = int(prob_region * size)                     # Use size parameter for Left-to-Right probability
    x_cut_flipped = int(prob_region_flipped * size)     # Use size parameter for Right-to-Left probability

    outfilename = filename + ".txt"
    outpath = os.path.join(savespath, outfilename)
    out = open(outpath, "w")
    out.write(
        "Initial Parameters: \n"
        + "trap_height = " + str(trap_height) + "\n"
        + "height_asymmetric = " + str(height_asymmetric) + "\n"
        + "v_0 = " + str(v_0) + "\n"
        + "cooling_offset = " + str(cooling_offset) + "\n"
        + "x_amplitude = " + str(x_amplitude) + "\n"
        + "x_grid_dim = " + str(x_grid_dim) + "\n"
        + "propagation_dt = " + str(propagation_dt) + "\n"
        + "Epsilon = " + str(eps) + "\n"
        + "Scale of g = " + str(scale) + "\n"
        + "Time run for = " + str(T) + "seconds \n"
        + "Probability Regions: " + str(prob_region) + ' and flipped: ' + str(prob_region_flipped) + "\n"
    )

    figV = plt.figure(fignum, figsize=(8, 6))
    fignum += 1
    plt.title('Potential')
    potential = v(pos_grid)
    plt.plot(pos_grid, potential, color='k')
    plt.hlines((qsys['schrodinger']['hamiltonian_average'][0]), pos_grid.min(), pos_grid.max(), colors='r')
    plt.fill_between(
        pos_grid[x_cut:],
        potential[x_cut:],
        potential.max(),
        facecolor="b",
        color='b',
        alpha=0.2
    )
    plt.fill_between(
        pos_grid[:x_cut_flipped],
        potential[:x_cut_flipped],
        potential.max(),
        facecolor="orange",
        color='orange',
        alpha=0.2
    )
    plt.xlabel('$x$ (au) ')
    plt.ylabel('$V(x)$ (au)')
    plt.xlim(-x_amplitude, x_amplitude)
    plt.savefig(savespath + 'Potential' + '.pdf')

    ####################################################################################################################
    # Generate plots to test the propagation
    ####################################################################################################################

    def analyze_propagation(qsys_dict, title, figure_number):
        """
        Make plots to check the quality of propagation
        :param qsys_dict: dict with parameters
        :param title: str
        :param figure_number: tracking figure number across plots
        :return: an updated figure number
        """

        # Plot the density over time
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
            norm=SymLogNorm(vmin=1e-7, vmax=1., linthresh=1e-15)
        )
        plt.xlabel('Coordinate $x$ (a.u.)')
        plt.ylabel('Time $t$ (a.u.)')
        plt.colorbar()
        plt.savefig(savespath + title + '.pdf')

        # Save the density for further testing
        density = np.abs(qsys_dict['wavefunctions']) ** 2
        np.save('Density_' + title, density)

        # Plot tests of the Ehrenfest theorems
        figefr = plt.figure(figure_number, figsize=(24, 6))
        figure_number += 1
        time = qsys_dict['times']
        plt.subplot(141)
        plt.title("Verify the first Ehrenfest theorem", pad=15)
        # Calculate the derivative using the spline interpolation because times is not a linearly spaced array
        plt.plot(time, UnivariateSpline(time, qsys_dict['x_average'], s=0).derivative()(time),
                 '-r', label='$d\\langle\\hat{x}\\rangle / dt$')
        plt.plot(time, qsys_dict['x_average_rhs'], '--b', label='$\\langle\\hat{p}\\rangle$')
        plt.legend(loc='lower right')
        plt.ylabel('momentum')
        plt.xlabel('time $t$ (a.u.)')

        plt.subplot(142)
        plt.title("Verify the second Ehrenfest theorem", pad=15)
        # Calculate the derivative using the spline interpolation because times is not a linearly spaced array
        plt.plot(time, UnivariateSpline(time, qsys_dict['p_average'], s=0).derivative()(time),
                 '-r', label='$d\\langle\\hat{p}\\rangle / dt$')
        plt.plot(time, qsys_dict['p_average_rhs'], '--b', label='$\\langle -U\'(\\hat{x})\\rangle$')
        plt.legend(loc='lower right')
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

        out.write("\n" + str(plot_title) + " Hamiltonian is preserved within the accuracy of {:.1e} percent".format(
                100. * (1. - h.min() / h.max())) + "\n"
                + str(plot_title) + " Initial Energy {:.4e}".format(h[0]) + "\n"
        )

        plt.plot(time, h)
        plt.ylabel('Energy (au)')
        plt.xlabel('Time $t$ (au)')

        plt.subplot(144)
        plt.title('Time Increments $dt$', pad=15)
        plt.plot(qsys_dict['time_increments'])
        plt.ylabel('$dt$')
        plt.xlabel('Time Step')
        figefr.suptitle(plot_title)
        plt.savefig(savespath + 'EFT_' + plot_title + '.pdf')

        return figure_number

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

    ####################################################################################################################
    # Calculate and plot the transmission probability
    ####################################################################################################################

    figTP = plt.figure(fignum, figsize=(18, 6))
    fignum += 1
    plt.subplot(121)
    plt.plot(
        times,
        np.sum(np.abs(qsys['schrodinger']['wavefunctions'])[:, x_cut:] ** 2, axis=1) * dx,
        label='Schrodinger'
    )
    plt.plot(
        times,
        np.sum(np.abs(qsys_flipped['schrodinger']['wavefunctions'])[:, :x_cut_flipped] ** 2, axis=1) * dx,
        label='Flipped Schrodinger'
    )
    plt.legend(loc='upper left')
    plt.xlabel('time $t$ (au)')
    plt.ylabel("transmission probability")

    plt.subplot(122)
    plt.plot(
        times,
        np.sum(np.abs(qsys['gpe']['wavefunctions'])[:, x_cut:] ** 2, axis=1) * dx,
        label='GPE'
    )
    plt.plot(
        times,
        np.sum(np.abs(qsys_flipped['gpe']['wavefunctions'])[:, :x_cut_flipped] ** 2, axis=1) * dx,
        label='Flipped GPE'
    )
    plt.legend(loc='upper left')
    plt.xlabel('Time $t$ (au)')
    plt.ylabel("Transmission Probability")
    plt.savefig(savespath + 'Transmission Probability' + '.pdf')

    # Get current time to finish timing of program
    End_time = datetime.datetime.now(pytz.timezone('US/Central'))

    # Print times for review
    print("Start time: {}:{}:{}".format(Start_time.hour, Start_time.minute, Start_time.second))
    print("End time: {}:{}:{}".format(End_time.hour, End_time.minute, End_time.second))

    print("Start time: {}:{}:{}".format(Start_time.hour, Start_time.minute, Start_time.second), file=out)
    print("End time: {}:{}:{}".format(End_time.hour, End_time.minute, End_time.second), file=out)

    # Close reference file and generate all plots
    out.close()
    plt.show()
