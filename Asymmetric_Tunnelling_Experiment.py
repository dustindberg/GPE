import utlt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import SymLogNorm

from scipy.integrate import simps
from scipy.interpolate import interp1d, UnivariateSpline, InterpolatedUnivariateSpline

import copy
from split_op_gpe1D import SplitOpGPE1D, imag_time_gpe1D    # class for the split operator propagation
from numba import njit
import pickle as pickle

from tqdm.notebook import tqdm
from multiprocessing import Pool


from sklearn import linear_model

import glob
import os




ok = utlt.standardize_plots(plt.rcParams)
sys = utlt.BEC(omega_x=100*np.pi)

########################################################################################################################
# Determine global functions
########################################################################################################################


@njit(parallel=True)
def pulse(pos_grid, width, center):
    """
    Adjustable width Gaussian. Passed as a function for repeated use and readability
    :param pos_grid:
    :param height:
    :param center:
    :param width:
    :return:
    """
    return  np.exp(-((pos_grid - center) / width) ** 2)


@njit(parallel=True)
def diff_pulse(pos_grid, width, center):
    """
    Derivative of the
    :param pos_grid:
    :param height:
    :param center:
    :param width:
    :return:
    """
    return -2 * (pos_grid - center) / (width ** 2) * np.exp(-((pos_grid - center) / width) ** 2)


@njit(parallel=True)
def k(p):
    """
    Non-relativistic kinetic energy
    """
    return 0.5 * p ** 2


@njit
def diff_k(p):
    """
    the derivative of the kinetic energy for Ehrenfest theorem evaluation
    """
    return p


# Dimensionless System Params
waist = sys.dimless_x(20, -6)
σ = 0.5 * waist
fwhm = 2 * σ * np.sqrt(2 * np.log(2))
z_lim = int(3 * fwhm)
z_res = 2 ** 12
eps = 1e-5

T = 10
times = np.linspace(0, T, 500)  # Time grid: required a resolution of 500 for split operator class
prop_dt = 1e-7                  # Initial guess for the adaptive step propagation

kick = 10
V_0 = sys.dimless_energy(0.75, -6)

g = sys.g
dz = 2 * z_lim / z_res
z = np.arange(-z_lim, z_lim, dz)

named_params = {
    'T': T,
    'V_0': round(V_0, 2),
    'sigma': round(σ, 2),
}

all_params = copy.deepcopy(named_params)
file_path, file_name = utlt.build_saves_tag(params_list=named_params,
                                            unique_identifier=f'BEC_AsymmetricTunneling',
                                            parent_directory='Archive_Data/AsymmetricTunneling')


# Potential Parameters
@njit(parallel=True)
def V(x):
    p1 = np.exp(-((x + 2 * σ) / σ) ** 2)
    p2 = 0.75 * np.exp(-((x - 0.05 * σ) / σ) ** 2)
    p3 = np.exp(-((x - 2 * σ) / σ) ** 2)
    return V_0 * (1 - (p1 + p2 + p3))


@njit(parallel=True)
def dif_V(x):
    p1 = -2 * (x + 2 * σ) / (σ ** 2) * np.exp(-((x + 2 * σ) / σ) ** 2)
    p2 = -1.5 * (x - 0.05 * σ) / (σ ** 2) * np.exp(-((x - 0.05 * σ) / σ) ** 2)
    p3 = -2 * (x - 2 * σ) / (σ ** 2) * np.exp(-((x - 2 * σ) / σ) ** 2)
    return -V_0 * (p1 + p2 + p3)


@njit
def init_V(x):
    return V_0 * (1 - pulse(x, σ, -2*σ))


fignum = 0
fignum += 1
plt.figure(fignum, layout='constrained')
plt.plot(z, V(z))
plt.plot(z, init_V(z))
plt.xlim(z[0], z[-1])

prob_region = 0.63                                  # For calculating probability
prob_region_flipped = 1 - prob_region               # For calculating probability of the flipped case
size = len(z)                     # Create size parameter for probability evaluation
x_cut = int(prob_region * size)                     # Use size parameter for Left-to-Right probability
x_cut_flipped = int(prob_region_flipped * size)     # Use size parameter for Right-to-Left probability

plt.figure(fignum, layout='constrained')
fignum += 1
potential = V(z)
plt.plot(z, potential, color='k')
#plt.hlines((qsys['gpe']['hamiltonian_average'][0]), z.min(), z.max(), colors='r')
plt.fill_between(
    z[x_cut:],
    potential[x_cut:],
    potential.max(),
    facecolor=ok['blue'],
    color=ok['blue'],
    alpha=0.5
)
plt.fill_between(
    z[:x_cut_flipped],
    potential[:x_cut_flipped],
    potential.max(),
    facecolor=ok["orange"],
    color=ok['orange'],
    alpha=0.5
)
plt.xlabel('$x$ (au) ')
plt.ylabel('$V(x)$ (au)')
plt.xlim(-z_lim, z_lim)
plt.show()


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
        v=params['V'],
        g=g,
        dt=prop_dt,
        epsilon=eps,
        **params
    )
    gpe_propagator.set_wavefunction(
        init_state * np.exp(1j * params['init_momentum_kick'] * gpe_propagator.x)
    )

    # propagate till time T and for each time step save a probability density
    gpe_wavefunctions = [
        gpe_propagator.propagate(t).copy() for t in params['times']
    ]

    ####################################################################################################################
    # Propagate Schrodinger equation
    ####################################################################################################################

    print("\nPropagate Schrodinger equation")

    schrodinger_propagator = SplitOpGPE1D(
        v=params['V'],
        g=0.,
        dt=prop_dt,
        epsilon=eps,
        **params
    )
    schrodinger_propagator.set_wavefunction(
        init_state * np.exp(1j * params['init_momentum_kick'] * schrodinger_propagator.x)
    )

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



if __name__ == '__main__':
    # Redeclare final parameters for dictionary because it breaks if this isn't here
    T = T
    times = times
    x_amplitude = z_lim
    x_grid_dim = z_res

    # save parameters as a separate bundle
    sys_params = dict(
        x_amplitude=x_amplitude,
        x_grid_dim=x_grid_dim,
        N=sys.N,
        k=k,
        initial_trap=init_V,
        V=V,
        diff_v=dif_V,
        diff_k=diff_k,
        times=times,
        init_momentum_kick=kick,
    )

    sys_params_flipped = sys_params.copy()  # Copy to create parameters for the flipped case

    # This is used to flip the initial trap about the offset and then init_momentum_kick it
    sys_params_flipped['initial_trap'] = njit(lambda x: init_V(-x))
    sys_params_flipped['init_momentum_kick'] = -sys_params_flipped['init_momentum_kick']
    ####################################################################################################################
    # Run calculations in parallel
    ####################################################################################################################

    # Get the unflip and flip simulations run in parallel;
    with Pool(processes=4) as pool:
        qsys, qsys_flipped = pool.map(run_single_case, [sys_params, sys_params_flipped])
    # Results will be saved in qsys and qsys_flipped, respectively

    with open(file_path + file_name + ".pickle", "wb") as f:
        pickle.dump([qsys, qsys_flipped], f)

    with open(file_path + file_name + ".pickle", "rb") as f:
        qsys, qsys_flipped = pickle.load(f)


    def analyze_propagation(qsys_dict, title, figure_number):
        """
        Make plots to check the quality of propagation
        :param qsys_dict: dict with parameters
        :param title: str
        :param figure_number: tracking figure number across plots
        :return: an updated figure number
        """

        # Plot the density over time
        plt.figure(figure_number, layout='constrained')
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
            norm=SymLogNorm(vmin=1e-7, vmax=1., linthresh=1e-7)
        )
        plt.xlabel('Coordinate $x$ (a.u.)')
        plt.ylabel('Time $t$ (a.u.)')
        plt.colorbar()
        plt.savefig(file_path + title + '.pdf')

        # Save the density for further testing
        density = np.abs(qsys_dict['wavefunctions']) ** 2
        np.save('Density_' + title, density)

        # Plot tests of the Ehrenfest theorems
        plt.figure(figure_number, figsize=(24, 6), layout='constrained')
        figure_number += 1
        time = qsys_dict['times']
        plt.subplot(141)
        plt.title("Verify the first Ehrenfest theorem", pad=15)
        # Calculate the derivative using the spline interpolation because times is not a linearly spaced array
        plt.plot(time, UnivariateSpline(time, qsys_dict['x_average'], s=0).derivative()(time),
                 '-r', label='$d\\langle\\hat{x}\\rangle / dt$')
        plt.plot(time, qsys_dict['x_average_rhs'], '--b', label='$\\langle\\hat{p}\\rangle$')
        plt.legend(loc='lower left')
        plt.ylabel('momentum')
        plt.xlabel('time $t$ (a.u.)')

        plt.subplot(142)
        plt.title("Verify the second Ehrenfest theorem", pad=15)
        # Calculate the derivative using the spline interpolation because times is not a linearly spaced array
        plt.plot(time, UnivariateSpline(time, qsys_dict['p_average'], s=0).derivative()(time),
                 '-r', label='$d\\langle\\hat{p}\\rangle / dt$')
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
        plt.suptitle(plot_title)
        plt.savefig(file_path + 'EFT_' + plot_title + '.pdf')

        return figure_number


    # Analyze the GPE propagation
    fignum = analyze_propagation(qsys['gpe'], "GPE evolution", fignum)

    # Analyze the Flipped GPE propagation
    fignum = analyze_propagation(qsys_flipped['gpe'], "Flipped GPE evolution", fignum)

    # Also get schrodinger
    fignum = analyze_propagation(qsys['schrodinger'], "SE evolution", fignum)

    # Analyze the Flipped GPE propagation
    fignum = analyze_propagation(qsys_flipped['schrodinger'], "Flipped SE evolution", fignum)


    plt.figure(fignum, figsize=(8, 3), layout='constrained')
    fignum += 1
    plt.subplot(121)
    plt.plot(
        times,
        np.sum(np.abs(qsys['schrodinger']['wavefunctions'])[:, x_cut:] ** 2, axis=1) * dz,
        label='Schrodinger'
    )
    plt.plot(
        times,
        np.sum(np.abs(qsys_flipped['schrodinger']['wavefunctions'])[:, :x_cut_flipped] ** 2, axis=1) * dz,
        label='Flipped Schrodinger'
    )
    plt.legend(loc='upper left')
    plt.xlabel('time $t$ (au)')
    plt.ylabel("transmission probability")

    plt.subplot(122)
    plt.plot(
        times,
        np.sum(np.abs(qsys['gpe']['wavefunctions'])[:, x_cut:] ** 2, axis=1) * dz,
        label='GPE'
    )
    plt.plot(
        times,
        np.sum(np.abs(qsys_flipped['gpe']['wavefunctions'])[:, :x_cut_flipped] ** 2, axis=1) * dz,
        label='Flipped GPE'
    )
    plt.legend(loc='upper left')
    plt.xlabel('Time $t$ (au)')
    plt.ylabel("Transmission Probability")
    plt.savefig(file_path + 'Transmission Probability' + '.pdf')

    plt.figure(fignum, layout='constrained')
    fignum += 1
    potential = V(z)
    plt.plot(z, potential, color='k')
    plt.hlines((qsys['gpe']['hamiltonian_average'][0]), z.min(), z.max(), colors='r')
    plt.fill_between(
        z[x_cut:],
        potential[x_cut:],
        potential.max(),
        facecolor=ok['blue'],
        color=ok['blue'],
        alpha=0.5
    )
    plt.fill_between(
        z[:x_cut_flipped],
        potential[:x_cut_flipped],
        potential.max(),
        facecolor=ok["orange"],
        color=ok['orange'],
        alpha=0.5
    )
    plt.xlabel('$x$ (au) ')
    plt.ylabel('$V(x)$ (au)')
    plt.xlim(-z_lim, z_lim)
    plt.savefig(file_path + 'Potential' + '.pdf')


