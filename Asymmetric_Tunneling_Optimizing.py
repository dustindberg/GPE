import utlt
from split_op_gpe1D import imag_time_gpe1D, SplitOpGPE1D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from numba import njit
from scipy.constants import hbar, Boltzmann
from scipy.interpolate import UnivariateSpline
from itertools import product
from multiprocessing import Pool, Process, Manager
import tqdm
import pickle
import h5py
import sys
import os

########################################################################################################################
# Determine global functions
########################################################################################################################

@njit
def pulse(pos_grid, width, center):
    """
    Adjustable width Gaussian. Passed as a function for repeated use and readability
    :param pos_grid:
    :param height:
    :param center:
    :param width:
    :return:
    """
    return np.exp(-((pos_grid - center) / width) ** 2)

@njit
def diff_pulse(pos_grid, width, center):
    """
    Derivative of the
    :param pos_grid:
    :param height:
    :param center:
    :param width:
    :return:
    """
    return (-2 / width) * (pos_grid - center) * np.exp(-((pos_grid - center) / width) ** 2)

@njit
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


########################################################################################################################
# Get the initial parameters and determine pulses for traps
########################################################################################################################
# Trap Frequencies for creating the Quasi-1D case
propagation_freq = 0.5 * 2 * np.pi          # Frequency of the shallow trap along axis of propagation (we use x-axis)
perpendicular_freqs = 200 * 2 * np.pi       # Frequency of the axes perpendicular to propagation (we use y- and z-axis)

# Beam Parameters
waist = 2                                  # Waist of the gaussian beam (dimensionless)
sigma = waist / 2                           # Delta = sqrt(2) sigma, such that delta^2 = 2 * sigma^2
fwhm = 2 * sigma * np.sqrt(2 * np.log(2))   # Full Width, Half Maximum (used for spacing parameters)
delta = 2 * sigma ** 2                      # Minimum Width Parameter for the Gaussian function, pulse()
gap = sigma * np.sqrt(2 * np.log(2))        # Half of Full-width, Half-Max for creating uniform gaps between Gaussians
barrier_height = 500                        # Dimensionless height of the barrier

# Create the position grid and times
pos_grid_dim = 1 * 1024         # Resolution for the position grid (in our case, this is the x-axis resolution)
pos_amplitude = 75              # Code is always centered about 0, so +/- amplitude are the bounds of the position grid
tp_cut = 0.3                    # % of position grid to integrate over for tunneling
T = 0.2                         # Final time. Declare as a separate parameter for conversions
times = np.linspace(0, T, 500)  # Time grid: required a resolution of 500 for split operator class
prop_dt = 1e-6                  # Initial guess for the adaptive step propagation

atom_params = dict(
    atom='R87',
    omega_x=propagation_freq,
    omega_y=perpendicular_freqs,
    omega_z=perpendicular_freqs
)

gpe = utlt.BEC(**atom_params)
g = gpe.g
N = gpe.N
kick = 25


########################################################################################################################
# Create the BEC in a trap
########################################################################################################################


@njit
def initial_trap(x):
    """
    Trapping potential to get the initial state
    :param x:
    :return:
    """
    return 0.25 * (x + (0.5 * pos_amplitude)) ** 2


def run_in_parallel(opt, param):
    print('Beginning {} side GPE propagation with asymm parameter: {:.4f}'.format(param['side'], opt))

    @njit(parallel=True)
    def v(x, t=0):
        """
        Function for the propagation potential
        :param x: position grid as an array
        :param t: time grid as an array
        :return: The potential as a function of x (and t if relevant)
        """
        p1 = (1 - opt) * pulse(x, sigma, 4 * gap)
        p2 = pulse(x, sigma, 3 * gap)
        p3 = (1 + opt) * pulse(x, sigma, 2 * gap)
        p4 = (1 - opt) * pulse(x, sigma, -2 * gap)
        p5 = pulse(x, sigma, -3 * gap)
        p6 = (1 + opt) * pulse(x, sigma, -4 * gap)
        pulses = p1 + p2 + p3 + p4 + p5 + p6
        pulses *= barrier_height / pulses.max()
        return pulses

    @njit(parallel=True)
    def diff_v(x, t=0):
        """

        :param x: position grid as an array
        :param t: time grid as an array
        :return: Derivative of the potential as a function of x (and t if relevant)
        """
        p1 = (1 - opt) * pulse(x, sigma, 4 * gap)
        p2 = pulse(x, sigma, 3 * gap)
        p3 = (1 + opt) * pulse(x, sigma, 2 * gap)
        p4 = (1 - opt) * pulse(x, sigma, -2 * gap)
        p5 = pulse(x, sigma, -3 * gap)
        p6 = (1 + opt) * pulse(x, sigma, -4 * gap)
        pulses = p1 + p2 + p3 + p4 + p5 + p6
        pulses *= barrier_height / pulses.max()
        height_mod = barrier_height / pulses.max()
        return height_mod * ((1 - opt) * diff_pulse(x, sigma, 4 * gap) +
                             diff_pulse(x, sigma, 3 * gap) +
                             (1 + opt) * diff_pulse(x, sigma, 2 * gap) +
                             (1 - opt) * diff_pulse(x, sigma, -2 * gap) +
                             diff_pulse(x, sigma, -3 * gap) +
                             (1 + opt) * diff_pulse(x, sigma, -4 * gap)
                             )

    param['v'] = v
    param['diff_v'] = diff_v
    param['iterator'] = opt

    return np.array(gpe.run_single_case_structured(param))


########################################################################################################################
# Begin the optimization process
########################################################################################################################


if __name__ == '__main__':
    print('Waist size is {} um'.format(gpe.convert_x(waist, -6)))
    print('Initial energy of cooled BEC is {} nK'.format(gpe.convert_energy(1.0226e2, -9)))
    n_iters = 2
    unique_tag = n_iters
    savespath = './Archive_Data/IterationsOver{}'.format(unique_tag)
    filename = 'Test3'

    prob_region = 0.4
    # To do, move cooling here pleeeease, that way it only runs twice
    left_cool_params = dict(
        x_grid_dim=pos_grid_dim,
        x_amplitude=pos_amplitude,
        k=k,
        initial_trap=initial_trap,
        g=g,
        dt1=1e-3,
        dt2=1e-5,
        eps1=1e-8,
        eps2=1e-10,
    )

    left_init, left_mu = gpe.cooling(left_cool_params)
    right_cool_params = left_cool_params.copy()
    right_cool_params['initial_trap'] = njit(lambda x: initial_trap(-x))
    right_init, right_mu = gpe.cooling(right_cool_params)
    sys_params_left = dict(
        x_amplitude=pos_amplitude,
        x_grid_dim=pos_grid_dim,
        g=g,
        N=N,
        k=k,
        dt=prop_dt,
        init_state=left_init,
        initial_trap=initial_trap,
        diff_k=diff_k,
        times=times,
        init_momentum_kick=kick,
        side='left',
    )

    # Copy the Left starting params, and then modify so that it starts on the right
    sys_params_right = sys_params_left.copy()
    sys_params_right['initial_trap'] = njit(lambda x: initial_trap(-x))
    sys_params_right['init_state'] = right_init
    sys_params_right['init_momentum_kick'] = -sys_params_left['init_momentum_kick']
    sys_params_right['side'] = 'right'
    offsets = -1 * np.linspace(0.0, 1.0, n_iters)     # Use 41 when this starts working
    both_side_params = [sys_params_left, sys_params_right]

    with Pool() as pool:
        task = [pool.apply_async(run_in_parallel, [a, b]) for a, b in product(offsets, both_side_params)]
        pool.close()
        pool.join()
        results = np.array([_.get() for _ in task])

    ####################################################################################################################
    # Save the Results
    ####################################################################################################################
    with open(savespath + filename + ".pickle", "wb") as f:
        pickle.dump(results, f)

    with open(savespath + filename + ".pickle", "rb") as f:
        loaded = pickle.load(f)

    ####################################################################################################################
    # Plot the Results
    ####################################################################################################################
    # Pair the results of different iterators together in shared directories
    res_lib = dict()
    for o in offsets:
        res_lib['{:.4f}'.format(o)] = dict()
        for _ in loaded:
            if _['parameters']['iterator'] == o:
                res_lib['{:.4f}'.format(o)]['{}'.format(_['parameters']['side'])] = _

    figure_number = 0

    for _ in loaded:
        plt.figure(figure_number)
        figure_number += 1

        plot_title = 'Density{:.4f}param-{}'.format(_['parameters']['iterator'], _['parameters']['side'])
        # plot the time dependent density
        extent = _['extent']
        plt.imshow(
            np.abs(_['wavefunctions']) ** 2,
            # some plotting parameters
            origin='lower',
            extent=extent,
            aspect=(extent[1] - extent[0]) / (extent[-1] - extent[-2]),
            norm=SymLogNorm(vmin=1e-7, vmax=1., linthresh=1e-8)
        )
        plt.xlabel('Coordinate $x$ (a.u.)')
        plt.ylabel('Time $t$ (a.u.)')
        plt.colorbar()
        plt.savefig(savespath + plot_title + '.pdf')

    plt.close('all')

    TP_lib = dict()
    for _ in res_lib:
        l2r = res_lib[_]['left']
        r2l = res_lib[_]['right']
        x_um = gpe.convert_x(res_lib[_]['left']['x'], -6)
        t_ms = gpe.convert_time(times, -6)
        l_size = l2r['x'].size  # Create size parameter for probability evaluation
        r_size = r2l['x'].size
        l_cut = int((1 - prob_region) * l_size)  # Use size parameter for Left-to-Right probability
        r_cut = int(prob_region * r_size)  # Use size parameter for Right-to-Left probability
        T_l2r = np.sum(np.abs(l2r['wavefunctions'])[:, l_cut:] ** 2, axis=1) * l2r['dx']
        T_r2l = np.sum(np.abs(r2l['wavefunctions'])[:, :r_cut] ** 2, axis=1) * r2l['dx']
        # Plot the Tunneling results
        plt.figure(figure_number)
        figure_number += 1
        plot_title = 'TunnelingProbability-{}param'.format(_)
        plt.plot(t_ms, T_l2r, label='Tunneling probability from Left to Right')
        plt.plot(t_ms, T_l2r, label='Tunneling probability from Left to Right')
        plt.xlabel('Coordinate $x (\mu$m))')
        plt.ylabel('Time $t (\mu$s)')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(savespath + plot_title + '.pdf')
        # Plot the Potential
        x_line = [x_um[0], x_um[-1]]
        en_left = gpe.convert_energy(np.array([l2r['hamiltonian_average'], l2r['hamiltonian_average']]), -9)
        en_right = gpe.convert_energy(np.array([r2l['hamiltonian_average'], r2l['hamiltonian_average']]), -9)
        plt.figure(figure_number)
        figure_number += 1
        plot_title = 'Potential-{}param'.format(_)
        plt.plot(x_um, gpe.convert_energy(l2r['parameters']['v'](l2r['x']), -9))
        plt.plot(x_um, gpe.convert_energy(r2l['parameters']['v'](r2l['x']), -9))
        plt.plot(x_line, en_left)
        plt.plot(x_line, en_right)
        plt.xlabel('Coordinate $x (\mu$m))')
        plt.ylabel('Energy $E$ (nK)')
        plt.tight_layout()
        plt.savefig(savespath + plot_title + '.pdf')
        TP_lib[_] = {'l2r': T_l2r, 'r2l': T_r2l, 'diff': np.sum(np.abs(T_l2r - T_r2l))}

    plt.close('all')

    plt.plot(offsets, [TP_lib[_]['diff'] for _ in TP_lib])
    plt.xlabel('Offsets')
    plt.ylabel('$\int\|T_r - T_l\| dt$')
    plt.tight_layout()
    plt.savefig(savespath + 'ProbabilityPlot')
    plt.show()
