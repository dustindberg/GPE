import utlt
from split_op_gpe1D import imag_time_gpe1D, SplitOpGPE1D
import numpy as np
import matplotlib.pyplot as plt
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
def pulse(pos_grid, height, width, center):
    """
    Adjustable width Gaussian. Passed as a function for repeated use and readability
    :param pos_grid:
    :param height:
    :param center:
    :param width:
    :return:
    """
    return height * np.exp(-((pos_grid - center) / width) ** 2)

@njit
def diff_pulse(pos_grid, height, width, center):
    """
    Derivative of the
    :param pos_grid:
    :param height:
    :param center:
    :param width:
    :return:
    """
    return (-2 * height / width) * (pos_grid - center) * np.exp(-((pos_grid - center) / width) ** 2)

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
waist = 15                                  # Waist of the gaussian beam (dimensionless)
sigma = waist / 2                           # Delta = sqrt(2) sigma, such that delta^2 = 2 * sigma^2
fwhm = 2 * sigma * np.sqrt(2 * np.log(2))   # Full Width, Half Maximum (used for spacing parameters)
delta = 2 * sigma ** 2                      # Minimum Width Parameter for the Gaussian function, pulse()
gap = sigma * np.sqrt(2 * np.log(2))        # Half of Full-width, Half-Max for creating uniform gaps between Gaussians
barrier_height = 400                        # Dimensionless height of the barrier

# Create the position grid and times
pos_grid_dim = 1 * 1024         # Resolution for the position grid (in our case, this is the x-axis resolution)
pos_amplitude = 100             # Code is always centered about 0, so +/- amplitude are the bounds of the position grid
tp_cut = 0.3                    # % of position grid to integrate over for tunneling
T = 1.                          # Final time. Declare as a separate parameter for conversions
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
kick = 4.5

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
        p1 = pulse(x, barrier_height, sigma, 3 * gap)
        p2 = pulse(x, barrier_height, sigma, 0.5 * opt - 2 * gap)
        p3 = pulse(x, barrier_height, sigma, opt - gap)
        p4 = pulse(x, barrier_height, sigma, opt)
        p5 = pulse(x, barrier_height, sigma, opt + gap)
        p6 = pulse(x, barrier_height, sigma, 0.5 * opt + 2 * gap)
        p7 = pulse(x, barrier_height, sigma, 3 * gap)
        pulses = p1 + p2 + p3 + p4 + p5 + p6 + p7
        pulses *= barrier_height / pulses.max()
        return pulses

    @njit(parallel=True)
    def diff_v(x, t=0):
        """

        :param x: position grid as an array
        :param t: time grid as an array
        :return: Derivative of the potential as a function of x (and t if relevant)
        """
        p1 = pulse(x, barrier_height, sigma, -3 * gap)
        p2 = pulse(x, barrier_height, sigma, -2 * gap + 0.5 * opt)
        p3 = pulse(x, barrier_height, sigma, -gap + opt)
        p4 = pulse(x, barrier_height, sigma, 0. + opt)
        p5 = pulse(x, barrier_height, sigma, gap + opt)
        p6 = pulse(x, barrier_height, sigma, 2 * gap + 0.5 * opt)
        p7 = pulse(x, barrier_height, sigma, 3 * gap)
        pulses = p1 + p2 + p3 + p4 + p5 + p6 + p7
        pulses *= barrier_height / pulses.max()
        height_mod = barrier_height / pulses.max()
        return height_mod * (diff_pulse(x, barrier_height, sigma, 3 * gap) +
                             diff_pulse(x, barrier_height, sigma, 0.5 * opt - 2 * gap) +
                             diff_pulse(x, barrier_height, sigma, opt - gap) +
                             diff_pulse(x, barrier_height, sigma, opt) +
                             diff_pulse(x, barrier_height, sigma, opt + gap) +
                             diff_pulse(x, barrier_height, sigma, 0.5 * opt + 2 * gap) +
                             diff_pulse(x, barrier_height, sigma, 3 * gap)
                             )

    param['v'] = v
    param['diff_v'] = diff_v
    param['iterator'] = opt

    return np.array(gpe.run_single_case_structured(param))


########################################################################################################################
# Begin the optimization process
########################################################################################################################
savespath = './'
filename = 'Test'

if __name__ == '__main__':
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
    offsets = -1 * np.logspace(0.0 * gap, 0.5 * gap, 2)     # Use 41 when this starts working
    full_params = [sys_params_left, sys_params_right]

    with Pool() as pool:
        task = [pool.apply_async(run_in_parallel, [a, b]) for a, b in product(offsets, full_params)]
        pool.close()
        pool.join()
        results = np.array([_.get() for _ in task])
    h5f = h5py.File('data.h5', 'w')
    for _ in results:
        tag = utlt.replace(str(_))
        h5f.create_dataset(tag, data=np.array(results(_)))
    h5f.close()

    h5f = h5py.File('data.h5', 'r')
    b = h5f['dataset_1'][:]
    h5f.close()
    np.allclose(results, b)


########################################################################################################################
# Save the Results
########################################################################################################################

########################################################################################################################
# Plot the Results
########################################################################################################################

# probs = np.sum(np.abs(qsys_flipped['gpe']['wavefunctions'])[:, :x_cut_flipped] ** 2, axis=1) * dx,
