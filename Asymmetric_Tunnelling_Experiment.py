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
import sys
import os

########################################################################################################################
# Declare all functions here
########################################################################################################################

# Set the initial plotting parameters

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
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'legend.fontsize': 10,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'axes.prop_cycle': plt.cycler('color', (ok[_] for _ in ok)),
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'lines.linewidth': 3.5,
}
plt.rcParams.update(plt_params)

########################################################################################################################
# Get the BEC
########################################################################################################################

propagation_freq = 0.5 * 2 * np.pi  # Frequency of the shallow trap along axis of propagation (we use x-axis)
perpendicular_freqs = 200 * 2 * np.pi  # Frequency of the axes perpendicular to propagation (we use y- and z-axis)

atom_params = dict(
    atom='R87',
    omega_x=propagation_freq,
    omega_y=perpendicular_freqs,
    omega_z=perpendicular_freqs,
    kicked=False,
)

gpe = utlt.BEC(**atom_params)
g = gpe.g
N = gpe.N

# Declare Physical parameters
bounds = 100  # Declare the boundaries in physical units (micrometers)
waist = 23  # Declare the inter-gaussian spacing in physical units (micrometers)
sigma = waist / 2
trap_height = 60  # Declare the trap height/depth in physical units (microKelvin)
th_coding = gpe.dimless_energy(trap_height, -6, units='K')
resolution = 2 ** 14

dz = 2. * bounds / resolution

# generate coordinate range
z = (np.arange(resolution) - resolution / 2) * dz
max_asymmetry = 0.5
asym_position = sigma
well_center = 3 * sigma
barrier_height = 0.5
T = 3
times = np.linspace(0, T, 500)


# @njit
def asymmetry_potential(x, asymmetry):
    # Define the asymmetric barriers
    left_asymmetry = (1 - asymmetry) * utlt.pulse(x, sigma, -asym_position)
    right_asymmetry = utlt.pulse(x, sigma, asym_position)
    asymmetric_barrier = left_asymmetry + right_asymmetry
    height_mod = (1 + 0.5 * asymmetry) * barrier_height / asymmetric_barrier.max()
    asymmetric_barrier *= height_mod
    left_correction_amplitude = height_mod * (
            (1 - asymmetry) * utlt.pulse(-well_center, sigma, -asym_position) +
            utlt.pulse(-well_center, sigma, asym_position)
    )
    left_well_correction = -left_correction_amplitude * utlt.pulse(x, sigma, -well_center)

    # Define the right portion of the barrier

    right_correction_amplitude = height_mod * (
            (1 - asymmetry) * utlt.pulse(well_center, sigma, -asym_position) +
            utlt.pulse(well_center, sigma, asym_position)
    )
    right_well_correction = -right_correction_amplitude * utlt.pulse(x, sigma, well_center)
    return asymmetric_barrier + left_well_correction + right_well_correction


@njit
def left_barrier(x):
    return utlt.pulse(x, sigma, -well_center)


@njit
def right_barrier(x):
    return utlt.pulse(x, sigma, well_center)


def path(x_0, x_f, t, time_stop=times.max(), path_type='sinusoid'):
    t = np.array(t)
    length = (x_f - x_0) / 2
    tracked_time = t[t <= time_stop]
    if path_type == 'sinusoid':
        tracked_position = np.array(x_0 + length * (1 - np.cos(tracked_time * np.pi / time_stop)))
    if path_type == 'half sine':
        tracked_position = np.array(x_0 + 2 * length * (1 - np.cos(tracked_time * np.pi / (2 * time_stop))))

    if time_stop != t[-1]:
        untracked_time = t[t > time_stop]
        stationary = np.full(np.shape(untracked_time), x_f)
        complete_path = np.concatenate((tracked_position, stationary))
        return complete_path
    else:
        return tracked_position



plt.figure()
for _ in np.linspace(0, max_asymmetry, 11):
    def v(x, t=0):
        static = asymmetry_potential(x, _) + left_barrier(x)
        #left_correction = static_potential(-well_center, _)
        #right_correction = static_potential(well_center, _)
        # print('Corrections - Left: {}    Right: {}'.format(left_correction, right_correction))
        # plt.plot(x, left_c, '-.', label='%.2f Left Correction' % _)
        # plt.plot(x, right_c, '-.', label='%.2f Right Correction' % _)
        # plt.plot(x, opposite_barrier(x), '-.')
        # plt.plot(x, asymish, '-.')
        #plt.plot(x, trap_height - static, '--')
        # plt.plot(x, 1 - (1 - lc) * init_barrier(x), '-.')
        #plt.plot(x, static, '--')
        barrier = static + right_barrier(x)
        return trap_height * (1 - barrier)


    #print('Asymmetry: {}    Left Well: {}    Right Well: {}'.format(_, v(-well_center), v(well_center)))
    #plt.plot(z, v(z), label='%.2f' % _)
#plt.plot(z, trap_height - left_barrier(z), '--')
#plt.plot(z, trap_height - right_barrier(z), '--')
#plt.axvline(well_center, color=ok['red'])
#plt.axvline(-well_center, color=ok['red'])
#plt.legend(loc='lower left')
#plt.show()

plt.plot(times, path(-2 * well_center, -well_center, times, time_stop=0.5*T, path_type='sinusoid'))
plt.show()

if __name__ == '__main__':
    print('done')
