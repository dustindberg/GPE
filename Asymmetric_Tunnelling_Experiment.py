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
        'legend.fontsize': 16,
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

propagation_freq = 0.5 * 2 * np.pi          # Frequency of the shallow trap along axis of propagation (we use x-axis)
perpendicular_freqs = 200 * 2 * np.pi       # Frequency of the axes perpendicular to propagation (we use y- and z-axis)

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
bounds = 100         # Declare the boundaries in physical units (micrometers)
waist = 23        # Declare the inter-gaussian spacing in physical units (micrometers)
sigma = waist / 2
trap_height = 60    # Declare the trap height/depth in physical units (microKelvin)
th_coding = gpe.dimless_energy(trap_height, -6, units='K')
resolution = 2 ** 14

dz = 2. * bounds / resolution

# generate coordinate range
z = (np.arange(resolution) - resolution / 2) * dz
max_asymmetry = 0.25
asym_position = 1 * sigma
well_center = 3 * sigma


#@njit
def static_potential(x, asymmetry):
    left_barrier = (2 * max_asymmetry - asymmetry) * utlt.pulse(x, sigma, -asym_position)
    left_correction = utlt.pulse(x, sigma, -well_center) * (
            (2 * max_asymmetry - asymmetry) * utlt.pulse(-well_center, sigma, -asym_position) +
            (2 * max_asymmetry + asymmetry) * utlt.pulse(-well_center, sigma, asym_position))
    right_barrier = (2 * max_asymmetry + asymmetry) * utlt.pulse(x, sigma, asym_position)
    right_correction = utlt.pulse(x, sigma, well_center) * (
            (2 * max_asymmetry + asymmetry) * utlt.pulse(well_center, sigma, asym_position) +
            (2 * max_asymmetry - asymmetry) * utlt.pulse(well_center, sigma, -asym_position))
    return left_barrier - left_correction + right_barrier - right_correction


@njit
def opposite_barrier(x):
    return trap_height * utlt.pulse(x, sigma, well_center)

@njit
def init_barrier(x):
    return trap_height * utlt.pulse(x, sigma, -well_center)


plt.figure()
for _ in np.linspace(0, max_asymmetry, 11):
    def v(x, t=0):
        static = trap_height * static_potential(x, _)
        #plt.plot(x, left_c, '-.', label='%.2f Left Correction' % _)
        #plt.plot(x, right_c, '-.', label='%.2f Right Correction' % _)
        #plt.plot(x, opposite_barrier(x), '-.')
        #plt.plot(x, asymish, '-.')
        #plt.plot(x, static, '--')
        #plt.plot(x, 1 - (1 - lc) * init_barrier(x), '-.')
        #plt.plot(x, static, '--')
        return (trap_height + 2) - (static + opposite_barrier(x) + init_barrier(x))
    plt.plot(z, v(z), label='%.2f' % _)
plt.plot(z, (trap_height + 2) - opposite_barrier(z), '--')
plt.plot(z, (trap_height + 2) - init_barrier(z), '--')
plt.axvline(well_center, color=ok['red'])
plt.axvline(-well_center, color=ok['red'])
#plt.legend(loc='lower left')

plt.show()
