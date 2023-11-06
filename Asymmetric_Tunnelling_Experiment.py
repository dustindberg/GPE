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
bounds = 80         # Declare the boundaries in physical units (micrometers)
spacing = 20        # Declare the inter-gaussian spacing in physical units (micrometers)
trap_height = 500    # Declare the trap height/depth in physical units (nanoKelvin)
resolution = 2 ** 14

dz = 2. * bounds / resolution

# generate coordinate range
z = (np.arange(resolution) - resolution / 2) * dz
@njit
def v(x, height, asymmetry):
    left_v = utlt.pulse(x, 0.5*spacing, -2*spacing)
    right_v = asymmetry * utlt.pulse(
    #    x, 0.5*spacing, 0
    #) + asymmetry * utlt.pulse(
        x, 0.5*spacing, spacing
    ) + utlt.pulse(
        x, 0.5*spacing, 2*spacing
    )
    left_v *= height/left_v.max()
    right_v *= height/right_v.max()
    return height - (left_v + right_v)

plt.plot(v(z, trap_height, 0.5))
plt.show()
