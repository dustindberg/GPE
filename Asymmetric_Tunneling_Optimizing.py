import utlt
from split_op_gpe1D import imag_time_gpe1D, SplitOpGPE1D
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.constants import hbar, Boltzmann
from scipy.interpolate import UnivariateSpline
import tqdm
import h5py
import sys
import os

########################################################################################################################
# Determine global functions
########################################################################################################################


def pulse(pos_grid, height, center, width):
    """
    Adjustable width Gaussian. Passed as a function for repeated use and readability
    :param pos_grid:
    :param height:
    :param center:
    :param width:
    :return:
    """
    return height * np.exp(np.exp(-((pos_grid - center) / width) ** 2))


def diff_pulse(pos_grid, height, center, width):
    """
    Derivative of the
    :param pos_grid:
    :param height:
    :param center:
    :param width:
    :return:
    """

########################################################################################################################
# Get the initial parameters and determine pulses for traps
########################################################################################################################

########################################################################################################################
# Get Njit functions (while it is best practices to determine functions at the beginning of a program, these functions
# need the above determined parameters in order to run
########################################################################################################################

########################################################################################################################
# Begin the optimization process
########################################################################################################################

########################################################################################################################
# Save the Results
########################################################################################################################

########################################################################################################################
# Plot the Results
########################################################################################################################

