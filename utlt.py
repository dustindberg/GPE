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
import sys
import os

class Tools:
    """
    This is an externalized class for parameters, repeated use tools, and for swapping between experimental types
    """
    def __init__(self, atom='R87'):
        # Define physical parameters
        self.a_0 = 5.291772109e-11  # Bohr Radius in meters

        # Rubidium-87 properties
        self.m_R87 = 1.4431609e-25  # Calculated mass of 87Rb in kg
        self.a_s_R87 = 100 * self.a_0  # Background scattering length of 87Rb in meters

        # Potassium-41 properties
        self.m_K41 = 6.80187119e-26                  # Calculated mass of 41K in kg
        self.a_s_K41 = 65.42 * self.a_0                  # Background scattering length of 41K in meters

        # Reduction to 1D params
        self.N = 1e4  # Number of particles
        self.omeg_x = 50 * 2 * np.pi  # Harmonic oscillation in the x-axis in Hz
        self.omeg_y = 500 * 2 * np.pi  # Harmonic oscillation in the y-axis in Hz
        self.omeg_z = 500 * 2 * np.pi  # Harmonic oscillation in the z-axis in Hz
        self.omeg_cooling = 450 * 2 * np.pi  # Harmonic oscillation for the trapping potential in Hz

        if atom == "R87":
            self.mass = self.m_R87
            self.a_s = self.a_s_R87
        elif atom == "K41":
            self.mass = self.m_K41
            self.a_s = self.a_s_K41
        else:
            sys.exit('ERROR: Parameter(atom) must be of the form: \n '
                      'Type:string \n '
                      'For Rubidium-87: R87 \n'
                      'For Potassium-41: K41 \n '
                      'Your input was: {} of type: {}'.format(atom, type(atom)))
        # Parameters calculated by Python
        self.L_x = np.sqrt(hbar / (self.m_R87 * self.omeg_x))  # Characteristic length in the x-direction in meters
        self.L_y = np.sqrt(hbar / (self.m_R87 * self.omeg_y))  # Characteristic length in the y-direction in meters
        self.L_z = np.sqrt(hbar / (self.m_R87 * self.omeg_z))  # Characteristic length in the z-direction in meters

        self.interaction_param = 2 * self.N * self.L_x * self.mass * self.a_s * np.sqrt(
            self.omeg_y * self.omeg_z) / hbar  # Dimensionless interaction parameter

    def units_conversion(self, x_grid_array, time_array, density):
        """

        Args:
            x_grid_array:
            time_array:
            density:
        """





    def replace(self, str1):
        str1 = str1.replace('.', ',')
        return str1

