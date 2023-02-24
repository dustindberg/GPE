from numba import njit
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
from scipy.constants import hbar, Boltzmann
from scipy.interpolate import UnivariateSpline
from split_op_gpe1D import SplitOpGPE1D, imag_time_gpe1D    # class for the split operator propagation
import tqdm
import h5py
import sys
import os


class BEC:
    """
    This is an externalized class for parameters, repeated use tools, and for swapping between experimental types
    """
    def __init__(self, atom='R87', number_of_atoms=1e4, omega_x=100*np.pi, omega_y=1000*np.pi, omega_z=1000*np.pi,
                 omega_cooling=900*np.pi):
        """

        :param str atom: Specify R87 of K41 for rubidium or Potassium BEC respectively
        :param int number_of_atoms: Number of atoms for BEC
        """
        # Define physical parameters
        self.a_0 = 5.291772109e-11          # Bohr Radius in meters

        # Rubidium-87 properties
        self.m_R87 = 1.4431609e-25          # Calculated mass of 87Rb in kg
        self.a_s_R87 = 100 * self.a_0       # Background scattering length of 87Rb in meters

        # Potassium-41 properties
        self.m_K41 = 6.80187119e-26         # Calculated mass of 41K in kg
        self.a_s_K41 = 65.42 * self.a_0     # Background scattering length of 41K in meters

        # Reduction to 1D params
        self.N = number_of_atoms            # Number of particles
        self.omeg_x = omega_x               # Harmonic oscillation in the x-axis in 2pi Hz
        self.omeg_y = omega_y               # Harmonic oscillation in the y-axis in 2pi Hz
        self.omeg_z = omega_z               # Harmonic oscillation in the z-axis in 2pi Hz
        self.omeg_cooling = omega_cooling   # Harmonic oscillation for the trapping potential in Hz (if needed)

        if atom == "R87":
            self.mass = self.m_R87
            self.a_s = self.a_s_R87
        elif atom == "K41":
            self.mass = self.m_K41
            self.a_s = self.a_s_K41
        else:
            sys.exit('ERROR: Parameter(atom) must be of the form: \n '
                      'Type: string \n '
                      'For Rubidium-87: R87 \n'
                      'For Potassium-41: K41 \n '
                      'Your input was: {} of type: {}'.format(atom, type(atom)))
        # Parameters calculated by Python
        self.L_x = np.sqrt(hbar / (self.m_R87 * self.omeg_x))  # Characteristic length in the x-direction in meters
        self.L_y = np.sqrt(hbar / (self.m_R87 * self.omeg_y))  # Characteristic length in the y-direction in meters
        self.L_z = np.sqrt(hbar / (self.m_R87 * self.omeg_z))  # Characteristic length in the z-direction in meters

        self.g = 2 * self.N * self.L_x * self.mass * self.a_s * np.sqrt(
            self.omeg_y * self.omeg_z) / hbar  # Dimensionless interaction parameter

    def qatc(self, param):
        """
        Quick Array Type Checking: This will brute force any input that is not an int, float, or numpy.ndarray into a
        numpy.ndarray
        :param param: pass a parameter that you want to make sure is not a list
        :return: If you passed anything other than a list
        """
        if not isinstance(param, (int, float, np.ndarray)):
            param = np.array(param)
        return param

    def convert_x(self, position, order):
        """
        Takes dimensionless units used in simulation and convert them to dimensioned units
        :param int|float|np.ndarray position: Pass a point or np.array to be converted
        :param int|float order: Specify the order, X, for 1eX meters. For example, to convert to micrometers, X=-6
        :return: Position grid in Xm, where X is the order of the units you desire. Eg. nanometers = 1e-9 meters
        """
        return self.qatc(position) * self.L_x * 10 ** (-order)

    def convert_time(self, time, order):
        """

        :param int|float|np.ndarray time: Pass a point in time or np.array to be converted
        :param int|float order: Specify the order, X, with respect to seconds. Eg. for milliseconds, X=-3
        :return: Time point or time grid in order Xs, where X is the order of the units you want. Eg. nanoseconds, X=-9
        """
        return self.qatc(time) * (10 ** -order) / self.omeg_x

    def convert_energy(self, energy, order, units='K'):
        """
        Converts dimensionless energy into Either Joules or Kelvin
        :param int|float|np.ndarray energy: dimensionless energy
        :param int|float order: Specify the order, X, with respect to Joules or Kelvin. Eg, for microKelvin, X=-6
        :param str units:
        :return:
        """
        energy = self.qatc(energy) * hbar * self.omeg_x
        if units == 'K':
            energy /= Boltzmann
        return energy * 10 ** -order

    def convert_dens(self, density, order):
        """

        :param density: Pass a specific density quantity or np.array to be converted
        :param order: Specify the order, X, with respect to m^3. Eg. for square kilometer, X=3
        :return: density array of
        """
        return density / (self.L_x * self.L_y * self.L_z * 10 ** (3 * -order))

    def run_single_case(self, params):
        """
        Does a single propagation of a BEC. Since interaction parameter, g, is specified here, this can be used for
        Schrodinger propagation as well
        :param params: system parameters such as, x_grid_dimension, times, g, potential and kinetic energies, etc.
        :return: THe results of the single approximation.
        """
        gpe_propagator = SplitOpGPE1D(**params)
        gpe_wavefunctions = [gpe_propagator.propagate(t).copy() for t in tqdm(params['times'])]

        return {
            # Returns bundled results of a single set of parameters.
            'wavefunctions': gpe_wavefunctions,
            'extent': [gpe_propagator.x.min(), gpe_propagator.x.max(),
                       min(gpe_propagator.times), max(gpe_propagator.times)],
            'times': gpe_propagator.times,

            'x_average': gpe_propagator.x_average,
            'x_average_rhs': gpe_propagator.x_average_rhs,

            'p_average': gpe_propagator.p_average,
            'p_average_rhs': gpe_propagator.p_average_rhs,
            'hamiltonian_average': gpe_propagator.hamiltonian_average,

            'time_increments': gpe_propagator.time_increments,

            'dx': gpe_propagator.dx,
            'x': gpe_propagator.x,

            'parameters': params
        }


def replace(string_for_replacement):
    """
    For naming conventions, this will remove any decimal points from parameters to avoid strange document types
    Doesn't need to be a string, but only because I am a kind and benevolent god. Please pass a string.
    :param str string_for_replacement: The filename/path which contains periods, such as T=0.5
    :return: A string with all instances of '.' replaced with ',' such as T=0.5 -> T=0,5
    """
    return str(string_for_replacement).replace('.', ',')


