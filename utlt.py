from numba import njit
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
from numpy import linalg
from scipy.constants import hbar, Boltzmann
from scipy.interpolate import UnivariateSpline
from split_op_gpe1D import SplitOpGPE1D, imag_time_gpe1D  # class for the split operator propagation
from tqdm import tqdm
import sys
import pyfftw
import pickle as pickle
import h5py
import os
from collections import namedtuple


threads = 4


def relative_diff(psi_next, psi):
    """
    Efficiently calculate the relative difference of two wavefunctions. (Used in thea adaptive scheme)
    :param psi_next: numpy.array
    :param psi: numpy.array
    :return: float
    """
    return linalg.norm(psi_next - psi) / linalg.norm(psi_next)


class BEC:
    """
    This is an externalized class for parameters, repeated use tools, and for swapping between experimental types
    """

    def __init__(self, atom='R87', number_of_atoms=1e4, omega_x=100 * np.pi, omega_y=1000 * np.pi, omega_z=1000 * np.pi,
                 kicked=False):
        """
        Creates an instance of a BEC. This is done to reduce the amount of lines for
        :param str atom: Specify R87 of K41 for rubidium or Potassium BEC respectively
        :param int|float number_of_atoms: Number of atoms for BEC
        :param int|float omega_x: The frequency of the trapping potential along the axis of propagation
        :param int|float omega_y: The frequency of the trapping potential along a perpendicular axis
        :param int|float omega_z: The frequency of the trapping potential along a perpendicular axis
        :param bool kicked: Are you giving the BEC an initial momentum kick? If so, you need to specify here

        """
        # Define physical parameters
        self.a_0 = 5.291772109e-11  # Bohr Radius in meters

        # Rubidium-87 properties
        self.m_R87 = 1.4431609e-25  # Calculated mass of 87Rb in kg
        self.a_s_R87 = 100 * self.a_0  # Background scattering length of 87Rb in meters

        # Potassium-41 properties
        self.m_K41 = 6.80187119e-26  # Calculated mass of 41K in kg
        self.a_s_K41 = 65.42 * self.a_0  # Background scattering length of 41K in meters

        # Reduction to 1D params
        self.N = number_of_atoms  # Number of particles
        self.omega_x = omega_x  # Harmonic oscillation in the x-axis in 2pi Hz (radians per second)
        self.omega_y = omega_y  # Harmonic oscillation in the y-axis in 2pi Hz (radians per second)
        self.omega_z = omega_z  # Harmonic oscillation in the z-axis in 2pi Hz (radians per second)

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
        self.L_x = np.sqrt(hbar / (self.m_R87 * self.omega_x))  # Characteristic length in the x-direction in meters
        self.L_y = np.sqrt(hbar / (self.m_R87 * self.omega_y))  # Characteristic length in the y-direction in meters
        self.L_z = np.sqrt(hbar / (self.m_R87 * self.omega_z))  # Characteristic length in the z-direction in meters

        self.g = 2 * self.N * self.L_x * self.mass * self.a_s * np.sqrt(
            self.omega_y * self.omega_z) / hbar  # Dimensionless interaction parameter
        self.kicked = kicked

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

    def dimless_x(self, position, order=0):
        """
        Takes a physical quantity of some order of meters and converts them to dimensionless units
        :param int|float|np.ndarray position: Pass a point or np.array to be converted
        :param int|float order: Specify the order, X, for 1eX meters. For example, to convert from micrometers, X=-6
        """
        return self.qatc(position) / self.L_x * (10 ** order)

    def convert_time(self, time, order):
        """
        Takes dimensionless time and converts it to some order of seconds
        :param int|float|np.ndarray time: Pass a point in time or np.array to be converted
        :param int|float order: Specify the order, X, with respect to seconds. Eg. for milliseconds, X=-3
        :return: Time point or time grid in order Xs, where X is the order of the units you want. Eg. nanoseconds, X=-9
        """
        return (self.qatc(time) / self.omega_x) * (10 ** -order)

    def dimless_time(self, time, order):
        """
        Takes time in some order of seconds and converts it to dimensionless units
        :param int|float|np.ndarray time: Pass a point in time or np.array to be converted
        :param int|float order: Specify the order, X, with respect to seconds. Eg. for milliseconds, X=-3
        :return: Dimensionless Time point or time grid  converted from time of order X. Eg. nanoseconds, X=-9
        """
        return self.qatc(time) * self.omega_x * (10 ** order)

    def convert_energy(self, energy, order, units='K'):
        """
        Converts dimensionless energy into Either Joules or Kelvin
        :param int|float|np.ndarray energy: dimensionless energy
        :param int|float order: Specify the order, X, with respect to Joules or Kelvin. Eg, for microKelvin, X=-6
        :param str units:
        :return:
        """
        energy = self.qatc(energy) * hbar * self.omega_x
        if units == 'K':
            energy /= Boltzmann
        return energy * 10 ** -order

    def dimless_energy(self, energy, order, units='K'):
        """
        Converts energy in units of some order of either Joules or Kelvin and converts it to dimensionless units
        :param int|float|np.ndarray energy: energy with units specified by params: order and units.
        :param int|float order: Specify the order, X, with respect to Joules or Kelvin. Eg, for microKelvin, X=-6
        :param str units:
        :return:
        """
        energy = self.qatc(energy) / (hbar * self.omega_x)
        if units == 'K':
            energy *= Boltzmann
        return energy * 10 ** order

    def convert_dens(self, density, order):
        """

        :param density: Pass a specific density quantity or np.array to be converted
        :param order: Specify the order, X, with respect to m^3. Eg. for square kilometer, X=3
        :return: density array of
        """
        return density / (self.L_x * self.L_y * self.L_z * 10 ** (3 * -order))

    def cooling(self, params):
        """init_state, mu = imag_time_gpe1D(
            v=params['initial_trap'],
            g=params['g'],
            dt=params['dt1'],
            epsilon=params['eps1'],
            **params
        )
        init_state, mu = imag_time_gpe1D(
            v=params['initial_trap'],
            g=params['g'],
            init_wavefunction=init_state,
            dt=params['dt1'],
            epsilon=params['eps1'],
            **params
        )"""
        init_state, mu = imag_time_gpe1D(
            v=params['initial_trap'],
            dt=params['dt1'],
            epsilon=params['eps1'],
            **params
        )
        init_state, mu = imag_time_gpe1D(
            v=params['initial_trap'],
            init_wavefunction=init_state,
            dt=params['dt1'],
            epsilon=params['eps1'],
            **params
        )

        return init_state, mu

    @njit(parallel=True)
    def run_single_case(self, params):
        """
        Does a single propagation of a BEC. Since interaction parameter, g, is specified here, this can be used for
        Schrodinger propagation as well
        :param params: system parameters such as, x_grid_dimension, times, g, potential and kinetic energies, etc.
        :return: THe results of the single approximation.
        """
        gpe_propagator = SplitOpGPE1D(**params)
        if self.kicked:
            gpe_propagator.set_wavefunction(
                params['init_state'] * np.exp(1j * params['init_momentum_kick'] * gpe_propagator.x)
            )
        else:
            gpe_propagator.set_wavefunction(params['init_state'])

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

    def run_single_case_structured(self, params):
        """
        Does a single propagation of a BEC. Since interaction parameter, g, is specified here, this can be used for
        Schrodinger propagation as well
        :param params: system parameters such as, x_grid_dimension, times, g, potential and kinetic energies, etc.
        :param kicked: Is the system kicked?
        :return: THe results of the single approximation.
        """
        gpe_propagator = SplitOpGPE1D(**params)
        if self.kicked:
            gpe_propagator.set_wavefunction(
                params['init_state'] * np.exp(1j * params['init_momentum_kick'] * gpe_propagator.x)
            )
        else:
            gpe_propagator.set_wavefunction(params['init_state'])

        gpe_wavefunctions = [gpe_propagator.propagate(t).copy() for t in tqdm(params['times'])]
        extent = [gpe_propagator.x.min(), gpe_propagator.x.max(),
                  min(gpe_propagator.times), max(gpe_propagator.times)]

        """if not self.kicked:
            params['init_momentum_kick'] = 0

        struct_params = np.array(
            (params['x_amplitude'], params['x_grid_dim'], params['g'], params['N'], params['dt'],
             params['init_state'], params['initial_trap'], params['k'], params['diff_k'], params['v'],
             params['diff_v'], params['times'], params['init_momentum_kick'], params['side']),
            dtype=[('x_amplitude', type(params['x_amplitude'])), ('x_grid_dim', type(params['x_grid_dim'])),
                   ('g', type(params['g'])), ('N', type(params['N'])), ('dt', type(params['dt'])),
                   ('init_state', type(params['init_state'])), ('initial_trap', type(params['initial_trap'])),
                   ('k', type(params['k'])), ('diff_k', type(params['diff_k'])), ('v', type(params['v'])),
                   ('diff_v', type(params['diff_v'])), ('times', type(params['times'])),
                   ('init_momentum_kick', type(params['init_momentum_kick'])), ('side', type(params['side']))]
        )"""

        return np.array((gpe_wavefunctions[:], extent, gpe_propagator.times, gpe_propagator.x_average,
                         gpe_propagator.x_average_rhs, gpe_propagator.p_average, gpe_propagator.p_average_rhs,
                         gpe_propagator.hamiltonian_average, gpe_propagator.time_increments, gpe_propagator.dx,
                         gpe_propagator.x, {**params}),
                        dtype=[('wavefunctions', np.ndarray), ('extent', list), ('times', list), ('x_average', list),
                               ('x_average_rhs', list), ('p_average', list), ('p_average_rhs', list),
                               ('hamiltonian_average', list), ('time_increments', list), ('dx', '<f4'),
                               ('x', type(gpe_propagator.x)), ('parameters', dict)]
                        )


def replace(string_for_replacement):
    """
    For naming conventions, this will remove any decimal points from parameters to avoid strange document types
    Doesn't need to be a string, but only because I am a kind and benevolent god. Please pass a string.
    :param str string_for_replacement: The filename/path which contains periods, such as T=0.5
    :return: A string with all instances of '.' replaced with ',' such as T=0.5 -> T=0,5
    """
    return str(string_for_replacement).replace('.', ',')


@njit
def pulse(pos_grid, width, center):
    """
    Adjustable width Gaussian. Passed as a function for repeated use and readability
    :param pos_grid:
    :param center:
    :param width:
    :return:
    """
    return np.exp(-0.5 * ((pos_grid - center) / width) ** 2)


@njit
def diff_pulse(pos_grid, width, center):
    """
    Derivative of the Gaussian pulse
    :param pos_grid:
    :param center:
    :param width:
    :return:
    """
    return -(pos_grid - center) / (width ** 2) * np.exp(-0.5 * ((pos_grid - center) / width) ** 2)


def paint_potential(x, w, t, sampling_freq):
    """

    :param x:
    :param w:
    :param t:
    :param sampling_freq:
    :return:
    """
    scan = []
    extent = [x[0], x[-1]]
    potential = UnivariateSpline(x, scan, k=3, s=0)
    return potential(x)

class SplitOpTrackingControl(object):
    """
    The second-order split-operator propagator of the 1D Gross–Pitaevskii equation in the coordinate representation
    with the time-dependent Hamiltonian

        H = K(p, t) + U(x, t) + g * abs(wavefunction) ** 2

    """
    def __init__(self, *, x_grid_dim, x_amplitude, v, k, dt, g, track, diff2_track, tracking_potential,
                 diff_tracking_potential, epsilon=1e-7, diff_k=None, diff_v=None, t=0, abs_boundary=1.,
                 fftw_wisdom_fname='fftw.wisdom', **kwargs):
        """
        :param x_grid_dim: the grid size
        :param x_amplitude: the maximum value of the coordinates
        :param v: the potential energy (as a function)
        :param k: the kinetic energy (as a function)
        :param diff_k: the derivative of the potential energy for the Ehrenfest theorem calculations
        :param diff_v: the derivative of the kinetic energy for the Ehrenfest theorem calculations
        :param t: initial value of time
        :param dt: initial time increment
        :param g: the coupling constant
        :param epsilon: relative error tolerance
        :param abs_boundary: absorbing boundary
        :param fftw_wisdom_fname: File name from where the FFT wisdom will be loaded from and saved to
        :param kwargs: ignored
        """

        # saving the properties
        self.x_grid_dim = x_grid_dim
        self.x_amplitude = x_amplitude
        self.v = v
        self.k = k
        self.diff_v = diff_v
        self.t = t
        self.dt = dt
        self.g = g
        self.epsilon = epsilon
        self.abs_boundary = abs_boundary
        self.track = track
        self.diff2_track = diff2_track
        self.tracking_potential = tracking_potential
        self.diff_tracking_potential = diff_tracking_potential


        ####################################################################################################
        #
        #   Initialize Fourier transform for efficient calculations
        #
        ####################################################################################################

        # Load the FFTW wisdom
        try:
            with open(fftw_wisdom_fname, 'rb') as fftw_wisdow:
                pyfftw.import_wisdom(pickle.load(fftw_wisdow))
        except (FileNotFoundError, EOFError):
            pass

        # allocate the array for wave function
        self.wavefunction = pyfftw.empty_aligned(x_grid_dim, dtype=complex)

        # allocate an extra copy for the wavefunction necessary for adaptive time step propagation
        self.wavefunction_next = pyfftw.empty_aligned(self.x_grid_dim, dtype=complex)

        # allocate the array for wave function in momentum representation
        self.wavefunction_next_p = pyfftw.empty_aligned(x_grid_dim, dtype=complex)

        # allocate the array for calculating the momentum representation for the energy evaluation
        self.wavefunction_next_p_ = pyfftw.empty_aligned(x_grid_dim, dtype=complex)

        # parameters for FFT
        self.fft_params = {
            "flags": ('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'),
            "threads": threads,                                       # Removed cpu_count from here
            "planning_timelimit": 60,
        }

        # FFT
        self.fft = pyfftw.FFTW(self.wavefunction_next, self.wavefunction_next_p, **self.fft_params)

        # iFFT
        self.ifft = pyfftw.FFTW(self.wavefunction_next_p, self.wavefunction_next, direction='FFTW_BACKWARD', **self.fft_params)

        # fft for momentum representation
        self.fft_p = pyfftw.FFTW(self.wavefunction_next_p, self.wavefunction_next_p_, **self.fft_params)

        # Save the FFTW wisdom
        with open(fftw_wisdom_fname, 'wb') as fftw_wisdow:
            pickle.dump(pyfftw.export_wisdom(), fftw_wisdow)

        ####################################################################################################
        #
        #   Initialize grids
        #
        ####################################################################################################

        # Check that all attributes were specified
        # make sure self.x_amplitude has a value of power of 2
        assert 2 ** int(np.log2(self.x_grid_dim)) == self.x_grid_dim, \
            "A value of the grid size (x_grid_dim) must be a power of 2"

        # get coordinate step size
        dx = self.dx = 2. * self.x_amplitude / self.x_grid_dim
        # generate coordinate range
        x = self.x = (np.arange(self.x_grid_dim) - self.x_grid_dim / 2) * self.dx

        # generate momentum range as it corresponds to FFT frequencies
        p = self.p = (np.arange(self.x_grid_dim) - self.x_grid_dim / 2) * (np.pi / self.x_amplitude)

        # the relative change estimators for the time adaptive scheme
        self.e_n = self.e_n_1 = self.e_n_2 = 0
        self.previous_dt = 0
        # list of self.dt to monitor how the adaptive step method is working
        self.time_increments = []

        ####################################################################################################
        #
        # Codes for efficient evaluation
        #
        ####################################################################################################

        # Decide whether the potential depends on time
        try:
            v(x, 0)
            time_independent_v = False
        except TypeError:
            time_independent_v = True

        # Decide whether the kinetic energy depends on time
        try:
            k(p, 0)
            time_independent_k = False
        except TypeError:
            time_independent_k = True

        # pre-calculate the absorbing potential and the sequence of alternating signs

        abs_boundary = (abs_boundary if isinstance(abs_boundary, (float, int)) else abs_boundary(x))
        abs_boundary = (-1) ** np.arange(self.wavefunction.size) * abs_boundary

        # Cache the potential if it does not depend on time
        if time_independent_v:
            pre_calculated_v = v(x)  # Test by removing T here
            v = njit(lambda _, __: pre_calculated_v)

        # Cache the kinetic energy if it does not depend on time
        if time_independent_k:
            pre_calculated_k = k(p)  # Test by removing T here
            k = njit(lambda _, __: pre_calculated_k)


        ####################################################################################################
        # Check whether the necessary terms are specified to calculate the first-order Ehrenfest theorems
        ####################################################################################################
        if diff_k and diff_v:

            # Cache the potential if it does not depend on time
            if time_independent_v:
                pre_calculated_diff_v = diff_v(x)
                diff_v = njit(lambda _, __: pre_calculated_diff_v)

            # Cache the kinetic energy if it does not depend on time
            if time_independent_k:
                pre_calculated_diff_k = diff_k(p)
                diff_k = njit(lambda _, __: pre_calculated_diff_k)

            # Get codes for efficiently calculating the Ehrenfest relations

            @njit
            def get_p_average_rhs(density, t):
                return -np.sum(density * (diff_v(x, t) + diff_tracking_potential(x, get_track_param(density, t))))

            self.get_p_average_rhs = get_p_average_rhs

            # The code above is equivalent to
            #self.get_p_average_rhs = njit(lambda density, t: np.sum(density * diff_v(x, t)))

            @njit
            def get_u_average(density, t):
                return np.sum((v(x, t)
                               + tracking_potential(x, get_track_param(density, t))
                               + 0.5 * g * density / dx) * density
                              )

            self.get_u_average = get_u_average

            @njit
            def get_x_average(density):
                return np.sum(x * density)

            self.get_x_average = get_x_average

            @njit
            def get_x_average_rhs(density, t):
                return np.sum(diff_k(p, t) * density)

            self.get_x_average_rhs = get_x_average_rhs

            @njit
            def get_k_average(density, t):
                return np.sum(k(p, t) * density)

            self.get_k_average = get_k_average

            @njit
            def get_p_average(density):
                return np.sum(p * density)

            self.get_p_average = get_p_average

            # since the variable time propagator is used, we record the time when expectation values are calculated
            self.times = []

            # Lists where the expectation values of x and p
            self.x_average = []
            self.p_average = []

            # Lists where the right hand sides of the Ehrenfest theorems for x and p
            self.x_average_rhs = []
            self.p_average_rhs = []

            # List where the expectation value of the Hamiltonian will be calculated
            self.hamiltonian_average = []

            # sequence of alternating signs for getting the wavefunction in the momentum representation
            self.minus = (-1) ** np.arange(self.x_grid_dim)

            # Flag requesting tha the Ehrenfest theorem calculations
            self.is_ehrenfest = True
        else:
            # Since diff_v and diff_k are not specified, we are not going to evaluate the Ehrenfest relations
            self.is_ehrenfest = False

        def get_track_param(density, t):
            return np.sum(density * diff_v(density, t) + get_x_average(density, t) + diff2_track(t))

        self.tracking_param = []

        @njit  # (parallel=True)
        def expV(wavefunction, t, dt):
            """
            function to efficiently evaluate
                wavefunction *= (-1) ** k * exp(-0.5j * dt * v)
            """
            dens = np.abs(wavefunction) ** 2
            e_t = get_track_param(dens, t + 0.5 * dt)
            wavefunction *= abs_boundary * np.exp(
                -0.5j * dt * (v(x, t + 0.5 * dt)
                              + tracking_potential(e_t)
                              + g * dens))
            wavefunction /= linalg.norm(wavefunction) * np.sqrt(dx)
            self.tracking_param.append(e_t)
        self.expV = expV

        @njit  # (parallel=True)
        def expK(wavefunction, t, dt):
            """
            function to efficiently evaluate
                wavefunction *= exp(-1j * dt * k)
            """
            wavefunction *= np.exp(-1j * dt * k(p, t + 0.5 * dt))

        self.expK = expK

    def propagate(self, time_final):
        """
        Time propagate the wave function saved in self.wavefunction
        :param time_final: until what time to propagate the wavefunction
        :return: self.wavefunction
        """
        e_n = self.e_n
        e_n_1 = self.e_n_1
        e_n_2 = self.e_n_2
        previous_dt = self.previous_dt

        # copy the initial condition into the propagation array self.wavefunction_next
        np.copyto(self.wavefunction_next, self.wavefunction)

        while self.t < time_final:

            ############################################################################################################
            #
            #   Adaptive scheme propagator
            #
            ############################################################################################################

            # propagate the wavefunction by a single dt
            self.single_step_propagation(self.dt)

            e_n = relative_diff(self.wavefunction_next, self.wavefunction)

            while e_n > self.epsilon:
                # the error is too high, decrease the time step and propagate with the new time step

                self.dt *= self.epsilon / e_n

                np.copyto(self.wavefunction_next, self.wavefunction)
                self.single_step_propagation(self.dt)

                e_n = relative_diff(self.wavefunction_next, self.wavefunction)

            # accept the current wave function
            np.copyto(self.wavefunction, self.wavefunction_next)

            # save self.dt for monitoring purpose
            self.time_increments.append(self.dt)

            # increment time
            self.t += self.dt

            # calculate the Ehrenfest theorems
            self.get_ehrenfest()

            ############################################################################################################
            #
            #   Update time step via the Evolutionary PID controller
            #
            ############################################################################################################

            # overwrite the zero values of e_n_1 and e_n_2
            previous_dt = (previous_dt if previous_dt else self.dt)
            e_n_1 = (e_n_1 if e_n_1 else e_n)
            e_n_2 = (e_n_2 if e_n_2 else e_n)

            # the adaptive time stepping method from
            #   http://www.mathematik.uni-dortmund.de/~kuzmin/cfdintro/lecture8.pdf
            # self.dt *= (e_n_1 / e_n) ** 0.075 * (self.epsilon / e_n) ** 0.175 * (e_n_1 ** 2 / e_n / e_n_2) ** 0.01

            # the adaptive time stepping method from
            #   https://linkinghub.elsevier.com/retrieve/pii/S0377042705001123
            self.dt *= (self.epsilon ** 2 / e_n / e_n_1 * previous_dt / self.dt) ** (1 / 12.)

            # update the error estimates in order to go next to the next step
            e_n_2, e_n_1 = e_n_1, e_n

        # save the error estimates
        self.previous_dt = previous_dt
        self.e_n = e_n
        self.e_n_1 = e_n_1
        self.e_n_2 = e_n_2

        return self.wavefunction

    def single_step_propagation(self, dt):
        """
        Propagate the wavefunction, saved in self.wavefunction_next, by a single time-step
        :param dt: time-step
        :return: None
        """
        wavefunction = self.wavefunction_next

        # efficiently evaluate
        #   wavefunction *= (-1) ** k * exp(-0.5j * dt * v)
        self.expV(wavefunction, self.t, dt)

        # going to the momentum representation
        wavefunction = self.fft(wavefunction)

        # efficiently evaluate
        #   wavefunction *= exp(-1j * dt * k)
        self.expK(wavefunction, self.t, dt)

        # going back to the coordinate representation
        wavefunction = self.ifft(wavefunction)

        # efficiently evaluate
        #   wavefunction *= (-1) ** k * exp(-0.5j * dt * v)
        self.expV(wavefunction, self.t, dt)

        # normalize
        # this line is equivalent to
        self.wavefunction /= np.sqrt(np.sum(np.abs(self.wavefunction) ** 2) * self.dx)
        # wavefunction /= linalg.norm(wavefunction) * np.sqrt(self.dx)

    def get_ehrenfest(self):
        """
        Calculate observables entering the Ehrenfest theorems
        """
        if self.is_ehrenfest:
            # alias
            density = self.wavefunction_next_p

            # evaluate the coordinate density
            np.abs(self.wavefunction, out=density)
            density *= density
            # normalize
            density /= density.sum()

            # save the current value of <x>
            self.x_average.append(
                self.get_x_average(density.real)
            )

            self.p_average_rhs.append(
                self.get_p_average_rhs(density.real, self.t)
            )

            # save the potential energy
            self.hamiltonian_average.append(
                self.get_u_average(density.real, self.t)
            )

            # calculate density in the momentum representation
            np.copyto(density, self.wavefunction)
            density *= self.minus
            density = self.fft_p(density)

            # get the density in the momentum space
            np.abs(density, out=density)
            density *= density
            # normalize
            density /= density.sum()

            # save the current value of <p>
            self.p_average.append(self.get_p_average(density.real))

            self.x_average_rhs.append(self.get_x_average_rhs(density.real, self.t))

            # add the kinetic energy to get the hamiltonian
            self.hamiltonian_average[-1] += self.get_k_average(density.real, self.t)

            # save the current time
            self.times.append(self.t)
            # print(self.dt)

    def set_wavefunction(self, wavefunc):
        """
        Set the initial wave function
        :param wavefunc: 1D numpy array or function specifying the wave function
        :return: self
        """

        if isinstance(wavefunc, np.ndarray):
            # wavefunction is supplied as an array

            # perform the consistency checks
            assert wavefunc.shape == self.wavefunction.shape, \
                "The grid size does not match with the wave function"

            # make sure the wavefunction is stored as a complex array
            np.copyto(self.wavefunction, wavefunc.astype(complex))

        else:
            self.wavefunction[:] = wavefunc(self.x)

        # normalize
        self.wavefunction /= linalg.norm(self.wavefunction) * np.sqrt(self.dx)

        return self

quantum_system = namedtuple('quantum_system', ['J', 'g', 'V', 'ψ', 'open_bounds'])


def standardize_plots(params=plt.rcParams):
    """
    Updates plot sizes and relevant params so that plots stay consistent
    :param params: rcParams or any other updatable plotting params set
    :return: The Okabe-Ito color scheme with modified Amber for better readability on white backgrounds
    """
    ok = {
        'blue': "#56B4E9",
        'orange': "#E69F00",
        'green': "#009E73",
        'amber': "#F5C710",
        'purple': "#CC79A7",
        'navy': "#0072B2",
        'red': "#D55E00",
        'black': "#000000",
        'grey': "#999999",
        'yellow': "#F0E442",
    }

    plt_params = {
        'figure.figsize': (4, 3),
        'figure.dpi': 300,
        'legend.fontsize': 8,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'axes.prop_cycle': plt.cycler('color', (ok[_] for _ in ok)),
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'lines.linewidth': 2,
    }
    params.update(plt_params)
    return ok


def build_saves_tag(params_list, unique_identifier='GPE', parent_directory='./Archived_Data/GPE'):
    save_tag = unique_identifier
    try:
        os.mkdir(parent_directory)
        print(f'Parent Directory created, saved to: {parent_directory}')
    except:
        FileExistsError
        print(f'Parent directory check passed! \nResults will be available in {parent_directory}\n')
    for _ in params_list:
        save_tag += f'-{_}{params_list[_]}'.replace('.', ',')
    unique_path = os.path.join(parent_directory, save_tag)
    try:
        os.mkdir(unique_path)
        print(f'Simulation Directory "{save_tag}" created')
    except:
        FileExistsError
        print(
            f'WARNING: The directory "{save_tag}" exists! \nYou may be overwriting previous data (; n;)\n')
    path_to_saves = unique_path + '/'
    return path_to_saves, save_tag

def save_hdf5(file_name, path, to_be_saved, save_method='w', group_name=None):
    with h5py.File(path + file_name + '_data.hdf5', mode=save_method) as file:
        if group_name:
            my_group = file.create_group(group_name)
            for _, __ in to_be_saved.items():
                my_group.create_dataset(_, data=__)
        else:
            for _, __ in to_be_saved.items():
                file.create_dataset(_, data=__)

def load_hdf5(file_name, file_path, things_to_grab, group_name=None):
    with h5py.File(file_path + file_name + '_data.hdf5', mode='r') as file:
        retrieval_path = str()
        if group_name:
            retrieval_path += f'{group_name}/'
        unpacked_results = dict()
        for _ in things_to_grab:
            unpacked_results[_] = file[retrieval_path+_][()]
        return unpacked_results
