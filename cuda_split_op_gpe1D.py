import numpy as np
import pyfftw
import _pickle as pickle
from numpy import linalg  # Linear algebra for dense matrix
from numba import njit, jit, cuda
from multiprocessing import cpu_count
import os
from fractions import gcd
from types import MethodType, FunctionType
import cufft


class CudaSplitOpGPE1D(object):
    """
    Second-order split-operator propagator of the quasi-1D Gross-Pitaevskii equation in the coordinate representation
    with the time-dependent Hamiltonian:
        H = K(p, t) + V(x, t) + g|Psi(x,t)|^2
    """

    def __init__(self, *, pos_grid_dim, pos_amp, v, k, dt, g, eps, diff_v=None, diff_k=None, t=0,
                 fftw_wis_name='fftw.wisdom', **kwargs):
        """

        :param pos_grid_dim: Number of Position grid vectors (must be a factor of 2 for fft)
        :param pos_amp:  Max/min value of the position grid
        :param v: potential energy as a function
        :param k: kinetic energy as a function
        :param dt: initial (or fixed) time increment
        :param g: Inter-particle interaction term
        :param eps: relative error tolerance
        :param diff_v: Derivative of potential energy as a function (for checking Ehrenfest theorem verification)
        :param diff_k: Derivative of kinetic energy as a function (for checking Ehrenfest theorem verification)
        :param t: initial value of time
        :param fftw_wis_name: File name from where the FFT wisdom will be loaded/saved
        :param kwargs: Did you pass extra params? If yes... why?
        """

        ############################################# USE EITHER THIS ##################################################
        # Save the grid and functions
        self.pos_grid_dim = pos_grid_dim
        self.pos_amp = pos_amp
        # make sure position grid dimensions are compatible with FFT (has a value of power of 2)
        assert 2 ** int(np.log2(self.pos_grid_dim)) == self.pos_grid_dim, \
            "A value of the grid size (x_grid_dim) must be a power of 2"
        self.v = v
        self.k = k
        self.diff_v = diff_v
        self.t = t
        self.dt = dt
        self.g = g
        self.epsilon = eps

        ################################################ OR THIS #######################################################
        for name, value in kwargs.items():
            # if the value supplied is a function, then dynamically assign it as a method;
            # otherwise bind it a property
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

        # Check that all attributes were specified
        try:
            self.pos_grid_dim
        except AttributeError:
            raise AttributeError("Coordinate grid size (X_gridDIM) was not specified")

        assert 2 ** int(np.log2(self.pos_grid_dim)) == self.pos_grid_dim, \
            "Coordinate grid size (X_gridDIM) must be a power of two"

        try:
            self.pos_amp
        except AttributeError:
            raise AttributeError("Coordinate grid range (X_amplitude) was not specified")

        try:
            self.g
        except AttributeError:
            raise AttributeError("Interaction parameter (g) was not specified")

        try:
            self.v
        except AttributeError:
            raise AttributeError("Potential energy (V) was not specified")

        try:
            self.k
        except AttributeError:
            raise AttributeError("Momentum dependence (K) was not specified")

        try:
            self.dt
        except AttributeError:
            raise AttributeError("Time-step (dt) was not specified")

        try:
            self.t
        except AttributeError:
            print("Warning: Initial time (t) was not specified, thus it is set to zero.")
            self.t = 0.

        # Save the current value of t as the initial time
        kwargs.update(t_initial=self.t)

        self.t = np.float64(self.t)

        ############################################## Generate the Grid ###############################################
        # Coordinate step size and range
        self.dx = dx = 2. * self.pos_amp / self.pos_grid_dim
        self.x = x = self.dx * (np.arrange(self.pos_grid_dim) - self.pos_grid_dim / 2)
        # Generate momentum range as it corresponds to FFT frequencies
        self.p = p = (np.pi / self.pos_amp) * (np.arrange(self.pos_grid_dim - self.pos_grid_dim / 2))

       ################################# Decide if you are using adaptive step or not ##################################
        self.e_n = self.e_n_1 = self.e_n_2 = 0      # Relative change estimators for adaptive time step
        self.prev_dt = 0                            # Set the initial dt as 0 to begin adaptive
        self.time_increments = []                   # list of self.dt to monitor how the adaptive step method is working
