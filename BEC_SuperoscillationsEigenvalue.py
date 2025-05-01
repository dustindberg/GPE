import utlt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.interpolate import interp1d, UnivariateSpline, InterpolatedUnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import lstsq, eigh, norm
from scipy.optimize import curve_fit, nnls, leastsq, minimize, least_squares
from scipy.constants import hbar, elementary_charge, speed_of_light, Boltzmann

from itertools import product, combinations
import copy


from multiprocessing import Pool, cpu_count
from tqdm.notebook import tqdm


from sklearn import linear_model

import glob
import os

from datetime import datetime

threads = 16
os.environ["OMP_NUM_THREADS"] = '{}'.format(threads)
os.environ['NUMEXPR_MAX_THREADS'] = '{}'.format(threads)
os.environ['NUMEXPR_NUM_THREADS'] = '{}'.format(threads)
os.environ['OMP_NUM_THREADS'] = '{}'.format(threads)
os.environ['MKL_NUM_THREADS'] = '{}'.format(threads)

ok = utlt.standardize_plots(plt.rcParams)


def order(number):
    return np.floor(np.log10(number))


def get_wide_plot(parameters=plt.rcParams):
    wide_plt_params = {
        'figure.figsize': (4, 3),
        'figure.dpi': 300,
        'legend.fontsize': 7,
        'axes.labelsize': 10,
        'axes.titlesize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'lines.linewidth': 2,
    }
    parameters.update(wide_plt_params)


def Phi(grid, wave_vector, phase=0, amplitude=1):
    return amplitude * np.sin(wave_vector * grid + phase)


def Phi_transform(time, frequency, amplitude_1, amplitude_2):
    return amplitude_1 * np.sin(frequency * time) + amplitude_2 * np.cos(frequency * time)


def Phi_combine(A, B, wave_vector, grid):
    amplitude = np.sqrt(A ** 2 + B ** 2)
    phase = np.angle(A + 1j * B)
    single_field = amplitude * np.cos(wave_vector * grid - phase)
    return amplitude, phase, single_field


def Phi_combine_nofield(A, B):
    amplitude = np.sqrt(A ** 2 + B ** 2)
    phase = -np.angle(A + 1j * B) + np.pi/2
    return amplitude, phase


"""
Call an instance of BEC to get physical params. This way wavelengths may be set precisely for dimensionless units
"""
# Frequency of targeted atomic transition goes here.
# D1 transition (in nm)
principal_wavelength = 794.978851e-9
# D2 transition (in nm)
#principal_wavelength = 780.251210e-9
# Resonance frequency of the atomic transition in units of 2π Hz to keep units consistent
ω_0 = 2 * np.pi * speed_of_light / principal_wavelength
# Model parameters for dimensionless units.
ω_x = 5 * 2 * np.pi  # From experimental cigar length params
ω_yz = 500 * 2 * np.pi
N = 2e3
print(f'Resonance Frequency is: 2π * {(ω_0 / ( 2* np.pi)) * 1e-12} THz')


atom_params = dict(
    atom='R87',
    kicked=False,
    omega_x=ω_x,
    omega_y=ω_yz,
    omega_z=ω_yz,
    number_of_atoms=N,
)
gpe = utlt.BEC(**atom_params)
g = gpe.g

"""
Define default params and use the default utility for saving
"""
# Number of pulses
n = 2
# Determine the error tolerance
eps = 1e-11
# Now declare the pulse parameters, starting with the wavelength closest to the transition
sweep_floor = 788
sweep_max = 794
# Wavelengths for sweep in nm
step = 0.5
λ_sweep = np.arange(sweep_floor, sweep_max+step, step)
λ_target = 0.333333333 * 780  #  0.33333 * sweep_floor
# Wave vectors for calculation in dimensionless units
k = 2 * np.pi * (gpe.dimless_x(λ_sweep, -9) ** -1)
# Flip the k array as it's passed, since files index from smallest to largest
k1k2 = list(combinations(np.flip(k), 2))
# Setting up the grid in dimensionless units
window = 0.5 * gpe.dimless_x(λ_target, -9)  # given as -λ_target/2 to λ_target/2
x_lim = gpe.dimless_x(0.5 * gpe.L_x)
x_res = 2 ** 20
dx = 2 * x_lim / x_res

x = np.arange(-window, window, dx)
x_full = np.arange(-x_lim, x_lim, dx)

# Rounding order for plots so names are as short as they can be
ro = int(np.abs(order(k[0] - k[1])))

def f2V_scaling(wavelength):
    # It is important to pass the wavelength in meters!
    frequency = 2 * np.pi * speed_of_light / wavelength
    gfo = order(frequency)
    frequency /= 10 ** gfo
    ω0_orderless = ω_0 / 10 ** gfo
    α = 1.602176634 ** 2 / 1.4431606    # the order of magnitude, 10^-13, will be added in the next line.
    return (0.5 * α / (ω0_orderless ** 2 - frequency ** 2)), (-13 - (gfo * 2))



named_params = {
    'tgt_wavelength': round(λ_target, 2),
    'max': round(sweep_max, ro),
    'min': round(sweep_floor, ro),
    'pulses': n,
    'swept': len(k1k2),
}
params = copy.deepcopy(named_params)
file_path, file_name = utlt.build_saves_tag(params_list=named_params,
                                            unique_identifier='D1-Sweep-Eigen',
                                            parent_directory='Archive_Data/Superoscilations')


def single_run_case(k1, k2):
    ok = utlt.standardize_plots(plt.rcParams)
    dif = k2-k1
    plot_name = f'k{round(k1, ro):.3f}+{round(dif, ro):02.3f}'.replace('.', ',')
    plot_path = file_path + plot_name + '/'
    try:
        os.mkdir(plot_path)
    except:
        FileExistsError

    kays = np.array([k1, k2])
    fields = np.array([
        Phi(x, i, j) for i, j in product(kays, [0, np.pi / 2])
    ]).T
    fields_full = np.array([
        Phi(x_full, i, j) for i, j in product(kays, [0, np.pi / 2])
    ]).T

    S = fields.T @ fields

    vals, vecs = eigh(S)
    eig_fields = [fields @ vecs[:, _] for _ in range(len(vecs[:]))]
    eig_fields_full = [fields_full @ vecs[:, _] for _ in range(len(vecs[:]))]

    lk_pulse = Phi(x, max(k1, k2))
    lk_pulse_full = Phi(x_full, max(k1, k2))

    color1 = ok['blue']
    color2 = ok['purple']
    color3 = ok['green']

    get_wide_plot(plt.rcParams)

    for _ in range(n * 2):
        fig, (ax1, ax1__) = plt.subplots(1, 2, figsize=(4, 2))
        eig_vector = vecs[:, _]
        eig_info = np.array(
            [Phi_combine_nofield(eig_vector[__], eig_vector[__+1]) for __ in np.arange(0, len(eig_vector), 2)])
        amps = eig_info[:, 0]
        phases = eig_info[:, 1]

        ax1.set_ylabel('$V$ (arb. u.)', color=color1)
        ax1.plot(x, np.abs(eig_fields[_]) ** 2, color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_xlim(x[0], x[-1])

        ax1.set_xlabel('$x {\\rm (arb. u.)}$')

        ax2 = ax1.twinx()

        ax2.set_ylabel('Shortest $\lambda$ $V$ (arb.u)', color=color2)
        ax2.plot(x, lk_pulse ** 2, color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax1.set_zorder(1)
        ax1.patch.set_visible(False)

        # plot pulses
        ax1 = ax1__

        ax1.bar(np.arange(1, n + 1, 1), amps, color=color3)
        ax1.set_xticks(np.arange(1, n + 1, 1))
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('$j$')
        ax1.set_ylabel('$a_{j}$ (arb. u.)')

        fig.tight_layout()
        plt.savefig(plot_path + plot_name + f'-solution{_}_combined.jpg')

        ok = utlt.standardize_plots(plt.rcParams)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.ylabel('$V$ (arb.u)', color=color1)
        plt.plot(x_full, eig_fields_full[_] ** 2, color=color1)
        plt.tick_params(axis='y', labelcolor=color1)
        ax1.set_xlabel('$x$ (arb.u)')

        ax2 = ax1.twinx()

        ax2.plot(x_full, lk_pulse_full ** 2, color=color2)
        ax2.set_ylabel('Shortest $\lambda$ $V$ (arb. u.)', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax1.set_zorder(1)
        ax1.patch.set_visible(False)
        fig.tight_layout()
        plt.savefig(plot_path + plot_name + f'-solution{_}_zoomout.jpg')

        plt.close('all')


for _ in k1k2:
    single_run_case(_[0], _[1])
