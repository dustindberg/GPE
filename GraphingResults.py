from numba import njit
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
from scipy.constants import hbar, Boltzmann
from split_op_gpe1D import SplitOpGPE1D, imag_time_gpe1D    # class for the split operator propagation
import pickle as pickle
from multiprocessing import Pool
import os

########################################################################################################################
#
# Section for checking individual parameters each time a new trial is loaded
#
########################################################################################################################

# Begin by loading the pickle file
location = './Archive_Data/Paper_Collection/'
method = 'Method1' # Input which method you are generating plots for
identifier = 'REDO_Kick6,5_Sigma8,5_Height45,0_Vo0,5_T11,0' # Paste from pickle file
foldername = identifier + '/'
path = location + foldername
filename = identifier + '.pickle'
savespath = 'Archive_Data/Paper_Collection/Generated_Plots/' + method + '/'

with open(path + filename, "rb") as f:
        qsys, qsys_flipped = pickle.load(f)


# Load Params from Pickle
x = qsys['gpe']['x']
size = qsys['gpe']['x'].size
x_cut = int(0.64 * size)
x_cut_flipped = int(0.36 * size)
dx = qsys['gpe']['dx']
x_amplitude = qsys['parameters']['x_amplitude']
x_grid_dim = qsys['parameters']['x_grid_dim']
times = qsys['parameters']['times']
N = qsys['parameters']['N']
extent = qsys['gpe']['extent']
init_gs_energy = qsys['gpe']['hamiltonian_average'][0]


# Factors which need to be checked, but aren't stored in pickle
scale = 1.0
trap_height = 400
height_asymmetric = 45.0
v_0 = 0.5
cooling_offset = 65.0
propagation_dt = 0.003
eps = 0.0005
sigma = 8.5
delta = 2. * (sigma ** 2)
# IMPORTANT: Check if you are using a time cutoff before plotting anything
time_start = times[0] # Use times[0] if your simulation does not need trimming
time_end = times[-1]  # Use times[-1] if your simulation does not need trimming
time_init = int((time_start/times[-1]) * len(times))
time_cutoff = int((time_end / times[-1]) * len(times))

gpe_l2r_wavefunction = qsys['gpe']['wavefunctions'][:time_cutoff][:]
gpe_r2l_wavefunction = qsys_flipped['gpe']['wavefunctions'][:time_cutoff][:]
schr_l2r_wavefunction = qsys['schrodinger']['wavefunctions'][:time_cutoff][:]
schr_r2l_wavefunction = qsys_flipped['schrodinger']['wavefunctions'][:time_cutoff][:]
mu_l2r = qsys['mu']
mu_r2l = qsys_flipped['mu']
init_state_l2r = qsys['init_state']
init_state_r2l = qsys_flipped['init_state']
T = np.array(times[time_init:time_cutoff])


# Storing functions
@njit (parallel=True)
def v(x):
    """
    Potential energy
    """
    # method 1 potential
    return height_asymmetric * (
        np.exp(-(x + 15.) ** 2 / delta)
        + 0.70 * np.exp(-x ** 2 / delta)
        + 0.30 * np.exp(-(x - 15.) ** 2 / delta)
    )

    ## method 2 potenial
    #return trap_height - height_asymmetric * (
    #    np.exp(-(x + 45.) ** 2 / delta)
    #    + np.exp(-(x + 30.) ** 2 / delta)
    #    + 0.85 * np.exp(-(x + 15.) ** 2 / delta)
    #    + 0.95 * np.exp(-(x - 0.3) ** 2 / delta)
    #    + 0.85 * np.exp(-(x - 15.) ** 2 / delta)
    #    + np.exp(-(x - 30.) ** 2 / delta)
    #    + np.exp(-(x - 45.) ** 2 / delta)
    #)


@njit
def initial_trap(x, offset):
    """
    Trapping potential to get the initial state
    :param x:
    :return:
    """
    return v_0 * (x + offset) ** 2

# define beams for plotting
def beam_1(x):
    return trap_height - height_asymmetric * np.exp(-(x + 45) ** 2 / delta)
def beam_2(x):
    return trap_height - height_asymmetric * np.exp(-(x - 45) ** 2 / delta)
def beam_3(x):
    return trap_height - height_asymmetric * np.exp(-(x + 30) ** 2 / delta)
def beam_4(x):
    return trap_height - height_asymmetric * np.exp(-(x - 30) ** 2 / delta)
def beam_5(x):
    return trap_height - height_asymmetric * 0.85 * np.exp(-(x + 15) ** 2 / delta)
def beam_6(x):
    return trap_height - height_asymmetric * 0.95 * np.exp(-(x - 0.3) ** 2 / delta)
def beam_7(x):
    return trap_height - height_asymmetric * 0.85 * np.exp(-(x - 15) ** 2 / delta)

########################################################################################################################
#
# Now determine physical parameters which are needed for conversion to physical units
#
########################################################################################################################

# Rubidium-87 properties
a_0 = 5.291772109e-11               # Bohr Radius in meters
m = 1.4431609e-25                   # Calculated mass of 87Rb in kg
a_s = 100 * a_0                     # Background scattering length of 87Rb in meters
omeg_x = 50 * 2 * np.pi             # Harmonic oscillation in the x-axis in Hz
omeg_y = 500 * 2 * np.pi            # Harmonic oscillation in the y-axis in Hz
omeg_z = 500 * 2 * np.pi            # Harmonic oscillation in the z-axis in Hz
omeg_cooling = 450 * 2 * np.pi      # Harmonic oscillation for the trapping potential in Hz
L_x = np.sqrt(hbar / (m * omeg_x))  # Characteristic length in the x-direction in meters
L_y = np.sqrt(hbar / (m * omeg_y))  # Characteristic length in the y-direction in meters
L_z = np.sqrt(hbar / (m * omeg_z))  # Characteristic length in the z-direction in meters
g = 2 * N * L_x * m * scale * a_s * np.sqrt(omeg_y * omeg_z) / hbar     # Dimensionless interaction parameter
chem_potential = (2 * v_0) ** (1/3) * (3 * g / 8) ** (2/3)  # Calculated chemical potential

# Conversion factors to plot in physical units
L_xmum = np.sqrt(hbar / (m * omeg_x)) * 1e6     # Characteristic length in the x-direction in meters
L_ymum = np.sqrt(hbar / (m * omeg_y)) * 1e6     # Characteristic length in the y-direction in meters
L_zmum = np.sqrt(hbar / (m * omeg_z)) * 1e6     # Characteristic length in the z-direction in meters
ms_conv = 1. / omeg_x * 1e3                   # Converts characteristic time into milliseconds
energy_conv = hbar * omeg_x                     # Converts dimensionless energy terms to Joules
muK_conv = energy_conv * (1e6 / Boltzmann)      # Converts dimensionless terms to microKelvin
nK_conv = energy_conv * (1e9 / Boltzmann)       # Converts dimensionless terms to nanoKelvin
specvol_mum = (L_xmum * L_ymum * L_zmum) / N    # Converts dimensionless spacial terms into micrometers^3 per particle
dens_conv = 1. / (L_xmum * L_ymum * L_zmum)     # Calculated version of density unit conversion


@njit
def Energy_muKelvin(v):
        """"
        The potential energy with corrected units microKelvin
        """
        return v * muK_conv

@njit
def Energy_nanoKelvin(v):
    """
    The initial trap with units of nanoKelvin
    :param v: The initial trap
    :return: The initial trap with corrected units
    """
    return v * nK_conv


# Convert dimensionless parameters to physical units
t_ms = T * ms_conv
x_mum = x * L_xmum
v_nK = Energy_nanoKelvin(v(x))
init_trap_left_nk = Energy_nanoKelvin(initial_trap(x, cooling_offset))
init_trap_right_nk = Energy_nanoKelvin(initial_trap(x, (-1 * cooling_offset)))
mu_l2r_nK = mu_l2r * nK_conv
mu_r2l_nK = mu_r2l * nK_conv
g_units = g * nK_conv * 2 * np.pi / dens_conv  # Converts dimensionless interaction parameter to nk/density

# Convert the densities into physical units
extent_units = [extent[0] * L_xmum, extent[1] * L_xmum, extent[2] * ms_conv, T[-1] * ms_conv]
aspect = (extent_units[1] - extent_units[0]) / (extent_units[-1] - extent_units[-2])


########################################################################################################################
#
# Declare Plotting Functions
#
########################################################################################################################

fignum = 1 # Used to declare multiple figures
plt_params = {'legend.fontsize': 'xx-large',
         'figure.figsize': (8, 7),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
plt.rcParams.update(plt_params)


def plot_potential(fignum, potential, init_energy, position):
    """
    Plots the potential in physical units
    :param fignum: Tracks the figure number so plots may be rendered simultaneously
    :param v: The potential in dimensional units
    :param init_energy: The initial ground state energy
    :param position: The position grid
    :return: The next figure number
    """
    x_line = [position[0],position[-1]]
    en_line = [init_energy, init_energy]
    plt.figure(fignum)
    fignum += 1
    plt.plot(x_line, en_line, color='r')
    plt.plot(position, potential,color='k')
    plt.plot(position, Energy_muKelvin(beam_1(x)), '--')
    plt.plot(position, Energy_muKelvin(beam_2(x)), '--')
    plt.plot(position, Energy_muKelvin(beam_3(x)), '--')
    plt.plot(position, Energy_muKelvin(beam_4(x)), '--')
    plt.plot(position, Energy_muKelvin(beam_5(x)), '--')
    plt.plot(position, Energy_muKelvin(beam_6(x)), '--')
    plt.plot(position, Energy_muKelvin(beam_7(x)), '--')
    plt.fill_between(
        position[x_cut:],
        potential[x_cut:],
        potential.max()+.50,
        facecolor="b",
            color='b',
        alpha=0.2
        )
    plt.fill_between(
        position[:x_cut_flipped],
        potential[:x_cut_flipped],
        potential.max()+.50,
        facecolor="orange",
            color='orange',
        alpha=0.2
        )
    plt.xlim(position[0], position[-1])
    plt.ylim(-.05, potential.max()+.20)
    plt.xlabel('$x$ ($\mu$m)')
    plt.ylabel('$V$ ($\mu$K)')
    plt.tight_layout()
    plt.savefig(savespath + method + 'potential' + '.pdf')
    return fignum


def plot_density(fignum, wave_title, wave):
    """
    Plots density evolution of a wavefunction
    :param fignum: Tracks the figure number so plots may be rendered simultaneously
    :param wave_title: Name of equation used to predict wave; either GPE or Schrodinger
    :param wave: The propagating wavefunction
    :return:
    """
    plt.figure(fignum)
    fignum+=1
    title = wave_title + ' Evolution'
    # plt.title(title)
    plt.imshow(
        np.abs(wave) ** 2,
        # some plotting parameters
        origin='lower',
        extent=extent_units,
        aspect=aspect,
        norm=SymLogNorm(vmin=1e-8, vmax=1., linthresh=1e-15)
    )
    plt.xlabel('Position ($\mu$m)')
    plt.ylabel('Time (ms)')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(savespath + method + title + '.pdf')
    return fignum


def plot_probability(fignum, wave_title, wave_l2r, wave_r2l, time):
    """
    Plots overlapping tunnelling probabilities for Left-to-Right and Right-to-Left wavefunctions
    :param fignum: Tracks the figure number so plots may be rendered simultaneously
    :param wave_title: Name of equation used to predict wave; either GPE or Schrodinger
    :param wave_l2r: Input the wavefunction incident from the left travelling to the right
    :param wave_r2l: Input the wavefunction incident from the right travelling to the left
    :param time: Input the time array to plot over time
    :return: Returns the figure number so that it is updated for each new plot
    """
    plt.figure(fignum)
    fignum+=1
    T_L = np.sum(np.abs(wave_l2r)[time_init:, x_cut:] ** 2, axis=1) * dx
    T_R = np.sum(np.abs(wave_r2l)[time_init:, :x_cut_flipped] ** 2, axis=1) * dx
    title = wave_title + ' Tunnelling Probability'
    # plt.title(title)
    plt.plot(
        time,
        T_L,
        label = 'Left to Right ' + wave_title
    )
    plt.plot(
        time,
        T_R,
        label='Right to Left ' + wave_title
    )
    plt.legend(loc='best')
    plt.xlim(time_start * ms_conv, time_end * ms_conv)
    plt.xlabel('Time (ms)')
    plt.ylabel("Tunnelling Probability")
    plt.tight_layout()
    plt.savefig(savespath + method + title + '.pdf')
    return fignum


def plot_probability_logscale(fignum, wave_title, wave_l2r, wave_r2l, time):
    """
    Plots overlapping log-scale tunnelling probabilities for Left-to-Right and Right-to-Left wavefunctions
    :param fignum: Tracks the figure number so plots may be rendered simultaneously
    :param wave_title: Name of equation used to predict wave; either GPE or Schrodinger
    :param wave_l2r: Input the wavefunction incident from the left travelling to the right
    :param wave_r2l: Input the wavefunction incident from the right travelling to the left
    :param time: Input the time array to plot over time
    :return: Returns the figure number so that it is updated for each new plot
    """
    # Creates a y-axis log scale plot
    plt.figure(fignum)
    fignum+=1
    T_L = np.sum(np.abs(wave_l2r)[time_init:, x_cut:] ** 2, axis=1) * dx
    T_R = np.sum(np.abs(wave_r2l)[time_init:, :x_cut_flipped] ** 2, axis=1) * dx
    title = wave_title + ' Tunnelling Probability Log Scale'
    # plt.title(title)
    plt.semilogy(
        time,
        T_L,
        label = 'Left to Right ' + wave_title
    )
    plt.semilogy(
        time,
        T_R,
        label='Right to Left ' + wave_title
    )
    plt.legend(loc='best')
    plt.xlim(time_start * ms_conv, time_end * ms_conv)
    plt.xlabel('Time (ms)')
    plt.ylabel("Log Scale Tunnelling Probability")
    plt.tight_layout()
    plt.savefig(savespath + title + identifier + '.pdf')
    return fignum


def plot_relativediff(fignum, wave_title, wave_l2r, wave_r2l, time):
    """
    Plots the percentage differance of tunneling probability
    :param fignum: Tracks the figure number so plots may be rendered simultaneously
    :param wave_title: Name of equation used to predict wave; either GPE or Schrodinger
    :param wave_l2r: Input the wavefunction incident from the left travelling to the right
    :param wave_r2l: Input the wavefunction incident from the right travelling to the left
    :param time: Input the time array to plot over time
    :return: Returns the figure number so that it is updated for each new plot
    """
    plt.figure(fignum)
    fignum+=1
    title = wave_title + ' Relative Difference'
    # plt.title(title)
    T_L = np.sum(np.abs(wave_l2r)[time_init:, x_cut:] ** 2, axis=1) * dx
    T_R = np.sum(np.abs(wave_r2l)[time_init:, :x_cut_flipped] ** 2, axis=1) * dx
    d_r = 100 * np.abs(T_L - T_R)/((T_L + T_R)/2)
    plt.plot(
        time, d_r
    )
    plt.xlabel('Time (ms)')
    plt.ylabel("$d_r$ %")
    plt.tight_layout()
    plt.savefig(savespath + title + identifier + '.pdf')
    print(wave_title + 'Final Relative Probability Difference = ' +
          str(np.abs(T_L[-1] - T_R[-1])/((T_L[-1] + T_R[-1])/2)))
    return fignum


def plot_relativediff_compare(fignum, gpe_l2r, grp_r2l, schr_l2r, schr_r2l, time):
    """
    Plots the percentage differance of tunneling probability
    :param fignum: Tracks the figure number so plots may be rendered simultaneously
    :param gpe_l2r: Input the GPE wavefunction incident from the left travelling to the right
    :param grp_r2l: Input the GPE wavefunction incident from the right travelling to the left
    :param schr_l2r: Input the Schrodinger wavefunction incident from the left travelling to the right
    :param schr_r2l: Input the Schrodinger wavefunction incident from the right travelling to the left
    :param time: Input the time array to plot over time
    :return: Returns the figure number so that it is updated for each new plot
    """
    plt.figure(fignum)
    fignum+=1
    title = 'Relative Difference Comparison'
    # plt.title(title)
    T_L_GPE = np.sum(np.abs(gpe_l2r)[time_init:, x_cut:] ** 2, axis=1) * dx
    T_R_GPE = np.sum(np.abs(grp_r2l)[time_init:, :x_cut_flipped] ** 2, axis=1) * dx
    d_r_GPE = 100 * np.abs(T_L_GPE - T_R_GPE)/((T_L_GPE + T_R_GPE)/2)
    T_L_schr = np.sum(np.abs(schr_l2r)[time_init:, x_cut:] ** 2, axis=1) * dx
    T_R_schr = np.sum(np.abs(schr_r2l)[time_init:, :x_cut_flipped] ** 2, axis=1) * dx
    d_r_schr = 100 * np.abs(T_L_schr - T_R_schr)/((T_L_schr + T_R_schr)/2)
    plt.plot(
        time, d_r_GPE, label='GPE'
    )
    plt.plot(
        time, d_r_schr, label='Schrodinger'
    )
    plt.xlabel('Time (ms)')
    plt.ylabel("$d_r$ %")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(savespath + title + identifier + '.pdf')
    return fignum


def plot_ThomasFermiTest(fignum, wave_title, position, mu, v, g, init_state, center):
    """
    Plots the Thomas-Fermi Approximation, which verifies cooling of our initial state
    :param fignum: Tracks the figure number for simultaneous plot generation
    :param wave_title: Tracks if you are testing left to right or right to left GPE
    :param position: The position grid, named to avoid overlap of "x"
    :param mu: The calculated chemical potential
    :param v: The potential energy for the initial trap
    :param g: Interaction parameter
    :param init_state: The initial state of the wavefunction before propagation
    :return: A plot of the Thomas-Fermi approximation's left and right hand side
    """
    plt.figure(fignum)
    fignum+=1
    title = wave_title
    # plt.title(title)
    y = (mu - v) / g
    lhs = (np.abs(init_state) ** 2) / L_xmum
    print('Pre-norm LHS density normalized to: ' + str(np.sum(lhs) * dx * L_xmum) + '\n')
    lhs_norm = np.sum(lhs) * dx * L_xmum
    lhs /= lhs_norm
    rhs = y * (y > 0) / L_xmum
    print('Pre-norm RHS density normalized to: ' + str(np.sum(rhs) * dx * L_xmum) + '\n')
    rhs_norm = np.sum(rhs) * dx * L_xmum
    rhs /= rhs_norm
    plt.plot(
        position, lhs, label='GPE'
    )
    plt.plot(
        position, rhs, label='Thomas Fermi'
    )
    plt.xlim(center - 30, center + 30)
    plt.xlabel('$x (\mu$m)')
    plt.ylabel('Density ($\mu$m)${}^{-3}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(savespath + title + identifier + '.pdf')
    print('RHS density normalized to: ' + str(np.sum(rhs) * dx * L_xmum) + '\n')
    print('LHS density normalized to: ' + str(np.sum(lhs) * dx * L_xmum) + '\n')
    print(mu)

    plt.figure(fignum)
    fignum+=1
    title = wave_title + '_LogScale'
    # plt.title(title)
    y = (mu - v) / g
    lhs = (np.abs(init_state) ** 2)
    rhs = y * (y > 0)
    plt.semilogy(
        position, lhs, label='GPE'
    )
    plt.semilogy(
        position, rhs, label='Thomas Fermi'
    )
    plt.xlim(center - 30, center + 30)
    plt.xlabel('$x (\mu$m)')
    plt.ylabel('Density ($\mu$m)^(-3)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(savespath + title + identifier + '.pdf')


    return fignum



########################################################################################################################
#
# Choose what to plot
#
########################################################################################################################

## For plotting the potential
#fignum = plot_potential(fignum, Energy_muKelvin(v(x)), Energy_muKelvin(init_gs_energy), x_mum)
## For plotting Density
#fignum = plot_density(fignum, 'GPE Left to Right', gpe_l2r_wavefunction)
#fignum = plot_density(fignum, 'GPE Right to Left', gpe_r2l_wavefunction)
#fignum = plot_density(fignum, 'Schrodinger Left to Right', schr_l2r_wavefunction)
#fignum = plot_density(fignum, 'Schrodinger Right to Left', schr_r2l_wavefunction)
## For plotting tunnelling probability
#fignum = plot_probability(fignum, 'GPE', gpe_l2r_wavefunction, gpe_r2l_wavefunction, t_ms)
#fignum = plot_probability(fignum, 'Schrodinger', schr_l2r_wavefunction, schr_r2l_wavefunction, t_ms)
## For plotting relative difference
#fignum = plot_relativediff(fignum, 'GPE', gpe_l2r_wavefunction, gpe_r2l_wavefunction, t_ms)
#fignum = plot_relativediff(fignum, 'Schrodinger', schr_l2r_wavefunction, schr_r2l_wavefunction, t_ms)
#fignum  = plot_relativediff_compare(fignum, gpe_l2r_wavefunction, gpe_r2l_wavefunction, schr_l2r_wavefunction,
#                                   schr_r2l_wavefunction, t_ms)
# For plotting Thomas-Fermi Approximation test
fignum = plot_ThomasFermiTest(fignum, 'Thomas-Fermi Left to Right', x_mum, mu_l2r,
                              initial_trap(x, cooling_offset), g, init_state_l2r, -1 * cooling_offset * L_xmum)
fignum = plot_ThomasFermiTest(fignum, 'Thomas-Fermi Right to Left', x_mum, mu_r2l_nK,
                             init_trap_right_nk, g * nK_conv, init_state_r2l, cooling_offset * L_xmum)

plt.show()
