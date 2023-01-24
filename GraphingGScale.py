from numba import njit
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
import numpy as np
from scipy.constants import hbar, Boltzmann
from scipy.interpolate import UnivariateSpline
from split_op_gpe1D import SplitOpGPE1D, imag_time_gpe1D
import _pickle as pickle
from multiprocessing import Pool
import os

# Define physical parameters
a_0 = 5.291772109e-11                             # Bohr Radius in meters

# Rubidium-87 properties
m = 1.4431609e-25                                 # Calculated mass of 87Rb in kg
a_s = 75 * a_0                                   # Background scattering length of 87Rb in meters

# Potassium-41 properties
# m= 6.80187119e-26                                # Calculated mass of 41K in kg
# a_s = 65.42 * a_0                                # Background scattering length of 41K in meters

# Experiment parameters
N = 1e4                                           # Number of particles
omeg_x = 50 * 2 * np.pi                           # Harmonic oscillation in the x-axis in Hz
omeg_y = 500 * 2 * np.pi                          # Harmonic oscillation in the y-axis in Hz
omeg_z = 500 * 2 * np.pi                          # Harmonic oscillation in the z-axis in Hz
omeg_cooling = 450 * 2 * np.pi                    # Harmonic oscillation for the trapping potential in Hz
scale = 0.05                                       # Scaling factor for the interaction parameter

# Parameters calculated by Python
L_x = np.sqrt(hbar / (m * omeg_x))                # Characteristic length in the x-direction in meters
L_y = np.sqrt(hbar / (m * omeg_y))                # Characteristic length in the y-direction in meters
L_z = np.sqrt(hbar / (m * omeg_z))                # Characteristic length in the z-direction in meters
# Dimensionless interaction parameter
g = 2 * N * L_x * m * scale * a_s * np.sqrt(omeg_y * omeg_z) / hbar

# Conversion factors to plot in physical units
L_xmum = np.sqrt(hbar / (m * omeg_x)) * 1e6       # Characteristic length in the x-direction in meters
L_ymum = np.sqrt(hbar / (m * omeg_y)) * 1e6       # Characteristic length in the y-direction in meters
L_zmum = np.sqrt(hbar / (m * omeg_z)) * 1e6       # Characteristic length in the z-direction in meters
time_conv = 1. / omeg_x * 1e3                     # Converts characteristic time into milliseconds
energy_conv = hbar * omeg_x                       # Converts dimensionless energy terms to Joules
muK_conv = energy_conv * (1e6 / Boltzmann)        # Converts Joule terms to microKelvin
nK_conv = energy_conv * (1e9 / Boltzmann)         # Converts Joule terms to nanoKelvin
specvol_mum = (L_xmum * L_ymum * L_zmum) / N      # Converts dimensionless spacial terms into micrometers^3 per particle
dens_conv = 1. / (L_xmum * L_ymum * L_zmum)       # Calculated version of density unit conversion

# Parameters for computation
propagation_dt = 3e-3
eps = 0.005
height_asymmetric = 300.                           # Height parameter of asymmetric barrier
sigma = 1.5
delta = 2. * (sigma ** 2)
v_0 = 0.5                                         # Coefficient for the trapping potential
peak_offset = sigma*np.sqrt(2*np.log(2))           # 0.5*FWHM = sigma*sqrt(2log(2))
cooling_offset = 100.                              # Center offset for cooling potential
kick = 22.

Glist = []
Glist_fine = []
Glist_last_fine = []
Tnormdiff = []
Tnormdiff_fine = []
Tprob = []
L2R_TP = []
R2L_TP = []

Energy_list = []
Init_Energy_list = []

fignum = 1                      # Declare starting figure number

T = 3.5
times = np.linspace(0, T, 500)
t_ms = np.array(times) * time_conv
x_amplitude = 100.              # Set the range for calculation
x_grid_dim = 16 * 1024
start_scale = 0.01


########################################################################################################################
#
# Load the files to be evaluated
#
########################################################################################################################

def Replace(str1):
    str1 = str1.replace('.', ',')
    return str1


parent_dir = "Archive_Data/Diode_Runs_Small/"
scale = start_scale

# plot the rough points
# for i in range(1,10):
#     stri = str(float(i))
#     if len(stri) < 4:
#         stri += '0'
#     tag = 'Diode_Kick' + str(kick) + '_Delta' + str(delta) + '_Gscale' + stri + '_Height' + str(height_asymmetric) + 'Eps0,005'
#     filename = Replace(tag)
#     path = os.path.join(parent_dir, filename)
#     with open(path + '/' + filename + ".pickle", "rb") as f:
#         qsys, qsys_flipped = pickle.load(f)
#     dx = qsys['gpe']['dx']
#     size = qsys['gpe']['x'].size
#     # These are cuts such that we observe the behavior about the initial location of the wave
#     x_cut = int(0.605 * size)
#     x_cut_flipped = int(0.395 * size)
#     tprob = np.sum(np.abs(qsys['gpe']['wavefunctions'])[:, x_cut:] ** 2, axis=1)[200:] * dx
#     tprob_flipped = np.sum(np.abs(qsys_flipped['gpe']['wavefunctions'])[:, :x_cut_flipped] ** 2, axis=1)[200:] * dx
#     normdiff = np.linalg.norm(tprob_flipped - tprob)
#     Tnormdiff.append(normdiff)
#     Glist.append(i)


scale = start_scale
# Plot the fine points
for j in range(0, 100):
    strscale = str(scale)
    if len(strscale) > 4:
        strscale = str(round(scale*100)/100)
    if len(strscale) < 4:
        strscale += '0'
    tag = 'GPE_Diode_Kick' + str(kick) + '_Delta' + str(delta) + '_Gscale' + strscale\
          + '_Height' + str(height_asymmetric) + '_Eps0,005'
    filename = Replace(tag)
    path = os.path.join(parent_dir, filename)
    with open(path + '/' + filename + ".pickle", "rb") as f:
        qsys, qsys_flipped = pickle.load(f)
    dx = qsys['gpe']['dx']
    size = qsys['gpe']['x'].size
    # These are cuts such that we observe the behavior about the initial location of the wave
    x_cut = int(0.605 * size)
    x_cut_flipped = int(0.395 * size)
    tprob = np.sum(np.abs(qsys['gpe']['wavefunctions'])[:, x_cut:] ** 2, axis=1)[200:] * dx
    tprob_flipped = np.sum(np.abs(qsys_flipped['gpe']['wavefunctions'])[:, :x_cut_flipped] ** 2, axis=1)[200:] * dx
    tprob_last = np.sum(np.abs(qsys['gpe']['wavefunctions'])[:, x_cut:] ** 2, axis=1)[-1] * dx
    tprob_last_flipped = np.sum(np.abs(qsys_flipped['gpe']['wavefunctions'])[:, :x_cut_flipped] ** 2, axis=1)[-1] * dx
    normdiff = 100 * np.linalg.norm(tprob_flipped - tprob)/np.linalg.norm(tprob_flipped + tprob)
    normdiff_last = 100 * (tprob_last_flipped - tprob_last)/(tprob_last + tprob_last_flipped)
    Tnormdiff_fine.append(normdiff_last)
    L2R_TP.append(tprob_last)
    R2L_TP.append(tprob_last_flipped)
    Glist_fine.append(scale*100) #*a_s*10**9))
    Glist_last_fine.append(scale*100) #*a_s)
    Energy_list.append(np.average(qsys['gpe']['hamiltonian_average'])*muK_conv)
    Init_Energy_list.append(qsys['gpe']['hamiltonian_average'][0]*muK_conv)
    Tprob.append(normdiff_last / ((tprob_last + tprob_last_flipped)/2))
    if scale < 0.50:
        scale += 0.01
    else:
        scale += 0.05


########################################################################################################################
#
# Plot the results
#
########################################################################################################################
plt_params = {'legend.fontsize': 'x-large',
        'figure.figsize': (8, 7),
        'axes.labelsize': 'x-large',
        'axes.titlesize':'xx-large',
        'xtick.labelsize':'x-large',
        'ytick.labelsize':'x-large'}
plt.rcParams.update(plt_params)
L2R_TP = np.array(L2R_TP)
R2L_TP = np.array(R2L_TP)
Energy_list = np.array(Energy_list)

figTPnorm = plt.figure(fignum)
fignum += 1
#plt.plot(Glist, Tnormdiff, label='integer step')
#plt.title('Effect of Feshbach Resonance on Asymmetry')
#plt.plot(Glist_fine, Tnormdiff_fine, '-*', label='0.05 step')
plt.plot(Glist_fine, L2R_TP, '-*', label='Left to Right Tunneling')
plt.plot(Glist_fine, R2L_TP, '-*', label='Right to Left Tunneling')
#plt.plot(Glist_last_fine, Tprob)
plt.xlabel('$a_s (a_0)$')
plt.ylabel('Tunnelling Probability at Final Time')
plt.legend(loc='best')

figTPbyEnergy = plt.figure(fignum)
fignum += 1
plt.plot(Glist_fine, (L2R_TP/Energy_list), '-*', label='Left to Right Tunneling')
plt.plot(Glist_fine, (R2L_TP/Energy_list), '-*', label='Right to Left Tunneling')
plt.xlabel('$a_s$ $(a_0)$')
plt.ylabel('Tunnelling Probability at Final Time')
plt.legend(loc='best')


figEnergy = plt.figure(fignum)
fignum += 1
plt.plot(Glist_fine, Energy_list)
plt.xlabel('$a_s (a_0)$')
plt.ylabel('Energy ($\\mu K$)')


figTPratio = plt.figure(fignum)
fignum += 1
ratio = L2R_TP / R2L_TP
plt.plot(Glist_fine, ratio)
plt.xlabel('$a_s (a_0)$')
plt.ylabel('$T_L / T_R$')

plt.show()
