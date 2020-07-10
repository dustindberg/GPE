from numba import njit # compile python
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
import numpy as np
from scipy.constants import hbar, proton_mass, Boltzmann
from scipy.constants import g as G
from scipy.interpolate import UnivariateSpline
from split_op_gpe1D import SplitOpGPE1D, imag_time_gpe1D # class for the split operator propagation
import datetime
import pytz


########################################################################################################################
#
#Define the parameters for interaction and potential
#
########################################################################################################################

Start_time = datetime.datetime.now(pytz.timezone('US/Central'))

#Calculated Physical Values and conversion factors
N = 1e4                                                     #number of particles
m = 1.4431609e-25                                           #Calculated mass of 87Rb in kg
Omeg_x = 50 * 2 * np.pi                                     #Harmonic oscillation in the x-axis in Hz
Omeg_y = 500 * 2 * np.pi                                    #Harmonic oscillation in the y-axis in Hz
Omeg_z = 500 * 2 * np.pi                                    #Harmonic oscillation in the z-axis in Hz
a_s = 5.291772109e-9                                        #Scattering length also in meters, from 100 * 5.291772109e-11
L_x = 1.52295528474e-6                                      #Externally calcilated characteristic length in the x-direction in micrometers
L_y = 4.81600747437e-7                                      #Externally calcilated characteristic length in the y-direction in micrometers
L_z = 4.81600747437e-7                                      #Externally calcilated characteristic length in the z-direction in micrometers
L_xmum = np.sqrt(1.0515718 / (14.431609 * Omeg_x)) * 100    #Externally calcilated characteristic length in the x-direction in micrometers
L_ymum = np.sqrt(1.0515718 / (14.431609 * Omeg_y)) * 100    #Externally calcilated characteristic length in the y-direction in micrometers
L_zmum = np.sqrt(1.0515718 / (14.431609 * Omeg_z)) * 100    #Externally calcilated characteristic length in the z-direction in micrometers
muK_calc = 0.00239962237                                    #Calculated convertion to microKelvin
specvol_mum = (L_xmum * L_ymum * L_zmum) / N                #Converts unit-less spacial terms into specific volume: micrometers^3 per particle
dens_convcalc = 1. / (L_xmum * L_ymum * L_zmum)             #calculated version of density unit conversion

#Reference values. Internally calculated and usually subject to rounding errors. Only use as a basis when params change.
Lx_ref = np.sqrt(hbar / (m * Omeg_x))                           #Characteristic length in the x-direction in meters
Ly_ref = np.sqrt(hbar / (m * Omeg_y))                           #Characteristic length in the y-direction in meters
Lz_ref = np.sqrt(hbar / (m * Omeg_z))                           #Characteristic length in the z-direction in meters
energy_conv  = hbar ** 2 / (m * Lx_ref ** 2)                    #Converts unit-less energy terms to joules
energy_conv_alt  = hbar * Omeg_x
muK_ref = energy_conv * (1e6 / Boltzmann)                       #Converts Joules energy terms to microKelvin
alt_muKevlin_conv = hbar * Omeg_x * 1e6 / Boltzmann             #An alternate conversion from dimensionless energy terms to microKelvin
nK_ref = energy_conv * (1e9 / Boltzmann)                        #Converts Joules energy terms to nanoKelvin
alt_nK_ref = hbar * Omeg_x * 1e9 / Boltzmann                    #An alternate conversion from dimensionless energy terms to nanoKelvin
specvol_ref = (Lx_ref * Ly_ref * Lz_ref) / N                    #Converts unit-less spacial terms into specific volume: m^3 per particle
time_ref = 1. / Omeg_x *1e3                                     #Converts characteristic time into milliseconds
xmum_ref = Lx_ref * 1e6                                         #Converts dimensionless x coordinate into micrometers
dens_ref = 1e-18 / (Lx_ref * Ly_ref * Lz_ref)                   #converts dimensionless wave function into units of micrometers^-3
g_ref = 2 * N * Lx_ref * m * a_s * np.sqrt(Omeg_y * Omeg_z) / hbar #In program calculation of the dimensionless interaction parameter


#Hand Calculated dimensionless interaction parameter
g = 692.956625255

propagation_dt = 1e-4

#height of asymmetric barrier
height_asymmetric = 35
#for ~550 nK, try 25 for #1 and 24.24195 for #4 at delta=5
#for ~800 nK, try 72.61931 for #1 at delta = 3.5, try 34.8903 for delta = 5
#for ~1microK, try 91.1142 for #1 at delta = 3.5, try 43.9528 for delta = 5

#This corresponds to sharpness parameter
delta = 5


#Increases the number of peaks for Option 2 or second peak width for Option 3
osc = 6

@njit
def v(x, t=0.):
    """
    Potential energy
    """
    #Option 1
    return 0.5 * x ** 2 + x ** 2 * height_asymmetric * np.exp(-(x / delta) ** 2) * (x < 0)
    #Option 2
    #return 0.5 * x ** 2 + height_asymmetric * np.sin(osc * x) ** 2 * np.exp(-(x / delta) ** 2) * (x < 0)
    #Option 3
    #return 0.5* x ** 2 + height_asymmetric * x ** 2 * np.exp(-(x / delta) ** 2) * (x < 0) + 0.05 * height_asymmetric * x ** 2 * np.exp(-((x / osc) + 1) ** 2) * (x < 0)
    #Option 4
    #return 0.25 * x ** 2 + x ** 2 * height_asymmetric * np.exp(-(x / delta) ** 2) * (x < 0)
@njit
def diff_v(x, t=0.):
    """
    the derivative of the potential energy for Ehrenfest theorem evaluation
    """
    #Option 1
    return x + (2. * x - 2. * (1. / delta) ** 2 * x ** 3) * height_asymmetric * np.exp(-(x / delta) ** 2) * (x < 0)
    #Option 2
    #return x + (2 * osc * np.sin(osc * x) * np.cos(osc * x) - 2. * x * (1. / delta) ** 2 * np.sin(osc * x) ** 2) * height_asymmetric * np.exp(-(x / delta) ** 2) * (x < 0)
    #Option 3
    #return x + ((x - x ** 3 * (1. / delta) ** 2) * 2 * height_asymmetric * np.exp(-(x / delta) ** 2))*(x < 0) + ((x - (x/osc + 1) * x **2 * (1. / osc)) * 0.1 * height_asymmetric * np.exp(-((x / osc) + 1) ** 2)) * (x < 0)
    #Option 4
    #return 0.5 * x + (2. * x - 2. * (1. / delta) ** 2 * x ** 3) * height_asymmetric * np.exp(-(x / delta) ** 2) * (x < 0)

@njit
def diff_k(p, t=0.):
    """
    the derivative of the kinetic energy for Ehrenfest theorem evaluation
    """
    return p



@njit
def k(p, t=0.):
    """
    Non-relativistic kinetic energy
    """
    return 0.5 * p ** 2

#add in functions for Thomas Fermi approximation test later
#@njit
#def g_compare(k):
#    """
#    Create an array of same size as k for comparison of g to k
#    """
#    return (-1) * k + g

#@njit
#def g_constant(t=0.):
#    """
#    Create an array of same size as k for baseline
#    """
#    return g

# save parameters as a separate bundle
params = dict(
    x_grid_dim=8 * 1024,
    #for faster testing, change x_grid_dim to 8*1024, for more accuracy, 32*1024. Experimenting shows 16 is the best blend of speed and accuracy. 8 should be used for bulk testing of code with needed accuracy
    x_amplitude=80.,

    k=k,

    diff_v=diff_v,
    diff_k=diff_k,

    # epsilon=1e-2,
)

########################################################################################################################
#
# Plot the potential
#
########################################################################################################################

#Change units to miliKelvin, move def function here
#plot this here

gpe_qsys = SplitOpGPE1D(
    v=v,
    g=g,
    dt=propagation_dt,
    **params
)

@njit
def v_muKelvin(v):
    """"
    The potential energy with corrected units microKelvin
    """
    #return v * muKelvin_conv
    return v * muK_calc

plt.title('Potential')
x = gpe_qsys.x  * L_xmum
plt.plot(x, v_muKelvin(v(x)))
plt.xlabel('$x$ ($\mu$m) ')
plt.ylabel('$V(x)$ ($\mu$K)')
plt.xlim([-80 * L_xmum, 80 * L_xmum])
plt.savefig('Potential' + '.pdf')
plt.show()

########################################################################################################################
#
# Get the initial state
#
########################################################################################################################

#parameters for trapping potential
v_0 = 12.5
offset = 20.

@njit
def initial_trap(x, t=0):
    """
    Trapping potential to get the initial state
    :param x:
    :return:
    """
    # omega = 2 * Pi * 100Hz
    #omega should be 2 * Pi * 250Hz
    #Convert to new Omega (leave offset of 20)
    return v_0 * (x + offset) ** 2

#Increase first step, and then tighten with intermediate step
init_state, mu = imag_time_gpe1D(
    v=initial_trap,
    g=g,
    dt=1e-3,
    epsilon=1e-8,
    **params
)

init_state, mu = imag_time_gpe1D(
    init_wavefunction=init_state,
    g=g,
    v=initial_trap,
    dt=1e-5,
    epsilon=1e-10,
    **params
)


flipped_initial_trap = njit(lambda x, t: initial_trap(-x, t))

flipped_init_state, mu_flip = imag_time_gpe1D(
    v=flipped_initial_trap,
    g=g,
    dt=1e-3,
    epsilon=1e-8,
    **params
)

flipped_init_state, mu_flip = imag_time_gpe1D(
    init_wavefunction=flipped_init_state,
    g=g,
    v=flipped_initial_trap,
    dt=1e-5,
    epsilon=1e-10,
    **params
)

# qsys = SplitOpGPE1D(
#     v=initial_trap,
#     g=g,
#     dt=1e-3,
#     **params
# ).set_wavefunction(init_state)
# test_init_state = qsys.propagate(0.05)
#
# x = qsys.x
# plt.semilogy(x, np.abs(init_state), label='initial condition')
# plt.semilogy(x, np.abs(test_init_state), label='propagated init condition')
#
# plt.semilogy(x, v(x))
# plt.xlabel('$x$')
# plt.ylabel('$v(x)$')
# plt.legend()
# plt.show()

########################################################################################################################
#
# Adding tests for the Thomas Fermi approximation
#
########################################################################################################################

#Define Parameters for graphing
@njit
def tf_test(mu, v, g):
    """"
    Right side of the equation for Thomas-Fermi approximation test
    """
    y = (mu - v) / g
    return y * (y > 0)

dx = 2. * params['x_amplitude'] / params['x_grid_dim']
x = (np.arange(params['x_grid_dim']) - params['x_grid_dim'] / 2) * dx
x_mum = x * L_xmum
chem_potential = (2 * v_0) ** (1/3) * (3 * g / 8) ** (2/3)
mu_nKelvin = mu * nK_ref                                              #Converts chemical potential to units of nanoKelvin
g_units = g * nK_ref * specvol_mum * 2 * np.pi                        #Converts dimensionless

rhs = tf_test(chem_potential, initial_trap(x), g)
lhs = np.abs(init_state) ** 2

#TF Approx plot
plt.plot(x * L_xmum, rhs, label='Thomas Fermi')
plt.plot((x * L_xmum), lhs, label='GPE')
#plt.xlim([-45, -20])
plt.legend(numpoints=1)
plt.xlabel('$x$ ($\mu$m)')
plt.ylabel('Density (dimensionless)')
plt.savefig('Thomas-Fermi Approximation' + '.pdf')
plt.show()

#Flip the initial conditions
rhs = tf_test(mu_flip, flipped_initial_trap(x, 0), g)
lhs = np.abs(flipped_init_state) ** 2

#TF Approx plot flipped
plt.plot(x * L_xmum, rhs, label='Thomas Fermi Flipped')
plt.plot((x * L_xmum), lhs, label='GPE flipped')
plt.xlim([20, 45])
plt.legend(numpoints=1)
plt.xlabel('$x$ ($\mu$m)')
plt.ylabel('Density (dimensionless)')
plt.savefig('Thomas-Fermi Approximation Flipped' + '.pdf')
plt.show()

########################################################################################################################
#
# Generate plots to test the propagation
#
########################################################################################################################


def analyze_propagation(qsys, wavefunctions, title):
    """
    Make plots to check the quality of propagation
    :param qsys: an instance of SplitOpGPE1D
    :param wavefunctions: list of numpy.arrays
    :param title: str
    :return: None
    """
    plt.title(title)
    plot_title = title
    # plot the time dependent density
    extent = [qsys.x.min(), qsys.x.max(), 0., T]

    plt.imshow(
        np.abs(wavefunctions) ** 2,
        # some plotting parameters
        origin='lower',
        extent=extent,
        aspect=(extent[1] - extent[0]) / (extent[-1] - extent[-2]),
        norm=SymLogNorm(vmin=1e-13, vmax=1., linthresh=1e-15)
    )
    plt.xlabel('coordinate $x$ (a.u.)')
    plt.ylabel('time $t$ (a.u.)')
    plt.colorbar()
    plt.savefig(title + '.pdf')

    plt.show()

    times = qsys.times
    t_ms = np.asarray(times) * time_ref
    plt.subplot(131)
    plt.title("Verify the first Ehrenfest theorem", pad = 10)
    #plt.title("Verify the first Ehrenfest theorem", pad= Try out values tomorrow)

    plt.plot(
        t_ms,
        # calculate the derivative using the spline interpolation
        # because times is not a linearly spaced array
        UnivariateSpline(times, qsys.x_average, s=0).derivative()(times),
        '-r',
        label='$d\\langle\\hat{x}\\rangle / dt$'
    )
    plt.plot(
        t_ms,
        qsys.x_average_rhs,
        '--b',
        label='$\\langle\\hat{p}\\rangle$'
    )
    plt.legend()
    plt.ylabel('momentum')
    plt.xlabel('time $t$ (ms)')

    plt.subplot(132)
    plt.title("Verify the second Ehrenfest theorem", pad = 10)

    plt.plot(
        t_ms,
        # calculate the derivative using the spline interpolation
        # because times is not a linearly spaced array
        UnivariateSpline(times, qsys.p_average, s=0).derivative()(times),
        '-r',
        label='$d\\langle\\hat{p}\\rangle / dt$'
    )
    plt.plot(t_ms, qsys.p_average_rhs, '--b', label='$\\langle -U\'(\\hat{x})\\rangle$')
    plt.legend()
    plt.ylabel('force')
    plt.xlabel('time $t$ (ms)')

    plt.subplot(133)
    plt.title("The expectation value of the hamiltonian", pad = 10)

    # Analyze how well the energy was preserved
    h = np.array(qsys.hamiltonian_average)
    print(
        "\nHamiltonian is preserved within the accuracy of {:.1e} percent".format(
            100. * (1. - h.min() / h.max())
        )
    )
    print("Initial energy {:.4e}".format(h[0]))

    plt.plot(t_ms, h * muK_calc)
    plt.ylabel('energy ($\mu$K)')
    plt.xlabel('time $t$ (ms)')
    plt.savefig('EFT_' + plot_title + '.pdf')
    plt.show()

    plt.title('time increments $dt$')
    plt.plot(qsys.time_increments)
    plt.ylabel('$dt$')
    plt.xlabel('time step')
    plt.show()


########################################################################################################################
#
# Propagate GPE equation
#
########################################################################################################################

print("\nPropagate GPE equation")

gpe_qsys = SplitOpGPE1D(
    v=v,
    g=g,
    dt=propagation_dt,
    **params
)
gpe_qsys.set_wavefunction(init_state)

# get time duration of 2 periods
T = 2. * 2. * np.pi
times = np.linspace(0, T, 500)
t_msplot = times * time_ref
# propagate till time T and for each time step save a probability density
gpe_wavefunctions = [
     gpe_qsys.propagate(t).copy() for t in times
]

########################################################################################################################
#
# Propagate GPE equation with flipped initial condition
#
########################################################################################################################

print("\nPropagate GPE equation with flipped initial condition")

flipped_gpe_qsys = SplitOpGPE1D(
    v=v,
    g=g,
    dt=propagation_dt,
    **params
).set_wavefunction(flipped_init_state)

# propagate till time T and for each time step save a probability density
flipped_gpe_wavefunctions = [
     flipped_gpe_qsys.propagate(t).copy() for t in times
]

########################################################################################################################
#
# Propagate Schrodinger equation
#
########################################################################################################################

print("\nPropagate Schrodinger equation")

schrodinger_qsys = SplitOpGPE1D(
    v=v,
    g=0.,
    dt=propagation_dt,
    **params
).set_wavefunction(init_state)

# propagate till time T and for each time step save a probability density
schrodinger_wavefunctions = [
     schrodinger_qsys.propagate(t).copy() for t in times
]

########################################################################################################################
#
# Propagate Schrodinger equation with flipped initial condition
#
########################################################################################################################

print("\nSchrodinger equation with flipped initial condition")

flipped_schrodinger_qsys = SplitOpGPE1D(
    v=v,
    g=0.,
    dt=propagation_dt,
    **params
).set_wavefunction(flipped_init_state)

# propagate till time T and for each time step save a probability density
flipped_schrodinger_wavefunctions = [
     flipped_schrodinger_qsys.propagate(t).copy() for t in times
]

Mid_time = datetime.datetime.now(pytz.timezone('US/Central'))

########################################################################################################################
#
# Analyze the simulations
#
########################################################################################################################

# Analyze the schrodinger propagation
analyze_propagation(schrodinger_qsys, schrodinger_wavefunctions, "Schrodinger evolution")

# Analyze the Flipped schrodinger propagation
analyze_propagation(
    flipped_schrodinger_qsys,
    flipped_schrodinger_wavefunctions, # [psi[::-1] for psi in flipped_schrodinger_wavefunctions],
    "Flipped Schrodinger evolution"
)

# Analyze the GPE propagation
analyze_propagation(gpe_qsys, gpe_wavefunctions, "GPE evolution")

# Analyze the Flipped GPE propagation
analyze_propagation(
    flipped_gpe_qsys,
    flipped_gpe_wavefunctions, # [psi[::-1] for psi in flipped_gpe_wavefunctions],
    "Flipped GPE evolution"
)

########################################################################################################################
#
# Calculate the transmission probability
#
########################################################################################################################

dx = gpe_qsys.dx
x_cut = int(0.6 * gpe_qsys.wavefunction.size)               #These are cuts such that we observe the behavior about the initial location of the wave
x_cut_flipped = int(0.4 * gpe_qsys.wavefunction.size)

#potential = v_muKelvin(v(x))
#plt.title('Potential')
#plt.plot(x, potential)
#plt.fill_between(
#    x[x_cut:],
#   potential[x_cut:],
#    potential.min(),
#    facecolor="orange",
#         color='orange',
#      alpha=0.2
#)
#plt.fill_between(
#    x[:x_cut_flipped],
#    potential[:x_cut_flipped],
#    potential.min(),
#    facecolor="green",
#         color='green',
#      alpha=0.2
#)
#plt.xlabel('$x$ ($\mu$m) ')
#plt.ylabel('$V(x)$ ($\mu$K)')
#plt.xlim([-80 * L_xmum, 80 * L_xmum])
#plt.show()

plt.subplot(121)
plt.plot(
    t_msplot,
    np.sum(np.abs(schrodinger_wavefunctions)[:, x_cut:] ** 2, axis=1) * dx,
    label='Schrodinger'
)
plt.plot(
    t_msplot,
    np.sum(np.abs(flipped_schrodinger_wavefunctions)[:, :x_cut_flipped] ** 2, axis=1) * dx,
    label='Flipped Schrodinger'
)
plt.legend()
plt.xlabel('time $t$ (ms)')
plt.ylabel("transmission probability")

plt.subplot(122)
plt.plot(
    t_msplot,
    np.sum(np.abs(gpe_wavefunctions)[:, x_cut:] ** 2, axis=1) * dx,
    label='GPE'
)
plt.plot(
    t_msplot,
    np.sum(np.abs(flipped_gpe_wavefunctions)[:, :x_cut_flipped] ** 2, axis=1) * dx,
    label='Flipped GPE'
)
plt.legend()
plt.xlabel('time $t$ (ms)')
plt.ylabel("transmission probability")
plt.savefig('Transmission Probability' + '.pdf')
plt.show()

End_time = datetime.datetime.now(pytz.timezone('US/Central'))

print ("Start time: {}:{}:{}".format(Start_time.hour,Start_time.minute,Start_time.second))
print ("Mid-point time: {}:{}:{}".format(Mid_time.hour,Mid_time.minute,Mid_time.second))
print ("End time: {}:{}:{}".format(End_time.hour, End_time.minute, End_time.second))
