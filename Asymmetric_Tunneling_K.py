from numba import njit
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
import numpy as np
from scipy.constants import hbar, proton_mass, Boltzmann
from scipy.interpolate import UnivariateSpline
from split_op_gpe1D import SplitOpGPE1D, imag_time_gpe1D # class for the split operator propagation
import datetime
import pytz

Start_time = datetime.datetime.now(pytz.timezone('US/Central'))

########################################################################################################################
#
# Define the initial parameters for interaction and potential
#
########################################################################################################################

#Define physical parameters
a_0 = 5.291772109e-11                                       #Bohr Radius in meters

#Rubidium-87 properties
m = 1.4431609e-25                                          #Calculated mass of 87Rb in kg
a_s = 100 * a_0                                            #Background scattering length of 87Rb in meters

#Potassium-41 properties
#m= 6.80187119e-26                                           #Calculated mass of 41K in kg
#a_s = 65.42 * a_0                                           #Background scattering length of 41K in meters

#Experiment parameters
N = 1e4                                                     #Number of particles
omeg_x = 50 * 2 * np.pi                                     #Harmonic oscillation in the x-axis in Hz
omeg_y = 500 * 2 * np.pi                                    #Harmonic oscillation in the y-axis in Hz
omeg_z = 500 * 2 * np.pi                                    #Harmonic oscillation in the z-axis in Hz
omeg_cooling = 450 * 2 * np.pi                              #Harmonic oscillation for the trapping potential in Hz
scale = 1                                                   #Scaling factor for the interaction parameter

#Parameters calculated by Python
L_x = np.sqrt(hbar / (m * omeg_x))                          #Characteristic length in the x-direction in meters
L_y = np.sqrt(hbar / (m * omeg_y))                          #Characteristic length in the y-direction in meters
L_z = np.sqrt(hbar / (m * omeg_z))                          #Characteristic length in the z-direction in meters
g = 2 * N * L_x * m * scale * a_s * np.sqrt(omeg_y * omeg_z) / hbar #Dimensionless interaction parameter

#Conversion factors to plot in physical units
L_xmum = np.sqrt(hbar / (m * omeg_x)) * 1e6                 #Characteristic length in the x-direction in meters
L_ymum = np.sqrt(hbar / (m * omeg_y)) * 1e6                 #Characteristic length in the y-direction in meters
L_zmum = np.sqrt(hbar / (m * omeg_z)) * 1e6                 #Characteristic length in the z-direction in meters
time_conv = 1. / omeg_x * 1e3                               #Converts characteristic time into milliseconds
energy_conv = hbar * omeg_x                                 #Converts dimensionless energy terms to Joules
muK_conv = energy_conv * (1e6 / Boltzmann)                  #Converts Joule terms to microKelvin
nK_conv = energy_conv * (1e9 / Boltzmann)                   #Converts Joule terms to nanoKelvin
specvol_mum = (L_xmum * L_ymum * L_zmum) / N                #Converts dimensionless spacial terms into micrometers^3 per particle
dens_conv = 1. / (L_xmum * L_ymum * L_zmum)                 #Calculated version of density unit conversion

#External calculations of physical parameters for testing accuracy
L_x_calc = 1.52295528474e-6                                         #Calculated characteristic length in the x-direction in micrometers
L_y_calc = 4.81600747437e-7                                         #Calculated characteristic length in the y-direction in micrometers
L_z_calc = 4.81600747437e-7                                         #Calculated characteristic length in the z-direction in micrometers
L_xmum_calc = np.sqrt(1.0515718 / (14.431609 * omeg_x)) * 100       #Calculated characteristic length in the x-direction in micrometers
L_ymum_calc = np.sqrt(1.0515718 / (14.431609 * omeg_y)) * 100       #Calculated characteristic length in the y-direction in micrometers
L_zmum_calc = np.sqrt(1.0515718 / (14.431609 * omeg_z)) * 100       #Calculated characteristic length in the z-direction in micrometers
muK_calc = 0.00239962237                                            #Calculated convertion to microKelvin
specvol_mum_calc = (L_xmum_calc * L_ymum_calc * L_zmum_calc) / N    #Converts unit-less spacial terms into specific volume: micrometers^3 per particle
dens_conv_calc = 1. / (L_xmum_calc * L_ymum_calc * L_zmum_calc)     #Calculated version of density unit conversion
g_calc = 692.956625255                                              #Calculated dimensionless interaction parameter for 87Rb at 1000 particles
v_0_calc = 0.5 * (omeg_cooling / omeg_x) ** 2                       #Python calculated coefficient for the cooling potential (unreliable)

#Parameters for computation
propagation_dt = 1e-4
height_asymmetric = 35                                      #Height parameter of asymmetric barrier
delta = 5                                                   #Sharpness parameter of asymmetric barrier
v_0 = 45.5                                                    #Coefficient for the trapping potential
offset = 40.                                                #Center offset for cooling potential
fignum = 1                                                  #Declare starting figure number
x_amplitude = 80.                                           #Set the range for calculation


#Functions for computation
@njit
def v(x, t=0.):
    """
    Potential energy
    """
    #return 0.5 * x ** 2 + x ** 2 * height_asymmetric * np.exp(-(x / delta) ** 2) * (x < 0)
    return 3000 - 2800 * np.exp(-((x + offset)/ 45) ** 2) - 2850 * np.exp(-((x - offset) / 47) ** 2) - 100 * np.exp(-((x - 5) / 10) ** 2)

@njit
def diff_v(x, t=0.):
    """
    the derivative of the potential energy for Ehrenfest theorem evaluation
    """
    #return x + (2. * x - 2. * (1. / delta) ** 2 * x ** 3) * height_asymmetric * np.exp(-(x / delta) ** 2) * (x < 0)
    return (224 / 81) * (x + offset) * np.exp(-((x + offset)/ 45) ** 2) + (5700 / 2209) * (x - offset) * np.exp(-((x - offset) / 47) ** 2) + 2 * (x - 5) * np.exp(-((x - 5) / 10) ** 2)

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


#Saves certain parameters as a separate bundle
params = dict(
    x_grid_dim=8 * 1024,    #for faster testing, change x_grid_dim to 8*1024, for more accuracy, 32*1024. Experimenting shows 16 is the best blend of speed and accuracy.
    x_amplitude=x_amplitude,
    N=N,
    k=k,
    diff_v=diff_v,
    diff_k=diff_k,
)

########################################################################################################################
#
# Get the initial state
#
########################################################################################################################

@njit
def initial_trap(x, t=0):
    """
    Trapping potential to get the initial state
    :param x:
    :return:
    """
    return v_0 * (x + offset) ** 2

#Increase first step, and then tighten with intermediate step
init_state, mu = imag_time_gpe1D(
    v=initial_trap,
    g=g,
    dt=7e-5,
    epsilon=1e-9,
    **params
)
init_state, mu = imag_time_gpe1D(
    init_wavefunction=init_state,
    g=g,
    v=initial_trap,
    dt=1e-6,
    epsilon=1e-11,
    **params
)

flipped_initial_trap = njit(lambda x, t: initial_trap(-x, t))
flipped_init_state, mu_flip = imag_time_gpe1D(
    v=flipped_initial_trap,
    g=g,
    dt=7e-5,
    epsilon=1e-9,
    **params
)
flipped_init_state, mu_flip = imag_time_gpe1D(
    init_wavefunction=flipped_init_state,
    g=g,
    v=flipped_initial_trap,
    dt=1e-6,
    epsilon=1e-11,
    **params
)

gpe_qsys = SplitOpGPE1D(
    v=v,
    g=g,
    dt=propagation_dt,
    **params
)


figTrap = plt.figure(fignum, figsize=(8,6))
fignum+=1
plt.title('Trapping Potential')
x = gpe_qsys.x * L_xmum
plt.plot(x, initial_trap(x) * muK_conv)
plt.xlabel('$x$ ($\mu$m) ')
plt.ylabel('$V(x)$ ($\mu$K)')
plt.xlim([-80 * L_xmum, 80 * L_xmum])
plt.savefig('Trapping Potential' + '.png')

########################################################################################################################
#
# Plot the potential in physical units before proceeding with simulation
#
########################################################################################################################

gpe_qsys = SplitOpGPE1D(
    v=v,
    g=g,
    dt=propagation_dt,
    **params
)

dx = gpe_qsys.dx
x_cut = int(0.645 * gpe_qsys.wavefunction.size)               #These are cuts such that we observe the behavior about the initial location of the wave
x_cut_flipped = int(0.355 * gpe_qsys.wavefunction.size)

@njit
def v_muKelvin(v):
    """"
    The potential energy with corrected units microKelvin
    """
    #return v * muKelvin_conv
    return v * muK_calc

figV = plt.figure(fignum, figsize=(8,6))
fignum+=1
plt.title('Potential')
x = gpe_qsys.x * L_xmum
v_muK = v(x) * muK_conv
potential = v_muKelvin(v(x))
plt.plot(x, v_muK)
plt.fill_between(
    x[x_cut:],
   potential[x_cut:],
    potential.min(),
    facecolor="orange",
         color='orange',
      alpha=0.2
)
plt.fill_between(
    x[:x_cut_flipped],
    potential[:x_cut_flipped],
    potential.min(),
    facecolor="green",
         color='green',
      alpha=0.2
)
plt.xlabel('$x$ ($\mu$m) ')
plt.ylabel('$V(x)$ ($\mu$K)')
plt.xlim([-80 * L_xmum, 80 * L_xmum])
plt.savefig('Potential' + '.png')

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


x = gpe_qsys.x
x_mum = x * L_xmum                                                      #Converts dimensionless x to micrometers for graphing
chem_potential = v_0 ** (1/3) * (3 * g / 4) ** (2/3)                    #formula for the chemical potential
mu_nKelvin = mu * nK_conv                                               #Converts chemical potential to units of nanoKelvin

#rhs = tf_test(chem_potential, initial_trap(x), g)
rhs = tf_test(mu, initial_trap(x), g)
lhs = np.abs(init_state) ** 2

#TF Approx plot
figTFA = plt.figure(fignum, figsize=(16,12))
fignum+=1

plt.subplot(221)
plt.title('Initial Trap')
plt.plot(x_mum, rhs, label='Thomas Fermi')
plt.plot(x_mum, lhs, label='GPE')
#plt.xlim([-40, -20])
plt.legend(numpoints=1)
plt.xlabel('$x$ ($\mu$m)')
plt.ylabel('Density (dimensionless)')

plt.subplot(223)
plt.title('Initial Trap in log scale')
plt.semilogy(x_mum, rhs, label='Thomas Fermi')
plt.semilogy(x_mum, lhs, label='GPE')
#plt.xlim([-40, -20])
plt.legend(numpoints=1)
plt.xlabel('$x$ ($\mu$m)')
plt.ylabel('Density (dimensionless)')

#Flip the initial conditions
#rhs = tf_test(chem_potential, flipped_initial_trap(x, 0), g)
rhs = tf_test(mu_flip, flipped_initial_trap(x,0), g)
lhs = np.abs(flipped_init_state) ** 2


#TF Approx plot flipped
plt.subplot(222)
plt.title('Flipped Initial Trap')
plt.plot(x_mum, rhs, label='Thomas Fermi')
plt.plot(x_mum, lhs, label='GPE')
#plt.xlim([20, 40])
plt.legend(numpoints=1)
plt.xlabel('$x$ ($\mu$m)')
plt.ylabel('Density (dimensionless)')

plt.subplot(224)
plt.title('Flipped Initial Trap in log scale')
plt.semilogy(x_mum, rhs, label='Thomas Fermi')
plt.semilogy(x_mum, lhs, label='GPE')
#plt.xlim([20, 40])
plt.legend(numpoints=1)
plt.xlabel('$x$ ($\mu$m)')
plt.ylabel('Density (dimensionless)')

figTFA.suptitle('Thomas-Fermi Approximations')
plt.savefig('Thomas-Fermi Approximations' + '.png')

#plt.show()

########################################################################################################################
#
# Generate plots to test the propagation
#
########################################################################################################################


def analyze_propagation(qsys, wavefunctions, title, fignum):
    """
    Make plots to check the quality of propagation
    :param qsys: an instance of SplitOpGPE1D
    :param wavefunctions: list of numpy.arrays
    :param title: str
    :return: None
    """

    #plot the density over time
    figdensplot = plt.figure(fignum, figsize=(8,6))
    fignum+=1
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
    plt.savefig(title + '.png')


    figefr = plt.figure(fignum, figsize=(24,6))
    fignum+=1
    times = qsys.times
    t_ms = np.asarray(times) * time_conv
    plt.subplot(141)
    plt.title("Verify the first Ehrenfest theorem", pad = 15)

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

    plt.subplot(142)
    plt.title("Verify the second Ehrenfest theorem", pad = 15)

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

    plt.subplot(143)
    plt.title("The expectation value of the hamiltonian", pad = 15)

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

    plt.subplot(144)
    plt.title('time increments $dt$', pad = 15)
    plt.plot(qsys.time_increments)
    plt.ylabel('$dt$')
    plt.xlabel('time step')
    figefr.suptitle(plot_title)
    plt.savefig('EFT_' + plot_title + '.png')

    return fignum

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
t_msplot = times * time_conv
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
fignum = analyze_propagation(schrodinger_qsys, schrodinger_wavefunctions, "Schrodinger evolution", fignum)

# Analyze the Flipped schrodinger propagation
fignum = analyze_propagation(
    flipped_schrodinger_qsys,
    flipped_schrodinger_wavefunctions, # [psi[::-1] for psi in flipped_schrodinger_wavefunctions],
    "Flipped Schrodinger evolution", fignum
)


# Analyze the GPE propagation
fignum = analyze_propagation(gpe_qsys, gpe_wavefunctions, "GPE evolution", fignum)


# Analyze the Flipped GPE propagation
fignum = analyze_propagation(
    flipped_gpe_qsys,
    flipped_gpe_wavefunctions, # [psi[::-1] for psi in flipped_gpe_wavefunctions],
    "Flipped GPE evolution", fignum
)


########################################################################################################################
#
# Calculate the transmission probability
#
########################################################################################################################


figTP = plt.figure(fignum,figsize=(18,6))
fignum+=1
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
plt.savefig('Transmission Probability' + '.png')

End_time = datetime.datetime.now(pytz.timezone('US/Central'))

print ("Start time: {}:{}:{}".format(Start_time.hour,Start_time.minute,Start_time.second))
print ("Mid-point time: {}:{}:{}".format(Mid_time.hour,Mid_time.minute,Mid_time.second))
print ("End time: {}:{}:{}".format(End_time.hour, End_time.minute, End_time.second))

plt.show()
