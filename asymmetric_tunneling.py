from numba import njit # compile python
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
import numpy as np
from scipy.constants import hbar, proton_mass, Boltzmann
from scipy.interpolate import UnivariateSpline
from split_op_gpe1D import SplitOpGPE1D, imag_time_gpe1D # class for the split operator propagation

########################################################################################################################
#
#Define the parameters for interaction and potential
#
########################################################################################################################

#Assign physical values used
N = 1e5                                                 #number of particles
m = 1.443161930e-25                                     #Mass of 87Rb in kg
Omeg_x = 50 * 2 * np.pi                                 #Harmonic oscillation in the x-axis in Hz
Omeg_y = 500 * 2 * np.pi                               #Harmonic oscillation in the y-axis in Hz
Omeg_z = 500 * 2 * np.pi                               #Harmonic oscillation in the z-axis in Hz
L_x = np.sqrt(hbar / (m * Omeg_x))                      #Characteristic length in the x-direction in meters
L_y = np.sqrt(hbar / (m * Omeg_y))                      #Characteristic length in the y-direction in meters
L_z = np.sqrt(hbar / (m * Omeg_z))                      #Characteristic length in the z-direction in meters
a_s = 100 * 5.291772109e-11                             #scattering length also in meters
Energy_con  = (hbar ** 2) / (L_x ** 2 * m)              #Converts unit-less energy terms to joules
mKelvin_con = Energy_con * (1e3 / Boltzmann)            #converts Joules energy terms to milliKelvin
SpecVol_con = (L_x * L_y * L_z)/N                       #converts unit-less spacial terms into specific volume: m^3 per particle

#Assign a value to the dimensionless interaction
#g = (2 * N * L_x * m * a_s * np.sqrt(Omeg_y * Omeg_z))/hbar
g = 2194.449140
propagation_dt = 1e-4

#height of asymmetric barrier
height_asymmetric = 190

#This corresponds to sharpness parameter
delta = 3.5

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
    x_grid_dim=16 * 1024,
    #for faster testing, change x_grid_dim to 2*1024, for more accuracy, 32*1024. Experimenting shows 16 is the best blend of speed and accuracy. 8 should be used for bulk testing of code with needed accuracy
    x_amplitude=80.,

    k=k,

    diff_v=diff_v,
    diff_k=diff_k,

    # epsilon=1e-2,
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
    # omega = 2 * Pi * 100Hz
    #Convert to new Omega (leave offset of 20)
    return 12.5 * (x + 20.) ** 2

#Increase first step, and then tighten with intermediate step
init_state, mu = imag_time_gpe1D(
    #for mod: init_state, mu = imag_time_gpe1D( add to all states and flipped
    v=initial_trap,
    g=g,
    dt=1e-3,
    epsilon=1e-8,
    **params
)

init_state, mu = imag_time_gpe1D(
    #for mod: init_state, mu = imag_time_gpe1D( add to all states and flipped
    v=initial_trap,
    g=g,
    dt=1e-4,
    epsilon=1e-9,
    **params
)

init_state, mu = imag_time_gpe1D(
    wavefunction=init_state,
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
    wavefunction=flipped_init_state,
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
#Adding tests for the Thomas Fermi approximation
#
########################################################################################################################

@njit
def tf_test(mu, v, g):
    """"
    Right side of the equation for Thomas-Fermi approximation test
    """
    y = (mu - v) / g
    return y * (y > 0)

#plt.plot()
#tf_test(mu, initial_trap(x), g)

#g_units = g * Energy_con * SpecVol_con / (2*np.pi)
dx = 2. * params['x_amplitude'] / params['x_grid_dim']
x = (np.arange(params['x_grid_dim']) - params['x_grid_dim'] / 2) * dx
rhs = tf_test(mu, initial_trap(x), g)
lhs = np.abs(init_state) ** 2
plt.title('Thomas-Fermi Approximation Test')
plt.plot(
    rhs, lhs,
    label='GPE'
)
plt.plot(rhs, rhs+0.06,label='Slope')
plt.legend(numpoints=1)
plt.xlabel('$\mu - V(x) / g$')
plt.ylabel('$|\Psi_0|^2$')
plt.show()

#TF Approx plot
plt.plot(x, rhs/rhs.max(), label='Thomas Fermi')
plt.plot(x, lhs/lhs.max(), label='GPE')
plt.legend(numpoints=1)
plt.xlabel('$x$')
plt.ylabel('Density')
plt.show()

rhs = tf_test(mu_flip, flipped_initial_trap(x, 0), g)
lhs = np.abs(flipped_init_state) ** 2
plt.title('Thomas-Fermi Approximation Test')
plt.plot(
    rhs, lhs,
    label='Flipped GPE'
)
plt.plot(rhs, rhs+0.06,label='Slope')
plt.legend(numpoints=1)
plt.xlabel('$\mu - V_0(x) / g$')
plt.ylabel('$|\Psi|^2$')

plt.show()

#TF Approx plot
plt.plot(x, rhs/rhs.max(), label='Thomas Fermi')
plt.plot(x, lhs/lhs.max(), label='Flipped GPE')
plt.legend(numpoints=1)
plt.xlabel('$x$')
plt.ylabel('Density')
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

    plt.subplot(131)
    plt.title("Verify the first Ehrenfest theorem")

    plt.plot(
        times,
        # calculate the derivative using the spline interpolation
        # because times is not a linearly spaced array
        UnivariateSpline(times, qsys.x_average, s=0).derivative()(times),
        '-r',
        label='$d\\langle\\hat{x}\\rangle / dt$'
    )
    plt.plot(
        times,
        qsys.x_average_rhs,
        '--b',
        label='$\\langle\\hat{p}\\rangle$'
    )
    plt.legend()
    plt.ylabel('momentum')
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(132)
    plt.title("Verify the second Ehrenfest theorem")

    plt.plot(
        times,
        # calculate the derivative using the spline interpolation
        # because times is not a linearly spaced array
        UnivariateSpline(times, qsys.p_average, s=0).derivative()(times),
        '-r',
        label='$d\\langle\\hat{p}\\rangle / dt$'
    )
    plt.plot(qsys.times, qsys.p_average_rhs, '--b', label='$\\langle -U\'(\\hat{x})\\rangle$')
    plt.legend()
    plt.ylabel('force')
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(133)
    plt.title("The expectation value of the hamiltonian")

    # Analyze how well the energy was preserved
    h = np.array(qsys.hamiltonian_average)
    print(
        "\nHamiltonian is preserved within the accuracy of {:.1e} percent".format(
            100. * (1. - h.min() / h.max())
        )
    )
    print("Initial energy {:.4e}".format(h[0]))

    plt.plot(times, h)
    plt.ylabel('energy')
    plt.xlabel('time $t$ (a.u.)')

    plt.show()

    plt.title('time increments $dt$')
    plt.plot(qsys.time_incremenets)
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
).set_wavefunction(init_state)

# get time duration of 2 periods
T = 1. * 2. * np.pi
times = np.linspace(0, T, 500)

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

plt.subplot(121)
plt.plot(
    times,
    np.sum(np.abs(schrodinger_wavefunctions)[:, x_cut:] ** 2, axis=1) * dx,
    label='Schrodinger'
)
plt.plot(
    times,
    np.sum(np.abs(flipped_schrodinger_wavefunctions)[:, :x_cut_flipped] ** 2, axis=1) * dx,
    label='Flipped Schrodinger'
)
plt.legend()
plt.xlabel("time")
plt.ylabel("transmission probability")

plt.subplot(122)
plt.plot(
    times,
    np.sum(np.abs(gpe_wavefunctions)[:, x_cut:] ** 2, axis=1) * dx,
    label='GPE'
)
plt.plot(
    times,
    np.sum(np.abs(flipped_gpe_wavefunctions)[:, :x_cut_flipped] ** 2, axis=1) * dx,
    label='Flipped GPE'
)
plt.legend()
plt.xlabel("time")
plt.ylabel("transmission probability")

plt.show()

########################################################################################################################
#
# Plot the potential
#
########################################################################################################################

#Change units to miliKelvin, move def function here

@njit
def v_mKelvin(v):
    """"
    The potential energy with corrected units milliKelvin
    """
    return v * mKelvin_con

plt.title('Potential')
x = gpe_qsys.x
plt.plot(x, v_mKelvin(v(x)))
plt.xlabel('$x / 2.4\mu m$ ')
plt.ylabel('$V(x) Joules$')
plt.show()



