from numba import njit # compile python
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
import numpy as np
from scipy.interpolate import UnivariateSpline
from split_op_gpe1D import SplitOpGPE1D, imag_time_gpe1D # class for the split operator propagation

omega = 1.5

@njit
def v(x, t=0.):
    """
    Potential energy
    """
    return 0.5 * (omega * x) ** 2


@njit
def diff_v(x, t=0.):
    """
    the derivative of the potential energy for Ehrenfest theorem evaluation
    """
    return (omega) ** 2 * x


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


# save parameters as a separate bundle
params = dict(
    x_grid_dim=2 * 1024,
    x_amplitude=30.,

    k=k,

    diff_v=diff_v,
    diff_k=diff_k,

    g=1000,

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
    return 0.5 * (8 * (x + 15.)) ** 2

init_state = imag_time_gpe1D(
    v=initial_trap,
    dt=1e-4,
    epsilon=1e-7,
    **params
)

init_state = imag_time_gpe1D(
    wavefunction=init_state,
    v=initial_trap,
    dt=1e-5,
    epsilon=1e-13,
    **params
)

qsys = SplitOpGPE1D(
    v=initial_trap,
    dt=1e-3,
    **params
).set_wavefunction(init_state)
test_init_state = qsys.propagate(0.5)

plt.semilogy(qsys.x, np.abs(init_state), label='initial condition')
plt.semilogy(qsys.x, np.abs(test_init_state), label='propagated init condition')
plt.legend()
plt.show()


########################################################################################################################
#
# Propagate
#
########################################################################################################################

qsys = SplitOpGPE1D(
    v=v,
    dt=0.1e-4,
    **params
).set_wavefunction(init_state)

# get time duration of 6 periods
T = 2 * 2. * np.pi / omega

# propagate till time T and for each time step save a probability density
wavefunctions = [
    qsys.propagate(t).copy() for t in np.arange(0, T, 50 * qsys.dt)
]

plt.title(
    "Test 1: Time evolution of harmonic oscillator with $\\omega$ = {:.1f} (a.u.)".format(omega)
)

# plot the time dependent density
extent = [qsys.x.min(), qsys.x.max(), 0., T]

plt.imshow(
    np.abs(wavefunctions) ** 2,
    # some plotting parameters
    origin='lower',
    extent=extent,
    aspect=(extent[1] - extent[0]) / (extent[-1] - extent[-2]),
    #norm=SymLogNorm(vmin=1e-13, vmax=1., linthresh=1e-15)
)
plt.xlabel('coordinate $x$ (a.u.)')
plt.ylabel('time $t$ (a.u.)')
plt.colorbar()
plt.show()

##################################################################################################

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

plt.plot(times, h)
plt.ylabel('energy')
plt.xlabel('time $t$ (a.u.)')

plt.show()

plt.title('time increments $dt$')
plt.plot(qsys.time_incremenets)
plt.ylabel('$dt$')
plt.xlabel('time step')
plt.show()





