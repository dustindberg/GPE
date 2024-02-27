from numba import jit, njit
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
import numpy as np
import scipy as sp
from scipy.constants import hbar, Boltzmann
from scipy.interpolate import UnivariateSpline
from quspin.operators import hamiltonian
from quspin.basis import boson_basis_1d
from quspin.tools.evolution import evolve
import datetime
import pytz
from tqdm import tqdm
import h5py
from multiprocessing import Pool
import os
import quspin

########################################################################################################################
# Set the initial plotting parameters
########################################################################################################################
# Colorblind friendly color scheme reordered for maximum efficiency
ok = {
    'blue': "#56B4E9",
    'orange': "#E69F00",
    'green': "#009E73",
    'amber': "#F5C710",
    'purple': "#CC79A7",
    'navy': "#0072B2",
    'red': "#D55E00",
    'black': "#000000",
    'yellow': "#F0E442",
    'grey': "#999999",
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
plt.rcParams.update(plt_params)

########################################################################################################################
# Define Model Parameters
# Using Quspin, we will be solving the Gross-Pitaevskii equation (GPE)
# i d/dt ψ_j(t) = -J[ψ_{j-1}(t) + ψ_{j+1}(t)] + V(x) ψ_j(t) + g|ψ_j(t)|² ψ_j(t)
# For more information, see: https://github.com/weinbe58/QuSpin/blob/master/examples/notebooks/GPE.ipynb
########################################################################################################################
from __future__ import print_function, division
import sys,os
# line 4 and line 5 below are for development purposes and can be removed
qspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,qspin_path)
#
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d # Hilbert space boson basis
from quspin.tools.evolution import evolve
import numpy as np # generic math functions
import h5py
from six import iteritems # loop over elements of dictionary
import matplotlib.pyplot as plt # plot library
#
##### define model parameters #####
L=180 # system size
# calculate centre of chain
if L%2==0:
	j0 = L//2-0.5 # centre of chain
else:
	j0 = L//2 # centre of chain
sites=np.arange(L)-j0
# static parameters
J=1.0 # hopping
U=1.0 # Bose-Hubbard interaction strength
# dynamic parameters
#kappa_trap_i=0.001 # initial chemical potential
#kappa_trap_f=0.0001 # final chemical potential
t_propagate = 6.0 / J 	# set total ramp time
cooling_potential_width = 0.01
cooling_potential_offset = 0
# Potential protocol
# calculate centre of chain
if L % 2 == 0:
	j0 = L // 2 - 0.5           # center of chain
else:
	j0 = L // 2               # center of chain
degrees = np.array(sites) * 180 / sites[-1]
##### construct single-particle Hamiltonian #####
# define site-coupling lists
hopping=[[-J, i, (i+1) % L] for i in range(L-1)]	# Periodic boundary conditions

def cooling_potential(j):
    v_cooling = cooling_potential_width * (j - cooling_potential_offset) ** 2
    return v_cooling - v_cooling.min()


# define basis
basis = boson_basis_1d(L,Nb=1,sps=2)
dynamic = []
init_v = cooling_potential(sites)
init_potential = [[init_v[_], _] for _ in range(L)]
init_static = [["+-",hopping], ["-+",hopping], ["n",init_potential]]
H_init=hamiltonian(init_static, dynamic, basis=basis, dtype=np.float64)
#print(f'Here the shape of the hamiltonian is {np.shape(Hsp)}')
E,V=H_init.eigsh(time=0.0, k=1, which='SA')


#########################################################
##### imaginary-time evolution to compute GS of GPE #####
################################################### ######
def GPE_imag_time(tau,phi,Hsp,U):
	"""
	This function solves the real-valued GPE in imaginary time:
	$$ -\dot\phi(\tau) = Hsp(t=0)\phi(\tau) + U |\phi(\tau)|^2 \phi(\tau) $$
	"""
	return -(H_init.dot(phi,time=0) + U*np.abs(phi)**2*phi )
# define ODE parameters
init_params = (H_init,U)
# define initial state to flow to GS from
phi0=V[:,0]*np.sqrt(L) # initial state normalised to 1 particle per site
# define imaginary time vector
tau=np.linspace(0.0,35.0,71)
# evolve state in imaginary time
psi_tau = evolve(phi0,tau[0],tau,GPE_imag_time,f_params=init_params, #GPE_params,
							imag_time=True,real=True,iterate=True)
#
# display state evolution
for i,psi0 in enumerate(psi_tau):
	# compute energy
	E_GS=(H_init.matrix_ele(psi0,psi0,time=0) + 0.5*U*np.sum(np.abs(psi0)**4)).real
	#(Hsp.matrix_ele(psi0,psi0,time=0) + 0.5*U*np.sum(np.abs(psi0)**4) ).real
	# plot wave function
	plt.plot(degrees, abs(phi0)**2, color='r',marker='s',alpha=0.2,
										label='$|\\phi_j(0)|^2$')
	plt.plot(degrees, abs(psi0)**2, color='b',marker='o',
								label='$|\\phi_j(\\tau)|^2$' )
	plt.plot(degrees, init_v, color='k', linestyle='--', label='Initial Trap')
	plt.xlabel('$\\mathrm{lattice\\ sites}$',fontsize=14)
	plt.title('$J\\tau=%0.2f,\\ E_\\mathrm{GS}(\\tau)=%0.4fJ$'%(tau[i],E_GS)
																,fontsize=14)
	plt.ylim([-0.01,max(abs(phi0)**2)*2])
	plt.legend(fontsize=14)
	plt.draw() # draw frame
	plt.pause(0.0005) # pause frame
	plt.clf() # clear figure
plt.close()
#
#########################################################
############## real-time evolution of GPE ###############
#########################################################
n_ramps = 2                 # Define the number of ramps in the periodic potential
ramp_width = int(L/(n_ramps ** 2))   # Give the length of the ramp potential in degrees
assert n_ramps * ramp_width < L,\
    (f"A system of grid size {L} cannot accommodate {n_ramps} barriers of width {ramp_width},\n"
     f"as the total width of all barriers {n_ramps * ramp_width}!")
ramp_spacing = int(L / n_ramps)  # Give the spacing between centers of ramps
ramp_height = 1.1 * E_GS  # Determine the maximum height of the
# Set the initial ramp center, and also grab the ends. The center is necessary for even grid sizes
ramp_center = [sites[int(0.5 * ramp_spacing)] + (1 - ramp_width % 2) / 2]
ramp_start = [ramp_center[0] - int(0.5 * ramp_width) + (1 - ramp_width % 2) / 2]
ramp_end = [np.floor(ramp_center[0] + 0.5 * ramp_width) - (1 - L % 2) / 2]

# Use a loop to set the remaining number of ramps
for _ in range(1, n_ramps):
	ramp_center.append(sites[int(0.5 * ramp_spacing) + _ * ramp_spacing] + (1 - ramp_width % 2) / 2)
	ramp_start.append(ramp_center[_] - int(0.5 * ramp_width) + (1 - ramp_width % 2) / 2)
	ramp_end.append(ramp_center[_] + int(0.5 * ramp_width) - (1 - ramp_width % 2) / 2)
#def ring_with_ramp(j):
#    """
#    This is absolutely the MOST over-engineered function I have ever written.
#    I would have saved literal days if I had just created the list manually.
#    :param j:
#    :return:
#    """
#    ring = np.zeros(len(j))
#    for _ in range(n_ramps):
#        ramp = np.where((j >= ramp_start[_]) & (j <= ramp_end[_]))[0]
#        for i in ramp:
#            ring[i] = ramp_height * ((j[i] - ramp_end[_]) / (ramp_start[_] - ramp_end[_]))
#    return ring

def ring_with_ramp(j):
	j = np.array(j)
	sbd = ramp_width / 2  # Single Beam Diameter
	sigma = (0.5 * sbd) / (2 * np.sqrt(2 * np.log(2)))
	space = sigma
	ring = np.zeros(len(j))
	for _ in range(n_ramps):
		ramp = np.exp(-((j - ramp_center[_] + 2 * space) / sigma) ** 2) +\
							0.8 * np.exp(-((j - ramp_center[_] + space) / sigma) ** 2) +\
							0.6 * np.exp(-((j - ramp_center[_]) / sigma) ** 2) +\
							0.4 * np.exp(-((j - ramp_center[_] - space) / sigma) ** 2) +\
							0.2 * np.exp(-((j - ramp_center[_] - 2 * space) / sigma) ** 2)
		ring += ramp
	ring *= ramp_height / ring.max()
	return ring

"""def ramp(t,kappa_trap_i,kappa_trap_f,t_ramp):
	return  (kappa_trap_f - kappa_trap_i)*t/t_ramp + kappa_trap_i
# ramp protocol parameters
ramp_args=[kappa_trap_i,kappa_trap_f,t_ramp]"""


#trap=[[0.0005*(i-j0)**2,i] for i in range(L)]
v = ring_with_ramp(sites)
potential = [[v[_], _] for _ in range(L)]
# define static and dynamic lists

static=[["+-",hopping],["-+",hopping], ["n",potential]] # [["+-",hopping],["-+",hopping]]
#print(f'Shapes of static is {np.shape(static)}')
# build Hamiltonian
Hsp=hamiltonian(static,dynamic,basis=basis,dtype=np.float64,check_symm=False)
GPE_params = (Hsp,U)

def GPE(time,psi):
	"""
	This function solves the complex-valued time-dependent GPE:
	$$ i\dot\psi(t) = Hsp(t)\psi(t) + U |\psi(t)|^2 \psi(t) $$
	"""
	# solve static part of GPE
	psi_dot = Hsp.static.dot(psi) + U*np.abs(psi)**2*psi
	# solve dynamic part of GPE
	#for f, Hd in iteritems(Hsp.dynamic):
	#	psi_dot += f(time)*Hd.dot(psi)
	return -1j*psi_dot
# define real time vector
t=np.linspace(0.0, t_propagate, 2001)
# time-evolve state according to GPE

psi_t = evolve(psi0,t[0],t,GPE,iterate=True,atol=1E-12,rtol=1E-12)
density = []
#
# display state evolution
for i,psi in enumerate(psi_t):
	# compute energy
	E=(Hsp.matrix_ele(psi,psi,time=t[i]) + 0.5*U*np.sum(np.abs(psi)**4)).real
	density.append(np.abs(psi)**2)
	# compute trap
	#kappa_trap= ramp(t[i],kappa_trap_i,kappa_trap_f,t_ramp)*(sites)**2
	# plot wave function
	plt.plot(degrees, abs(psi0)**2, color='r',marker='s',alpha=0.2
								,label='$|\\psi_{\\mathrm{GS},j}|^2$')
	plt.plot(degrees, abs(psi)**2, color='b',marker='o',label='$|\\psi_j(t)|^2$')
	plt.plot(degrees, max(abs(psi0)**2) * v/v.max(),'--',color='g',label='$V(x)$ normalized to density max')
	plt.ylim([-0.01,1.2*max(abs(psi0)**2)])
	plt.xlabel('$\\mathrm{lattice\\ sites}$',fontsize=14)
	plt.title('$Jt=%0.2f,\\ E(t) = %0.4fJ$' % (t[i], E), fontsize=14)
	#plt.title('$Jt=%0.2f,\\ E(t)-E_\\mathrm{GS}=%0.4fJ$'%(t[i],E-E_GS),fontsize=14)
	plt.legend(loc='upper right',fontsize=14)
	plt.draw() # draw frame
	plt.pause(0.005) # pause frame
	plt.clf() # clear figure
plt.close()
#
#######################################################################################
##### quantum real time evolution from GS of GPE with single-particle Hamiltonian #####
#######################################################################################
"""# define real time vector
t=np.linspace(0.0, 2 * t_propagate, 200)
# time-evolve state according to linear Hamiltonian Hsp (no need to define a GPE)
psi_sp_t = Hsp.evolve(psi0,t[0],t,iterate=True,atol=1E-12,rtol=1E-12)
#
# display state evolution
for i,psi in enumerate(psi_sp_t):
	# compute energy
	E=Hsp.matrix_ele(psi,psi,time=t[i]).real
	# compute trap
	kappa_trap=v # ramp(t[i],kappa_trap_i,kappa_trap_f,t_ramp)*(sites)**2
	# plot wave function
	plt.plot(sites, abs(psi0)**2, color='r',marker='s',alpha=0.2
								,label='$|\\psi_{\\mathrm{GS},j}|^2$')
	plt.plot(sites, abs(psi)**2, color='b',marker='o',label='$|U(t,0)|\\psi_{\\mathrm{GS},j}\\rangle|^2$')
	plt.plot(sites, kappa_trap,'--',color='g',label='$\\mathrm{trap}$')
	plt.ylim([-0.01,2*max(abs(phi0)**2)+0.01])
	plt.xlabel('$\\mathrm{lattice\\ sites}$',fontsize=14)
	plt.title('$Jt=%0.2f,\\ E(t)-E_\\mathrm{GS}=%0.4fJ$'%(t[i],E-E_GS),fontsize=14)
	plt.legend(loc='upper right',fontsize=14)
	plt.draw() # draw frame
	plt.pause(0.00005) # pause frame
	plt.clf() # clear figure
plt.close()"""

########################################################################################################################
# Quickly define the archive
########################################################################################################################
tag = f'Ring_L{L}-T{t_propagate}-J{J}-g{U:.2f}'
savesfolder = tag.replace('.', ',')
parent_dir = "./Archive_Data/QuspinGPE"
try:
    os.mkdir(parent_dir)
    print(f'Parent Directory created, saved to: {parent_dir}')
except:
    FileExistsError
    print(f'Parent directory check passed! \nResults will be saved to the path {parent_dir}\n')
path = os.path.join(parent_dir, savesfolder)
savespath = 'Archive_Data/QuspinGPE/' + str(savesfolder) + '/'

try:
    os.mkdir(path)
    print(f'Simulation Directory "{savesfolder}" created')
except:
    FileExistsError
    print('WARNING: The directory you are saving to already exists!!! \nYou may be overwriting previous data (; n;)\n')

plt.figure()
plt.imshow(np.flip(density, 0),
           extent=[degrees[0], degrees[-1], t[0], t[-1]],
           aspect=(degrees[-1] - degrees[0]) / (t[-1] - t[0])
           )
plt.xlabel('x (degrees)')
plt.ylabel('time')
plt.tight_layout()
plt.savefig(savespath+'_density.pdf')

with h5py.File(savespath+tag+'_data.hdf5', 'w') as file:
	density_set = file.create_dataset('Density', data=np.array(density))


