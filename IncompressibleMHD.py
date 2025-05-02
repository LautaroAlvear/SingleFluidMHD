import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import h5py
import mpi4py
import shutil
import math
from scipy.special import erf
import sys
#shutil.rmtree('snapshots', ignore_errors=True)

from mpl_toolkits.mplot3d import Axes3D

def plot_vector_field_spherical(field_components):
    """
    Grafica un campo vectorial definido en coordenadas esféricas.
    
    field_components: lista [Phi, Theta, R]
      - Cada uno es un array de misma forma, con los valores de cada componente del campo vectorial
    """
    Phi, Theta, R = field_components  # cada uno es un array 3D

    # Convertir las coordenadas esféricas (phi, theta, r) a cartesianas
    # phi: azimutal [0, 2pi], theta: polar [0, pi], r: radio positivo

    # Suponemos que los arrays tienen las mismas dimensiones y representan una malla (φ, θ, r)
    # Si no son mallas ya formadas, se debe adaptar

    # Crear la malla de posiciones en coordenadas esféricas
    phi_vals = np.linspace(0, 2 * np.pi, Phi.shape[0])
    theta_vals = np.linspace(0, np.pi, Phi.shape[1])
    r_vals = np.linspace(0, 1, Phi.shape[2])

    phi, theta, r = np.meshgrid(phi_vals, theta_vals, r_vals, indexing='ij')

    # Convertir puntos a cartesianas
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    # Convertir vectores esféricos a cartesianas
    # Fórmulas de conversión
    Bx = (np.sin(theta) * np.cos(phi)) * R + (np.cos(theta) * np.cos(phi)) * Theta - np.sin(phi) * Phi
    By = (np.sin(theta) * np.sin(phi)) * R + (np.cos(theta) * np.sin(phi)) * Theta + np.cos(phi) * Phi
    Bz = (np.cos(theta)) * R - (np.sin(theta)) * Theta

    # Subsamplear para mejor visualización (opcional)
    step = (slice(None, None, 4), slice(None, None, 4), slice(None, None, 2))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(
        x[step], y[step], z[step],
        Bx[step], By[step], Bz[step],
        length=0.1, normalize=True, color='b'
    )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Campo vectorial en coordenadas cartesianas')
    plt.tight_layout()
    plt.show()



#-----------------Parametros-------------------------#

#----Dedalus----#
#Nphi, Ntheta, Nr = 1, 128, 128
#Nphi, Ntheta, Nr = 4,3,3                       #For fast debugging
Nphi, Ntheta, Nr = 128, 96, 128
dealias = 3/2
stop_sim_time = 20
timestepper = d3.SBDF2
#timestepper = d3.RK222
max_timestep = 0.05
dtype = np.float64
mesh = None

#---Simulacion---#
mu_0 = 1
Re = 1
Rm = 100
R = 1
m0 = 1

#-----Inicializacion Objetos-----------------#
coords = d3.SphericalCoordinates("phi","theta","r")
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
ball = d3.BallBasis(coords, shape=(Nphi, Ntheta, Nr), radius=R, dealias=dealias, dtype=dtype)
sphere = ball.surface


#---------------------Fields--------------------------------#
u = dist.VectorField(coords,bases=ball ,name='u')
B = dist.VectorField(coords,bases=ball ,name='A')
p = dist.Field(bases=ball ,name="p")                                #Fields do not require coords

tau_u = dist.VectorField(coords,bases=sphere,name='tau_u')
tau_B = dist.VectorField(coords,bases=sphere,name="tau_B")                  #Field bc gauge is invariant under grad of field
tau_p = dist.Field(name='tau_p')                                    #constant bc gauge is invariant under constant


#-------------------------Sustituciones---------------------------------#
phi, theta, r = dist.local_grids(ball)
r_vec = dist.VectorField(coords, bases=ball.radial_basis)
lift = lambda A: d3.Lift(A, ball, -1)
lift2 = lambda A:d3.Lift(A,ball,-2)
cross = lambda A,B:d3.CrossProduct(A,B)

J = d3.Curl(B)/mu_0 
JcrossB = d3.CrossProduct(J,B)
curlucrossB = d3.Curl(d3.CrossProduct(u,B))
#---------------------Initial Conditions-----------------------------------------------------------#
u.fill_random(layout='g')
#B.fill_random(layout='g')
B['g'][1] = m0/(r**3)*np.sin(theta)
B['g'][2] = m0/(r**3)*2*np.cos(theta)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# Graficar
print(r,theta[0],len(B['g'][0]))

print(len(B['g'][0][0][0]))

Theta, R = np.meshgrid(theta[:][0], r[0][0])
matrix = B['g'][2][0]

pcm = ax.pcolormesh(Theta,R,np.transpose(matrix), shading="nearest", cmap='inferno',vmin=0,vmax=10)
plt.show()


#------------------------------------Equations----------------------------------------------------#
problem = d3.IVP([u,B, p,tau_u,tau_p,tau_B], namespace=locals())

#-----Main Equations-------#
problem.add_equation("dt(u) + grad(p) + lift(tau_u) - lap(u)*(1/Re) = -u@grad(u) + cross(J,B)")    #momentum
problem.add_equation("dt(B) - lap(B)*(1/Rm) + lift(tau_B)= Curl(cross(u,B))")                          #Faraday
problem.add_equation("div(u) + tau_p = 0")                                            #mass

#-----Boundary Conditions--#
problem.add_equation("u(r=R) = 0")
problem.add_equation("B(r=R) = 0")

#-----Gauge----------------#
problem.add_equation("integ(p) = 0")
                                                                            #es correcta


#------------------------------------Solver---------------------------------------------------------#
solver = problem.build_solver(timestepper)
solver.stop_sim_time = 20

snapshots = solver.evaluator.add_file_handler(f'snapshots_IMHD', sim_dt=.001, max_writes=30000,iter=10)
snapshots.add_task(u,name='u',layout='g')
snapshots.add_task(d3.DotProduct(u,u),name='mag(u)',layout='g')
snapshots.add_task(p,name='p',layout='g')
snapshots.add_task(d3.Curl(u),name='vorticity',layout='g')
snapshots.add_task(B,name="B",layout='g')
snapshots.add_task(d3.DotProduct(B,B),name="B^2",layout='g')
timestep = 5e-4
print(timestep)

while solver.proceed:
    solver.step(timestep)
    if math.isnan(u['g'][-1][0][0][0]) or math.isnan(B['g'][-1][0][0][0]):
        raise Exception(f"El sistema exploto en la iteracion {solver.iteration}")
    if solver.iteration % 10 == 0:
        print('Completed iteration {} of {}'.format(solver.iteration,solver.stop_sim_time/timestep))
        
