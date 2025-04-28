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

#-----------------Parametros-------------------------#

#----Dedalus----#
Nphi, Ntheta, Nr = 1, 96, 96
#Nphi, Ntheta, Nr = 4,3,3                       #For fast debugging
dealias = 3/2
stop_sim_time = 20
timestepper = d3.SBDF2
#timestepper = d3.RK222
max_timestep = 0.05
dtype = np.float64
mesh = None

#---Simulacion---#
mu_0 = 1
Re = 10
Rem = 10
R = 1

#-----Inicializacion Objetos-----------------#
coords = d3.SphericalCoordinates("phi","theta","r")
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
ball = d3.BallBasis(coords, shape=(Nphi, Ntheta, Nr), radius=R, dealias=dealias, dtype=dtype)
sphere = ball.surface


#---------------------Fields--------------------------------#
u = dist.VectorField(coords,bases=ball ,name='u')
A = dist.VectorField(coords,bases=ball ,name='A')
p = dist.Field(bases=ball ,name="p")                                #Fields do not require coords

tau_u = dist.VectorField(coords,bases=sphere,name='tau_u')
tau_A = dist.Field(bases=ball,name="tau_A")                  #Field bc gauge is invariant under grad of field
tau_p = dist.Field(name='tau_p')                                    #constant bc gauge is invariant under constant


#-------------------------Sustituciones---------------------------------#
phi, theta, r = dist.local_grids(ball)
r_vec = dist.VectorField(coords, bases=ball.radial_basis)
r_vec['g'][2] = r
lift = lambda A: d3.Lift(A, ball, -1)
lift2 = lambda A:d3.Lift(A,ball,-2)

B = d3.Curl(A)
J = d3.Curl(B)/mu_0 
JcrossB = d3.CrossProduct(J,B)
ucrossB = d3.CrossProduct(u,B)
#---------------------Initial Conditions-----------------------------------------------------------#
u['g'][1] = 1
A['g'][1] = 1


#------------------------------------Equations-----------------------------------------------------#
problem = d3.IVP([u,A, p,tau_u,tau_p,tau_A], namespace=locals())

#-----Main Equations-------#
problem.add_equation("dt(u) + grad(p) + lift(tau_u) - lap(u)/Re = -u@grad(u)")    #momentum
problem.add_equation("dt(A) - lap(A)/Rem = ucrossB")                          #Faraday
problem.add_equation("div(u) + tau_p = 0")                                            #mass

#-----Boundary Conditions--#
problem.add_equation("u(r=R) = 0")

#-----Gauge----------------#
problem.add_equation("integ(p) = 0")
problem.add_equation("div(A) + tau_A = 0")


#------------------------------------Solver---------------------------------------------------------#
solver = problem.build_solver(timestepper)
solver.stop_sim_time = 20

snapshots = solver.evaluator.add_file_handler(f'snapshots_IMHD', sim_dt=.001, max_writes=30000,iter=10)
snapshots.add_task(u,name='u',layout='g')
snapshots.add_task(p,name='p',layout='g')
snapshots.add_task(d3.Curl(u),name='vorticity',layout='g')
snapshots.add_task(B,name="B",layout='g')
snapshots.add_task(d3.DotProduct(B,B),name="B^2",layout='g')
timestep = 5e-4
print(timestep)

while solver.proceed:
    solver.step(timestep)
    if math.isnan(u['g'][-1][0][0][0]):
        raise Exception(f"El sistema exploto en la iteracion {solver.iteration}")
    if solver.iteration % 100 == 0:
        print('Completed iteration {} of {}'.format(solver.iteration,solver.stop_sim_time/timestep))
        