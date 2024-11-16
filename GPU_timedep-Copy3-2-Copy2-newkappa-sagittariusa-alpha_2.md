```python
# Import required libraries
#%run opacity_formulae.ipynb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import newton
from numba import jit, cuda
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
%matplotlib inline

# Constants in CGS units
G = 6.67430e-8  # Gravitational constant (cm^3 g^-1 s^-2)
M_sun = 1.989e33  # Mass of the Sun (g)
M_star = M_sun*4.3e6   # Mass of central star (g)
k_B = 1.380649e-16  # Boltzmann constant (erg/K)
m_p = 1.6726219e-24  # Proton mass (g)
sigma_SB = 5.670374419e-5  # Stefan-Boltzmann constant (erg cm^-2 s^-1 K^-4)
#alpha = 0.4  # Alpha-viscosity parameter
r_cons = 8.31446261815324e7 # Molar gas constant in cgs

alpha_cold = 0.02
alpha_hot = 0.2

# Disk parameters
R_1 = 4e12  # Inner radius of the disk (cm)
R_K = 1e15  # Radius where initial mass is added (cm)
R_N = 2e15  # Outer radius of the disk (cm)
M_dot = 1e17  # Mass transfer rate (g/s)

min_Sigma = 1e-5

#define X in terms of R
def X_func(r):
    return 2* np.sqrt(r)

def R_func(x):
    return np.float_power(x,2)/4

X_1 = X_func(R_1)
X_K = X_func(R_K)
X_N = X_func(R_N)

# Constants for opacity and energy equations
a_const = 7.5657e-15  # Radiation density constant (erg cm^-3 K^-4)
mu = 0.5  # Mean molecular weight for ionized gas of pure hydrogen
c = 2.99792458e10  # Speed of light (cm/s)

# Simulation parameters
N = 100  # Number of grid points for the simulation
N_n = 3 # Extra grid points after bath outer radius
# Define the radial grid
X = np.linspace(X_1, X_N, N)
ΔX = np.diff(X)[0]  # Assuming uniform grid spacing
X = np.linspace(X_1,X_N+N_n*ΔX,N+N_n)
r = R_func(X)
Δr = np.diff(r)  # Assuming uniform grid spacing
Δr = np.insert(Δr, 0, 0)
ΔX = np.diff(X)[0]  # Assuming uniform grid spacing
X_N = X_N+N_n*ΔX
N = N+N_n #redefine N to match new X array


# Initial conditions
#Sigma = np.zeros(N)
Sigma = np.full(N, min_Sigma) # Surface density array (g cm^-2)
#Sigma = np.zeros(N) # Surface density array (g cm^-2)
T_c = np.full(N, 1e3)  # Mid-plane temperature array (K)
H = np.full(N, 1e7) #Height array
p = np.full(N, 1e3) #Pressure array
#nu = np.full(N, 1e11) #inital viscosity array
alpha = np.full(N,alpha_cold) #alpha_array

@jit(target_backend='cuda', nopython=True)
def alpha_visc(T_c):
    log_alpha_0 = np.log(alpha_cold)
    log_alpha_1 = np.log(alpha_hot) - np.log(alpha_cold)
    log_alpha_2 = (1 + (2.5e4 / T_c)**8)
    log_alpha = log_alpha_0 + log_alpha_1 / log_alpha_2
    alpha_visc = np.exp(log_alpha)
    return alpha_visc  

@jit(target_backend='cuda', nopython=True)
def S_factor(X,Sigma):
    return X*Sigma

@jit(target_backend='cuda', nopython=True)
def Marr(X,Sigma):
    S_f = X*Sigma
    M_array = np.pi * S_f * (X**2) * ΔX / 4
    return M_array

@jit(target_backend='cuda', nopython=True)
def Sigma_from_S(S,X):
    return S/X

# Function to calcualte omega
@jit(target_backend='cuda', nopython=True)
def omega(R):
    return np.sqrt(G * M_star/ (R**3))

# kinematic_viscosity function
@jit(target_backend='cuda', nopython=True)
def kinematic_viscosity(H, R, alpha):
    return (2/3) * alpha * omega(R) * H**2

# Function to calculate the scale height H
@jit(target_backend='cuda', nopython=True)
def scale_height(Sigma, p, R):
    a1 = 2 * p /Sigma
    a2 = R**3 / (G * M_star)
    return a1*a2

# Function to calculate density rho
@jit(target_backend='cuda', nopython=True)
def density(H, Sigma):
    return Sigma/(2*H)

# Function to calculate the pressure P
@jit(target_backend='cuda', nopython=True)
def pressure(H, Sigma, T):
    rho = Sigma/(2*H)
    return ((r_cons * Sigma * T)/(mu * 2 * H)) + ((1/3) * (a_const * T**4))

@jit(target_backend='cuda', nopython=True)
def pressure_2(H, Sigma, R):
    p_2 = (1/2) * Sigma * H * (G * M_star / (R**3))
    return p_2

# Function to calculate azimuthal velocity vφ = sqrt(G*M1/R)
@jit(target_backend='cuda', nopython=True)
def azimuthal_velocity(R):
    return np.sqrt(G * M_star / R)

nu = kinematic_viscosity(H, r, alpha)

# Function to calculate the derivative dΩ/dR
@jit(target_backend='cuda', nopython=True)
def derivative_omega(R):
    return -1.5 * np.sqrt(G * M_star) / R**(-5/2)

j_val = 0
dMj = 0
dMj1 = 0

def add_mass(Sigma, M_dot, dt, X, N):
    dM = M_dot * dt  # Mass added during this timestep  # X_K is twice the square root of R_K, R_K already defined as a global constant
    X_KΔM = dM * X_K  # Angular momentum associated with the added mass at R_K

    # The index for the outermost zone (assuming X is sorted in ascending order)
    outer_index = N - 1

    # Mass at the outermost zone X_N
    Mass = Marr(X, Sigma)
    
    # Mass at the outermost zone X_N
    ΔM_N = Mass[-1]

    # Specific angular momentum D_N at the outermost zone
    D_N = (X_KΔM + X_N * ΔM_N) / (dM + ΔM_N)
 #   print('D_N =',D_N, 'X_N =',X_N)

    # Distribute the mass and angular momentum inward
    for j in range(N-N_n, 0, -1):
        
        # Sum up the masses from the outermost zone down to the current zone
        massarr = Mass[j:]
        xmassarr = X[j:]*Mass[j:]
        
        sum_ΔM_i = sum(massarr)
        X_sum_ΔM_i = sum(xmassarr)
            
        # Specific angular momentum D_J for the current zone
        D_J = (X_KΔM + X_sum_ΔM_i) / (dM + sum_ΔM_i)
        
        # Check if D_J > X[j-1] to decide if mass needs to be redistributed to inner zones
        if D_J > X[j-1]:
            
            # Calculate ΔM_J and ΔM_J_minus_1 based on conservation of angular momentum
            ΔM_J = ((X[j] - D_J) / ΔX ) * (dM + sum_ΔM_i)
            ΔM_J_minus_1 = ((D_J - X[j-1]) / ΔX ) * (dM + sum_ΔM_i) + Mass[j-1]
            
            Sj = 4 * (ΔM_J) / (np.pi * X[j]**2 * ΔX)
            Sj1 = 4 * (ΔM_J_minus_1) / (np.pi * X[j-1]**2 * ΔX) 
            
          #  print('ΔM_J =',ΔM_J,ΔM_J_minus_1)
            # Update Sigma with the new values ensuring it's not negative
       #     Sigma[j] = max(Sj / X[j+1], 0)
       #     Sigma[j-1] = max(Sj1 / X[j], 0)
        
            sjx = Sj / X[j+1]
            sj1x = Sj1 / X[j]
            
            if sjx < 200:
                Sigma[j] = max(sjx, min_Sigma)
            if sj1x < 200:
                Sigma[j-1] = max(sj1x, min_Sigma)
           
        #    Sigma[-2] = Sigma[-1]
        #    print('D_J =', D_J, 'X[j-1] =', X[j-1], 'Sigma[j] =', Sigma[j], 'Sigma[j-1 =]', Sigma[j-1])
            
            global dMj
            global dMj1
            global j_val
            
            dMj = ΔM_J
            dMj1 = ΔM_J_minus_1
            j_val = j
            
            break
 
    new_Sigma = np.copy(Sigma)
    
    return new_Sigma

# Additional code to initialize Sigma, M_dot, dt, X, N, and R_K would be required.
# This function should be called within the context of these defined variables.

# Function for Stage 2 - Evolution of surface density
@jit(target_backend='cuda', nopython=True)
def evolve_surface_density(Sigma, dt, nu, X, ΔX, j_val):
    
    S_arr = S_factor(X,Sigma)
    
    new_Sigma = Sigma.copy()
    new_S = S_arr.copy()
    
    r = np.float_power(X,2)/4
 
    cw, a_1, n_1 = 0.2,1.5e15,5
    T_tid = cw * r[1:-1] * nu[1:-1] * Sigma[1:-1] * (r[1:-1] / a_1) ** n_1
    
    trunc_rad = int(N*(9.3/10))
    
    #tidal torque in terms of Sigma (NOT S)
    tidal_torque = dt / (2 * np.pi * r[1:-1]) * T_tid
    
    # Update S using the tidal torque
    new_S[1:-1] += (12 * dt / (X[1:-1]**2 * ΔX**2)) * (
        S_arr[:-2] * nu[:-2] + S_arr[2:] * nu[2:] - 2 * S_arr[1:-1] * nu[1:-1]
    )

    new_Sigma = Sigma_from_S(new_S,X)
    new_Sigma[-1] = new_Sigma[-2]
#    print('newSigma=',new_Sigma)
    if j_val >= trunc_rad:
        new_Sigma[1+trunc_rad:-1] = new_Sigma[1+trunc_rad:-1] - tidal_torque[trunc_rad:]

    new_Sigma = np.maximum(new_Sigma, min_Sigma)
#    print('newnewSigma=',new_Sigma)
    return new_Sigma

# Continuing with the implementation of Stage 3 and plotting

# Define the timestep for stability based on the biggest viscosity
# In calculate_timestep function
def calculate_timestep(X, nu):
    max_nu = np.max(nu)
    Δt = (1 / 2) * ((X[0]**2) * (ΔX**2)) / (12 * max_nu)
    return Δt

# Update timestep for stability
nu = kinematic_viscosity(H, r, alpha)

# Update the timestep calculation for stability based on viscosity
dt = calculate_timestep(X, nu)

# Function for opacity using Kramers' law

@jit(target_backend='cuda', nopython=True)
def Y(R,j_val,dMj,dMj1):
    fact1 = G * M_star * M_dot / (2 * np.pi * (R[j_val - 1]**2)*(R[j_val - 1] - R[j_val - 2]))
    fact2 = dMj1 / (dMj1 + dMj)
    Yjminus1 = fact1 * fact2 
    fact3 = G * M_star * M_dot / (2 * np.pi * (R[j_val]**2) * (R[j_val] - R[j_val - 1]))
    fact4 = dMj / (dMj1 + dMj) 
    Yj = fact3 * fact4
    return Yjminus1, Yj
 
@jit(target_backend='cuda', nopython=True)
def kappa_bath(H, Sigma, T):
    rho = density(H, Sigma)
#    print(rho)
    X_a = .96
    Z = 1-X_a
    Z_star = 5e-10
    a = 0.2*(1+X_a)*(1/(1+(2.7e11)*(rho/np.float_power(T,2))))
    b = 1/(1+np.float_power((T/(4.5e8)),0.86))
    kappa_e = a*b
    a1 = (4e25)*(1+X_a)*(Z+0.001)
    b1 = rho/np.float_power(T,3.5)
    kappa_k = a1*b1    
    a2 = (1.1e-25)*(np.sqrt(Z*rho))
    b2 = np.float_power(T,7.7)
    kappa_hminus = a2*b2
    kappa_M = 0.1*Z
    a3 = kappa_e + kappa_k
    b3 = 1/kappa_hminus + 1/a3
    c3 = 1/b3
    kappa_rad = kappa_M + c3    
    a4 = 2.6e-7
    b4 = Z_star
    c4 = np.float_power(T,2)/np.float_power(rho,2)
    c5 = 1+np.float_power((rho/2e6),(2/3))
    kappa_cond = a4*b4*c4*c5
    a5 = 1/kappa_rad
    a6 = 1/kappa_cond
    a7 = a5 + a6
    kappa_t = 1/a7    
#    print(kappa_t)
    return kappa_t

@jit(target_backend='cuda', nopython=True)
def kappa_old(H, Sigma, T):
    kappa_t = 0.4 + 3.2e22 * Sigma / (H * T**3.5)
    return kappa_t

# Energy generation rate per unit area
@jit(target_backend='cuda', nopython=True)
def F(H, Sigma, R, j_val, dMj, dMj1, alpha):
    nu0 = kinematic_viscosity(H, R, alpha)
    #nu0 = (alpha * c_s ** 2) / omega(R)
    F0 = (9/8) * nu0 * Sigma * omega(R)**2
#    if R == r[j_val]:
#        F0 += Y(r,j_val,dMj,dMj1)[1]
 #   elif R == r[j_val-1]:
 #       F0 += Y(r,j_val,dMj,dMj1)[0]
#    else:
#        F0 += 0
    return F0

@jit(target_backend='cuda', nopython=True)
def F_out(H, Sigma, R, T):
    opacity = kappa_old(H, Sigma, T)
#    print(opacity, Sigma, T, R, c_s)
    return (2 * a_const * c * T**4) / (3 * Sigma * opacity)

# Function to solve for temperature, considering energy balance
@jit(target_backend='cuda', nopython=True)
def func_to_solve(T, H, Sigma, r, j_val, dMj, dMj1, alpha):
    # Energy balance equation
    left_hand_side =  2*F(H, Sigma, r, j_val, dMj, dMj1,alpha) 
    right_hand_side =   F_out(H, Sigma, r ,T)# Energy radiated away
    return left_hand_side - right_hand_side

#solving for sound speed c_s
@jit(target_backend='cuda', nopython=True)
def press_func(H, Sigma, R, T):
    p1 = pressure(H, Sigma, T)
    p2 = pressure_2(H, Sigma, R)
    return p1 - p2

def update_temperature(H, Sigma, r, T_c, j_val, dMj, dMj1, alpha):
    tolerance = 1e-3  # Set desired tolerance value
    T_min_limit = 1e-3  # Set a minimum temp limit
    # List of initial guesses to try
    initial_guesses = [1e0,1e1,1e2,1e3,1e4,1e5,1e6]
    for i in range(len(Sigma)):
        if Sigma[i] > 1e-4:  # Avoid division by zero for zero surface density
            valid_T_found = False
            for guess in initial_guesses:
                try:
                    new_T_c = newton(func_to_solve, guess, args=(H[i], Sigma[i], r[i], j_val, dMj, dMj1, alpha[i]))
                    # Check if new sound speed is valid
                    if new_T_c > T_min_limit and abs(new_T_c - guess) >= tolerance:
                        T_c[i] = new_T_c
                        valid_T_found = True
                        break  # Valid sound speed found
                except RuntimeError:
                    continue  # Try the next guess
            if not valid_T_found:
                # If no valid solution found, keep the old value or the minimum limit
                T_c[i] = max(T_min_limit, T_c[i])                      
        else:
            T_c[i] = max(T_min_limit, T_c[i])    
    return T_c



def update_H(H, Sigma, r, T):
    tolerance = 1e-3  # Set your desired tolerance value
    h_min_limit = 1  # Set a minimum sound speed limit based on physical considerations

    # List of initial guesses to try
    initial_guesses = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11]

    for i in range(len(Sigma)):
        if Sigma[i] > 1e-6:  # Avoid division by zero for zero surface density
            valid_h_found = False
            for guess in initial_guesses:
                try:
                    H_new = newton(press_func, guess, args=(Sigma[i], r[i], T[i]))
                    #print('cs_new =', cs_new)
                    # Check if new sound speed is valid
                    if H_new > h_min_limit and H_new < 1e12 and abs(H_new - guess) >= tolerance:
                        H[i] = H_new
                        valid_h_found = True
                        break  # Valid sound speed found
                except RuntimeError:
                    continue  # Try the next guess
            if not valid_h_found:
                # If no valid solution found, keep the old value or the minimum limit
                H[i] = H[i]
        else:
            H[i] = max(h_min_limit, H[i])
    return H

def update_temperature_fast(H, Sigma, r, T_c, j_val, dMj, dMj1, alpha):
#    tolerance = 1e-3  # Set desired tolerance value
    T_min_limit = 1e0  # Set a minimum temp limit
    initial_guesses = [1e0,1e1,1e2,1e3,1e4,1e5,1e6]
    for i in range(len(Sigma)):
        if Sigma[i] > 1e-100:
            try:
               # new_T_c = newton(func_to_solve, 4e5, args=(H[i], Sigma[i], r[i], j_val, dMj, dMj1, alpha[i]))
                new_T_c = newton(func_to_solve, T_c[i]*1.5, args=(H[i], Sigma[i], r[i], j_val, dMj, dMj1, alpha[i]))
                    # Check if new sound speed is valid
                if new_T_c > T_min_limit and new_T_c < 2e10:# and abs(new_T_c - T_c[i]) >= tolerance:
                    T_c[i] = new_T_c
            except RuntimeError:
                for guess in initial_guesses:
                    try:
                        new_T_c = newton(func_to_solve, guess, args=(H[i], Sigma[i], r[i], j_val, dMj, dMj1, alpha[i]))
                        # Check if new sound speed is valid
                        if new_T_c > T_min_limit and new_T_c < 2e10:# and abs(new_T_c - guess) >= tolerance:
                            T_c[i] = new_T_c
                    except RuntimeError:
                        continue  # Try the next guess   
    return T_c

def update_H_fast(H, Sigma, r, T):
#    tolerance = 1e-3  # Set your desired tolerance value
    h_min_limit = 1e7  # Set a minimum sound speed limit based on physical considerations
    initial_guesses = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11]
    for i in range(len(Sigma)):
        if Sigma[i] > 1e-100:
            try:
             #   H_new = newton(press_func, 5e9, args=(Sigma[i], r[i], T[i]))
                H_new = newton(press_func, H[i]*1.5, args=(Sigma[i], r[i], T[i]))
                if H_new > h_min_limit and H_new < 2e10:# and abs(H_new - H[i]) >= tolerance:
                    H[i] = H_new
            except RuntimeError:
                for guess in initial_guesses:
                    try:
                        H_new = newton(press_func, guess, args=(Sigma[i], r[i], T[i]))
                    #print('cs_new =', cs_new)
                    # Check if new sound speed is valid
                        if H_new > h_min_limit and H_new < 2e10:# and abs(H_new - guess) >= tolerance:
                            H[i] = H_new
                    except RuntimeError:
                        continue  # Try the next guess
    return H

def plot_s(Sigma,r):
    y_values = Sigma
    x_values = r
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values)
    plt.show()
```


```python
timesteps = 25001 # Number of timesteps to simulate
output_times = np.linspace(1,25000,2500,dtype='int64') # Specific timesteps to output for plotting
```


```python
#c = 2.99792458e10
@jit(target_backend='cuda', nopython=True)
def Flux_irr(Sigma_array, nu, r, r_array):
    
    Inner_index = np.where(Sigma_array > 1e-5)[0][0]
    
    M_dot_boundary = M_dot
    C = 5e-3
    
    L_edd = 1.4e38 * (M_star / M_sun)
    M_dot_inner = 2 * np.pi * r_array[Inner_index] * Sigma_array[Inner_index] * 3 * nu[Inner_index] / r_array[Inner_index] 
    L_inner = M_dot_inner * c ** 2
    
    if M_dot_inner >= M_dot_boundary:
        epsilon = 0.1
    else:
        epsilon = 0.1 * (M_dot_inner/M_dot_boundary)
    
    L_x = epsilon * min(L_edd,L_inner)
    
    Flux = C * L_x / (4 * np.pi * r ** 2)
    
    return Flux

@jit(target_backend='cuda', nopython=True)
def Epsilon_irr(Sigma, nu, r):
    
    Inner_index = np.where(Sigma > 1e-5)[0][0]
    
    M_dot_boundary = M_dot
    C = 5e-3
    
    L_edd = 1.4e38 * (M_star / M_sun)
    M_dot_inner = 2 * np.pi * r[Inner_index] * Sigma[Inner_index] * 3 * nu[Inner_index] / r[Inner_index] 
    L_inner = M_dot_inner * c ** 2
    
    if M_dot_inner >= M_dot_boundary:
        epsilon = 0.1
    else:
        epsilon = 0.1 * (M_dot_inner/M_dot_boundary)
    
    L_x = epsilon * min(L_edd,L_inner)
    
    T_irr = (C * L_x / (4 * np.pi * sigma_SB * r ** 2))**(1/4)
    
    return (T_irr/1e4)**2

@jit(target_backend='cuda', nopython=True)
def r10(r):
    return r/1e10

@jit(target_backend='cuda', nopython=True)
def Sigma_max(Sigma, nu, r):
    a1 = 10.8 - 10.3*Epsilon_irr(Sigma, nu, r)
    a2 = alpha_cold**(-0.84)
    a3 = (M_star/M_sun)**(-0.37+0.1*Epsilon_irr(Sigma, nu, r))
    a4 = r10(r)**(1.11-0.27*Epsilon_irr(Sigma, nu, r))
    return a1*a2*a3*a4

@jit(target_backend='cuda', nopython=True)
def Sigma_min(Sigma, nu, r):
    a1 = 8.3 - 7.1*Epsilon_irr(Sigma, nu, r)
    a2 = alpha_hot**(-0.77)
    a3 = (M_star/M_sun)**(-0.37)
    a4 = r10(r)**(1.12-0.23*Epsilon_irr(Sigma, nu, r))
    return a1*a2*a3*a4

@jit(target_backend='cuda', nopython=True)
def T_c_max(Sigma, nu, r):
    a1 = 10700*alpha_cold**(-0.1)
    a2 = r10(r)**(-0.05*Epsilon_irr(Sigma, nu, r))
    return a1*a2

@jit(target_backend='cuda', nopython=True)
def T_c_min(Sigma, nu, r):
    a1 = 20900 - 11300*Epsilon_irr(Sigma, nu, r)
    a2 = alpha_hot**(-0.22)
    a3 = (M_star/M_sun)**(-0.01)
    a4 = r10(r)**(0.05-0.12*Epsilon_irr(Sigma, nu, r))
    a5 = a1*a2*a3*a4
    return np.maximum(a5, 1e3)
    
@jit(target_backend='cuda', nopython=True)
def alpha_visc_irr(T_c, Sigma, nu, r):
    
    T_crit = 0.5*(T_c_max(Sigma, nu, r)+T_c_min(Sigma, nu, r))
    
    log_alpha_0 = np.log(alpha_cold)
    log_alpha_1 = np.log(alpha_hot) - np.log(alpha_cold)
    log_alpha_2 = (1 + (T_crit / T_c)**8)
    log_alpha = log_alpha_0 + log_alpha_1 / log_alpha_2
    alpha_visc = np.exp(log_alpha)
    
    return alpha_visc  
```


```python
#Function for disk evaporation
@jit(target_backend='cuda', nopython=True)
def disk_evap(r):
    R_min = 5e8
    #epsilon
    epsilon = 0.1 * (R_min/r) ** (-2)
    M_edd = 1.4e18 * M_star #gs^-1
    R_s = 2 * G * M_star / c**2
    M_ev = 0.08 * M_edd * ((r/R_s)**(1/4) + epsilon * (r/(800*R_s))**2) ** (-1)
    return M_ev

# Function for Stage 2 - Evolution of surface density with disk evaporation
@jit(target_backend='cuda', nopython=True)
def evolve_surface_density(Sigma, dt, nu, X, ΔX, j_val):
    
    S_arr = S_factor(X,Sigma)
    
    new_Sigma = Sigma.copy()
    new_S = S_arr.copy()
    
    r = np.float_power(X,2)/4
 
    cw, a_1, n_1 = 0.2,1.5e15,5
    T_tid = cw * r[1:-1] * nu[1:-1] * Sigma[1:-1] * (r[1:-1] / a_1) ** n_1
    
    trunc_rad = int(N*(9.3/10))
    
    #tidal torque in terms of Sigma (NOT S)
    tidal_torque = dt / (2 * np.pi * r[1:-1]) * T_tid
    
    #evaporation
    evap_rad_lim = 1
    evap_rate = disk_evap(r[1:-1])
    evap_rate_sigma = 4 * evap_rate / (np.pi * X[1:-1]**2 * ΔX)
    
    # Update S using the tidal torque
    new_S[1:-1] += (12 * dt / (X[1:-1]**2 * ΔX**2)) * (
        S_arr[:-2] * nu[:-2] + S_arr[2:] * nu[2:] - 2 * S_arr[1:-1] * nu[1:-1]
    )

    new_Sigma = Sigma_from_S(new_S,X)
    new_Sigma[-1] = new_Sigma[-2]
    
    new_Sigma[1:evap_rad_lim+1] -= evap_rate_sigma[:evap_rad_lim] * dt
    
#    print('newSigma=',new_Sigma)
    if j_val >= trunc_rad:
        new_Sigma[1+trunc_rad:-1] = new_Sigma[1+trunc_rad:-1] - tidal_torque[trunc_rad:]

        
    new_Sigma = np.maximum(new_Sigma, min_Sigma)
#    print('newnewSigma=',new_Sigma)
    return new_Sigma

@jit(target_backend='cuda', nopython=True)
def F_out(H, Sigma, R, T):
    opacity = kappa_bath(H, Sigma, T)
#    print(opacity, Sigma, T, R, c_s)
    return (2 * a_const * c * T**4) / (3 * Sigma * opacity)
@jit(target_backend='cuda', nopython=True)
def func_to_solve(T, H, Sigma, nu, r, j_val, dMj, dMj1, alpha, Sarr, rarr):
    # Energy balance equation
    left_hand_side =  2*F(H, Sigma, r, j_val, dMj, dMj1,alpha) 
    right_hand_side =   F_out(H, Sigma, r ,T)# Energy radiated away
    F_irr = Flux_irr(Sarr, nu, r, rarr) #irradiation flux
    return left_hand_side + F_irr - right_hand_side
def update_temperature_fast(H, Sigma, nu, r, T_c, j_val, dMj, dMj1, alpha):
#    tolerance = 1e-3  # Set desired tolerance value
    T_min_limit = 1e0  # Set a minimum temp limit
    initial_guesses = [1e0,1e1,1e2,1e3,1e4,1e5,1e6]
    for i in range(len(Sigma)):
        if Sigma[i] > 1e-100:
            try:
               # new_T_c = newton(func_to_solve, 4e5, args=(H[i], Sigma[i], r[i], j_val, dMj, dMj1, alpha[i]))
                new_T_c = newton(func_to_solve, T_c[i]*1.5, args=(H[i], Sigma[i], nu, r[i], j_val, dMj, dMj1, alpha[i], Sigma, r))
                    # Check if new sound speed is valid
                if new_T_c > T_min_limit and new_T_c < 2e10:# and abs(new_T_c - T_c[i]) >= tolerance:
                    T_c[i] = new_T_c
            except RuntimeError:
                for guess in initial_guesses:
                    try:
                        new_T_c = newton(func_to_solve, guess, args=(H[i], Sigma[i], nu, r[i], j_val, dMj, dMj1, alpha[i], Sigma, r))
                        # Check if new sound speed is valid
                        if new_T_c > T_min_limit and new_T_c < 2e10:# and abs(new_T_c - guess) >= tolerance:
                            T_c[i] = new_T_c
                    except RuntimeError:
                        continue  # Try the next guess   
    return T_c
```


```python
df_0 = pd.read_csv('Sigma_history_bath_array_new_Fcs_final_newopac_4dbb.csv')
Sigma_history = df_0.to_numpy().tolist()
df_1 = pd.read_csv('Temp_history_bath_array_new_Fcs_final_newopac_4dbb.csv')
Temp_history = df_1.to_numpy().tolist()
df_2 = pd.read_csv('H_history_bath_array_new_Fcs_final_newopac_4dbb.csv')
H_history = df_2.to_numpy().tolist()
df_3 = pd.read_csv('alpha_history_bath_array_new_Fcs_final_newopac_4dbb.csv')
alpha_history = df_3.to_numpy().tolist()
df_4 = pd.read_csv('t_history_bath_array_new_Fcs_final_newopac_4dbb.csv')
t_history = df_4.to_numpy().flatten().tolist()
df_5 = pd.read_csv('Sigma_transfer_history_bath_array_new_Fcs_final_newopac_4dbb.csv')
Sigma_transfer_history = df_5.to_numpy().flatten().tolist()
totalt = t_history[200]
Sigma = np.array(Sigma_history[200])*20
Sigma = Sigma.flatten()
T_c = np.array(Temp_history[200])*20
T_c = T_c.flatten()
H = np.array(H_history[200])*20
H = H.flatten()
nu = kinematic_viscosity(H, r, alpha)
dt = calculate_timestep(X, nu)

Sigma_history = []
Temp_history = []
H_history = []
t_history = []
alpha_history = []
Sigma_transfer_history = []
totalt = 0.0 

Sigma_transfer_history.append(Sigma[1])
Sigma_history.append(Sigma.copy())
Temp_history.append(T_c.copy())
H_history.append(H.copy())
alpha_history.append(alpha.copy())
Sigma_transfer_history.append(Sigma[1])
Sigma_history.append(Sigma.copy())
Temp_history.append(T_c.copy())
H_history.append(H.copy())
alpha_history.append(alpha.copy())
t_history = np.array([0.0,0.0])

np.savetxt("Sigma_history_bath_array_new_Fcs_final_newopac_smbh_a2.csv", Sigma_history, delimiter=",")
np.savetxt("Temp_history_bath_array_new_Fcs_final_newopac_smbh_a2.csv", Temp_history, delimiter=",")
np.savetxt("H_history_bath_array_new_Fcs_final_newopac_smbh_a2.csv", H_history, delimiter=",")
np.savetxt("t_history_bath_array_new_Fcs_final_newopac_smbh_a2.csv", t_history, delimiter=",")
np.savetxt("alpha_history_bath_array_new_Fcs_final_newopac_smbh_a2.csv", alpha_history, delimiter=",")
np.savetxt("Sigma_transfer_history_bath_array_new_Fcs_final_newopac_smbh_a2.csv", Sigma_transfer_history, delimiter=",")
```


```python
#%%time

#########################################
############### 2nd Part ################
#########################################

for i in range(1):

    
    df_0 = pd.read_csv('Sigma_history_bath_array_new_Fcs_final_newopac_smbh_a2.csv')
    Sigma_history = df_0.to_numpy().tolist()
    df_1 = pd.read_csv('Temp_history_bath_array_new_Fcs_final_newopac_smbh_a2.csv')
    Temp_history = df_1.to_numpy().tolist()
    df_2 = pd.read_csv('H_history_bath_array_new_Fcs_final_newopac_smbh_a2.csv')
    H_history = df_2.to_numpy().tolist()
    df_3 = pd.read_csv('alpha_history_bath_array_new_Fcs_final_newopac_smbh_a2.csv')
    alpha_history = df_3.to_numpy().tolist()
    df_4 = pd.read_csv('t_history_bath_array_new_Fcs_final_newopac_smbh_a2.csv')
    t_history = df_4.to_numpy().flatten().tolist()
    df_5 = pd.read_csv('Sigma_transfer_history_bath_array_new_Fcs_final_newopac_smbh_a2.csv')
    Sigma_transfer_history = df_5.to_numpy().flatten().tolist()

    totalt = t_history[-1]
    Sigma = np.array(Sigma_history[-1])
    T_c = np.array(Temp_history[-1])
    H = np.array(H_history[-1])
    nu = kinematic_viscosity(H, r, alpha)
    dt = calculate_timestep(X, nu)
    if dt < 1500:
        dt = 1500
    
    for timestep in range(timesteps):
    
    # Add mass at the outer radius
        Sigma = add_mass(Sigma, M_dot, dt, X, N)
  #  print('Sigma at add mass =', Sigma)
    # Add mass at the outer radius
        Sigma = evolve_surface_density(Sigma, dt, nu, X, ΔX, j_val)
 #   print('Sigma at evolve =', Sigma)    
    # Update disk temperature
        T_c = update_temperature_fast(H, Sigma, nu, r, T_c,j_val, dMj, dMj1,alpha) 
    #scale height at current Sigma
        H = update_H_fast(H, Sigma, r, T_c)
    #alpha for given T_c
        alpha = alpha_visc_irr(T_c, Sigma, nu, r)
    # Update viscosity for the next timestep
        nu = kinematic_viscosity(H, r, alpha)
    # Update timestep for stability
        dt = calculate_timestep(X, nu)
        if dt < 1500:
            dt = 1500
        
        totalt += dt
    
    # Store surface density at specific timesteps
        if timestep in output_times:
            Inner_index = np.where(Sigma > 1e-5*21)[0][0]
            Sigma_transfer_history.append(Sigma[Inner_index])
            Sigma_history.append(Sigma.copy())
            Temp_history.append(T_c.copy())
            H_history.append(H.copy())
            alpha_history.append(alpha.copy())
            #print('dt=',dt, 'timestep=',timestep,'max Sigma=', np.max(Sigma),'total t =',totalt, 'j =', j_val, 'Sigma[Inner] =', Sigma[Inner_index])
            #print(Sigma)
            t_history.append(totalt)
        
    #    if timestep in output_times[::50]:
    #        plot_s(Sigma,r)
    #        plot_s(H,r)
    #        plot_s(T_c,r)
    
    np.savetxt("Sigma_history_bath_array_new_Fcs_final_newopac_smbh_a2.csv", Sigma_history, delimiter=",")
    np.savetxt("Temp_history_bath_array_new_Fcs_final_newopac_smbh_a2.csv", Temp_history, delimiter=",")
    np.savetxt("H_history_bath_array_new_Fcs_final_newopac_smbh_a2.csv", H_history, delimiter=",")
    np.savetxt("t_history_bath_array_new_Fcs_final_newopac_smbh_a2.csv", t_history, delimiter=",")
    np.savetxt("alpha_history_bath_array_new_Fcs_final_newopac_smbh_a2.csv", alpha_history, delimiter=",")
    np.savetxt("Sigma_transfer_history_bath_array_new_Fcs_final_newopac_smbh_a2.csv", Sigma_transfer_history, delimiter=",")
```

    dt= 1294287847.0588672 timestep= 1 max Sigma= 365.27644887159585 total t = 5940773396.729391 j = 93 Sigma[Inner] = 88.24948408137851
    


    
![png](output_5_1.png)
    



    
![png](output_5_2.png)
    



    
![png](output_5_3.png)
    


    dt= 2276272018.9941616 timestep= 11 max Sigma= 365.00854073490973 total t = 19423225888.761353 j = 92 Sigma[Inner] = 81.16622632993514
    dt= 6559543674.374256 timestep= 21 max Sigma= 363.8677623949922 total t = 77619033431.43881 j = 90 Sigma[Inner] = 51.48969924351291
    dt= 6725187903.966266 timestep= 31 max Sigma= 373.6948777591093 total t = 144152421100.71234 j = 90 Sigma[Inner] = 43.11435467346629
    dt= 6609520325.03166 timestep= 41 max Sigma= 389.7078688431479 total t = 211693387350.57138 j = 90 Sigma[Inner] = 29.390877874157088
    dt= 6609520325.031702 timestep= 51 max Sigma= 407.9216633960842 total t = 277788590600.88824 j = 90 Sigma[Inner] = 30.586387726932035
    dt= 6847702367.4184885 timestep= 61 max Sigma= 422.30891286030703 total t = 345004322916.2714 j = 90 Sigma[Inner] = 33.516604835281164
    dt= 7057709775.550706 timestep= 71 max Sigma= 433.46176307172226 total t = 414761694821.0988 j = 90 Sigma[Inner] = 38.988978475435005
    dt= 7178617513.72852 timestep= 81 max Sigma= 442.1846352341945 total t = 486002313937.6929 j = 90 Sigma[Inner] = 47.206829102120125
    dt= 7297405265.74178 timestep= 91 max Sigma= 449.10930325982315 total t = 558446078849.1808 j = 90 Sigma[Inner] = 57.77342377804555
    dt= 7409582886.180445 timestep= 101 max Sigma= 454.6766025142774 total t = 632043455240.2002 j = 90 Sigma[Inner] = 69.91114167785771
    dt= 7513669100.582433 timestep= 111 max Sigma= 459.1840519768244 total t = 706718630893.2434 j = 90 Sigma[Inner] = 82.70183470149854
    dt= 7609481384.051543 timestep= 121 max Sigma= 462.844394940429 total t = 782389013056.0302 j = 90 Sigma[Inner] = 95.31625609674008
    dt= 7697389355.861346 timestep= 131 max Sigma= 465.8163348886695 total t = 858973617820.9763 j = 90 Sigma[Inner] = 107.14359314106592
    dt= 7772864007.696199 timestep= 141 max Sigma= 468.22194490691396 total t = 936388700707.0209 j = 90 Sigma[Inner] = 117.82042908346328
    dt= 7821776473.486329 timestep= 151 max Sigma= 470.15396148396 total t = 1014391382010.9158 j = 90 Sigma[Inner] = 127.17808136003866
    dt= 7865182766.618066 timestep= 161 max Sigma= 471.69030128010763 total t = 1092851978391.2435 j = 90 Sigma[Inner] = 135.21598992820648
    dt= 7904065309.800637 timestep= 171 max Sigma= 472.8962908125966 total t = 1171721051744.3147 j = 90 Sigma[Inner] = 142.035040745122
    dt= 7939179299.240967 timestep= 181 max Sigma= 473.82509021145887 total t = 1250957675969.1484 j = 90 Sigma[Inner] = 147.77652987891656
    dt= 7971115677.465338 timestep= 191 max Sigma= 474.520306082968 total t = 1330527531039.9233 j = 90 Sigma[Inner] = 152.5915729824169
    dt= 8000344268.591921 timestep= 201 max Sigma= 475.0179520210598 total t = 1410401510788.5815 j = 90 Sigma[Inner] = 156.6240845128321
    dt= 8027243918.165044 timestep= 211 max Sigma= 475.34794515024805 total t = 1490554686128.796 j = 90 Sigma[Inner] = 160.00283855634797
    dt= 8052123969.752074 timestep= 221 max Sigma= 475.53526541435843 total t = 1570965519212.7249 j = 90 Sigma[Inner] = 162.83893738784656
    dt= 8075239852.817193 timestep= 231 max Sigma= 475.6008649345833 total t = 1651615258142.5962 j = 90 Sigma[Inner] = 165.22616207276022
    dt= 8096804592.004829 timestep= 241 max Sigma= 475.56238898186945 total t = 1732487463927.2744 j = 90 Sigma[Inner] = 167.24265226369295
    dt= 8116997440.521361 timestep= 251 max Sigma= 475.4347528389753 total t = 1813567635908.0344 j = 90 Sigma[Inner] = 168.95304213629197
    dt= 8135970451.026049 timestep= 261 max Sigma= 475.23060699929476 total t = 1894842911624.0352 j = 90 Sigma[Inner] = 170.41060433967303
    dt= 8153853543.683784 timestep= 271 max Sigma= 474.9607148785449 total t = 1976301823738.9382 j = 90 Sigma[Inner] = 171.65919905332515
    dt= 8170758462.824376 timestep= 281 max Sigma= 474.6342613195455 total t = 2057934101265.7822 j = 90 Sigma[Inner] = 172.73495719626158
    dt= 8186781900.322498 timestep= 291 max Sigma= 474.25910589265465 total t = 2139730505582.3 j = 90 Sigma[Inner] = 173.66769254244787
    dt= 8202007986.239505 timestep= 301 max Sigma= 473.8419918405657 total t = 2221682694059.654 j = 90 Sigma[Inner] = 174.48206667756028
    dt= 8216510293.35777 timestep= 311 max Sigma= 473.3887191569506 total t = 2303783105820.667 j = 90 Sigma[Inner] = 175.19854088177607
    dt= 8230353464.223033 timestep= 321 max Sigma= 472.90428850059936 total t = 2386024865390.144 j = 90 Sigma[Inner] = 175.83414991816284
    dt= 8243594542.129305 timestep= 331 max Sigma= 472.3930212761305 total t = 2468401700929.1704 j = 90 Sigma[Inner] = 176.40312956795907
    dt= 8256284067.791105 timestep= 341 max Sigma= 471.85866015082723 total t = 2550907874446.225 j = 90 Sigma[Inner] = 176.9174252406986
    dt= 8268466989.007199 timestep= 351 max Sigma= 471.3044534475062 total t = 2633538121912.3306 j = 90 Sigma[Inner] = 177.38710436765555
    dt= 8280183419.9062605 timestep= 361 max Sigma= 470.73322619966774 total t = 2716287601619.011 j = 90 Sigma[Inner] = 177.82069110447102
    dt= 8291469278.330749 timestep= 371 max Sigma= 470.1474401364944 total t = 2799151849437.6743 j = 90 Sigma[Inner] = 178.22543830256436
    dt= 8302356823.828849 timestep= 381 max Sigma= 469.5492444510736 total t = 2882126739889.7344 j = 90 Sigma[Inner] = 178.6075487675122
    dt= 8312875114.071459 timestep= 391 max Sigma= 468.9405188725909 total t = 2965208452134.9243 j = 90 Sigma[Inner] = 178.97235544256873
    dt= 8323050393.922127 timestep= 401 max Sigma= 468.32291029476556 total t = 3048393440142.8706 j = 90 Sigma[Inner] = 179.32446825065944
    dt= 8332906428.598068 timestep= 411 max Sigma= 467.69786399509337 total t = 3131678406439.4287 j = 90 Sigma[Inner] = 179.66789381222273
    dt= 8342464790.174771 timestep= 421 max Sigma= 467.06665030221643 total t = 3215060278921.1455 j = 90 Sigma[Inner] = 180.00613305236396
    dt= 8351745104.962869 timestep= 431 max Sigma= 466.4303874238709 total t = 3298536190313.974 j = 90 Sigma[Inner] = 180.34226075431292
    dt= 8360765267.917378 timestep= 441 max Sigma= 465.79006102905095 total t = 3382103459919.832 j = 90 Sigma[Inner] = 180.67899035465058
    dt= 8369541629.145688 timestep= 451 max Sigma= 465.1465410802759 total t = 3465759577350.013 j = 90 Sigma[Inner] = 181.01872666737225
    dt= 8378089156.702523 timestep= 461 max Sigma= 464.50059633118354 total t = 3549502187990.1167 j = 90 Sigma[Inner] = 181.36360873568017
    dt= 8386421579.149993 timestep= 471 max Sigma= 463.8529068379189 total t = 3633329079979.018 j = 90 Sigma[Inner] = 181.71554461677894
    dt= 8394551510.784543 timestep= 481 max Sigma= 463.2040747774056 total t = 3717238172515.9243 j = 90 Sigma[Inner] = 182.07623958590736
    dt= 8402490561.962205 timestep= 491 max Sigma= 462.55463381952893 total t = 3801227505335.9023 j = 90 Sigma[Inner] = 182.44721898588412
    dt= 8410249436.567745 timestep= 501 max Sigma= 461.90505726185904 total t = 3885295229216.4067 j = 90 Sigma[Inner] = 182.82984673552946
    


    
![png](output_5_5.png)
    



    
![png](output_5_6.png)
    



    
![png](output_5_7.png)
    


    dt= 8417838018.35555 timestep= 511 max Sigma= 461.2557651034572 total t = 3969439597395.9805 j = 90 Sigma[Inner] = 183.22534033508495
    dt= 8425265447.627197 timestep= 521 max Sigma= 460.6071302074506 total t = 4053658957802.1084 j = 90 Sigma[Inner] = 183.63478306186704
    dt= 8432540189.492521 timestep= 531 max Sigma= 459.95948367951826 total t = 4137951745998.601 j = 90 Sigma[Inner] = 184.05913392911728
    dt= 8439670094.778456 timestep= 541 max Sigma= 459.31311957048496 total t = 4222316478774.3457 j = 90 Sigma[Inner] = 184.4992358808233
    dt= 8446662454.497952 timestep= 551 max Sigma= 458.6682989952625 total t = 4306751748305.03 j = 90 Sigma[Inner] = 184.95582261158393
    dt= 8453524048.662682 timestep= 561 max Sigma= 458.0252537469077 total t = 4391256216827.841 j = 90 Sigma[Inner] = 185.42952433050525
    dt= 8460261190.115574 timestep= 571 max Sigma= 457.3841894731882 total t = 4475828611776.39 j = 90 Sigma[Inner] = 185.92087272933293
    dt= 8466879763.967408 timestep= 581 max Sigma= 456.7452884733975 total t = 4560467721329.329 j = 90 Sigma[Inner] = 186.4303053656723
    dt= 8473385263.144483 timestep= 591 max Sigma= 456.10871216499334 total t = 4645172390331.563 j = 90 Sigma[Inner] = 186.95816963069947
    dt= 8479782820.487747 timestep= 601 max Sigma= 455.47460326267407 total t = 4729941516551.626 j = 90 Sigma[Inner] = 187.50472643596214
    dt= 8486077237.787786 timestep= 611 max Sigma= 454.84308770660033 total t = 4814774047242.904 j = 90 Sigma[Inner] = 188.07015372468612
    dt= 8492273012.091215 timestep= 621 max Sigma= 454.2142763714212 total t = 4899668975979.937 j = 90 Sigma[Inner] = 188.65454988856527
    dt= 8498374359.57273 timestep= 631 max Sigma= 453.58826658346237 total t = 4984625339744.179 j = 90 Sigma[Inner] = 189.25793715061988
    dt= 8504385237.230964 timestep= 641 max Sigma= 452.9651434697485 total t = 5069642216236.301 j = 90 Sigma[Inner] = 189.88026495774824
    dt= 8510309362.635579 timestep= 651 max Sigma= 452.3449811593769 total t = 5154718721394.555 j = 90 Sigma[Inner] = 190.52141341257126
    dt= 8516150231.925921 timestep= 661 max Sigma= 451.72784385505787 total t = 5239854007100.851 j = 90 Sigma[Inner] = 191.1811967626455
    dt= 8521911136.238471 timestep= 671 max Sigma= 451.11378679030446 total t = 5325047259058.027 j = 90 Sigma[Inner] = 191.85936695574537
    dt= 8527595176.719913 timestep= 681 max Sigma= 450.5028570857601 total t = 5410297694823.505 j = 90 Sigma[Inner] = 192.55561726237235
    dt= 8533205278.265079 timestep= 691 max Sigma= 449.8950945164235 total t = 5495604561985.945 j = 90 Sigma[Inner] = 193.26958596067837
    dt= 8538744202.10336 timestep= 701 max Sigma= 449.29053220004346 total t = 5580967136472.858 j = 90 Sigma[Inner] = 194.00086007437181
    dt= 8544214557.343858 timestep= 711 max Sigma= 448.6891972156694 total t = 5666384720978.266 j = 90 Sigma[Inner] = 194.7489791507008
    dt= 8549618811.577455 timestep= 721 max Sigma= 448.0911111602284 total t = 5751856643500.548 j = 90 Sigma[Inner] = 195.51343906312474
    dt= 8554959300.623623 timestep= 731 max Sigma= 447.4962906500352 total t = 5837382255981.542 j = 90 Sigma[Inner] = 196.29369582163443
    dt= 8560238237.500577 timestep= 741 max Sigma= 446.9047477733015 total t = 5922960933038.774 j = 90 Sigma[Inner] = 197.0891693727392
    dt= 8565457720.689113 timestep= 751 max Sigma= 446.31649049898385 total t = 6008592070783.454 j = 90 Sigma[Inner] = 197.8992473707925
    dt= 8570619741.753414 timestep= 761 max Sigma= 445.73152304667303 total t = 6094275085717.519 j = 90 Sigma[Inner] = 198.7232889024767
    dt= 8575726192.375663 timestep= 771 max Sigma= 445.14984622167276 total t = 6180009413703.625 j = 90 Sigma[Inner] = 199.56062814682426
    dt= 8580778870.855554 timestep= 781 max Sigma= 444.57145771893715 total t = 6265794509002.489 j = 90 Sigma[Inner] = 200.41057795403418
    dt= 8585779488.120962 timestep= 791 max Sigma= 443.99635239910975 total t = 6351629843372.513 j = 90 Sigma[Inner] = 201.27243332748967
    dt= 8590729673.29133 timestep= 801 max Sigma= 443.4245225395336 total t = 6437514905227.014 j = 90 Sigma[Inner] = 202.14547479472432
    dt= 8595630978.831398 timestep= 811 max Sigma= 442.8559580627885 total t = 6523449198844.813 j = 90 Sigma[Inner] = 203.02897165456662
    dt= 8600484885.329515 timestep= 821 max Sigma= 442.29064674501194 total t = 6609432243630.267 j = 90 Sigma[Inner] = 203.9221850892763
    dt= 8605292805.931221 timestep= 831 max Sigma= 441.7285744060223 total t = 6695463573419.147 j = 90 Sigma[Inner] = 204.8243711321174
    dt= 8610056090.456291 timestep= 841 max Sigma= 441.1697250830339 total t = 6781542735827.123 j = 90 Sigma[Inner] = 205.73478348246593
    dt= 8614776029.224607 timestep= 851 max Sigma= 440.6140811895636 total t = 6867669291637.7705 j = 90 Sigma[Inner] = 206.6526761621832
    dt= 8619453856.61408 timestep= 861 max Sigma= 440.0616236609543 total t = 6953842814227.349 j = 90 Sigma[Inner] = 207.5773060085865
    dt= 8624090754.371685 timestep= 871 max Sigma= 439.5123320877909 total t = 7040062889023.793 j = 90 Sigma[Inner] = 208.507935000877
    dt= 394240634.03273433 timestep= 881 max Sigma= 439.3501453907975 total t = 7057392295736.5205 j = 94 Sigma[Inner] = 82.13818322593035
    dt= 349898318.2698842 timestep= 891 max Sigma= 439.25935625388945 total t = 7071672848881.322 j = 94 Sigma[Inner] = 64.02057929552548
    dt= 1799622747.3964245 timestep= 901 max Sigma= 439.21788432827395 total t = 7079671075595.089 j = 94 Sigma[Inner] = 77.48504665058012
    dt= 4617792083.265932 timestep= 911 max Sigma= 438.99630750744626 total t = 7117533916092.871 j = 93 Sigma[Inner] = 66.21880112906473
    dt= 5311170243.951473 timestep= 921 max Sigma= 438.6838394185569 total t = 7167804680330.482 j = 93 Sigma[Inner] = 57.63012810127658
    dt= 8254495879.538884 timestep= 931 max Sigma= 438.3201185260181 total t = 7228685269262.13 j = 93 Sigma[Inner] = 51.47601222584168
    dt= 8638594736.96882 timestep= 941 max Sigma= 437.78319682853373 total t = 7315050937875.451 j = 93 Sigma[Inner] = 29.10518317715897
    dt= 8643074044.485897 timestep= 951 max Sigma= 437.24695490704374 total t = 7401461551975.111 j = 93 Sigma[Inner] = 30.609526229691912
    dt= 8647517052.68259 timestep= 961 max Sigma= 436.71374266372936 total t = 7487916758357.255 j = 93 Sigma[Inner] = 33.17776222849898
    dt= 8651925014.479029 timestep= 971 max Sigma= 436.18353841968747 total t = 7574416201128.411 j = 93 Sigma[Inner] = 37.54731998386333
    dt= 8656298990.396051 timestep= 981 max Sigma= 435.6563179344428 total t = 7660959535774.882 j = 93 Sigma[Inner] = 44.30178234367152
    dt= 8660639928.271193 timestep= 991 max Sigma= 435.132055328122 total t = 7747546427723.555 j = 93 Sigma[Inner] = 53.61715699758125
    dt= 8664948703.070044 timestep= 1001 max Sigma= 436.6592931275415 total t = 7834176551455.9375 j = 93 Sigma[Inner] = 65.18036698343307
    


    
![png](output_5_9.png)
    



    
![png](output_5_10.png)
    



    
![png](output_5_11.png)
    


    dt= 8669226137.684803 timestep= 1011 max Sigma= 438.45905273777225 total t = 7920849589904.443 j = 93 Sigma[Inner] = 78.30875256328483
    dt= 8673473014.395212 timestep= 1021 max Sigma= 440.25061456484207 total t = 8007565234000.008 j = 93 Sigma[Inner] = 92.17579264793967
    dt= 8677690081.563028 timestep= 1031 max Sigma= 442.0341029134314 total t = 8094323182305.57 j = 93 Sigma[Inner] = 106.02022256208238
    dt= 8681878057.770819 timestep= 1041 max Sigma= 443.80963883792015 total t = 8181123140701.203 j = 93 Sigma[Inner] = 119.26799968143618
    dt= 8686037634.506714 timestep= 1051 max Sigma= 445.5773402509426 total t = 8267964822101.997 j = 93 Sigma[Inner] = 131.5624455475615
    dt= 8690169477.970663 timestep= 1061 max Sigma= 447.33732202780135 total t = 8354847946197.609 j = 93 Sigma[Inner] = 142.73471491094153
    dt= 8694274230.329937 timestep= 1071 max Sigma= 449.0896961068023 total t = 8441772239206.655 j = 93 Sigma[Inner] = 152.7510801423079
    dt= 8698352510.63499 timestep= 1081 max Sigma= 450.8345715856233 total t = 8528737433641.616 j = 93 Sigma[Inner] = 161.66137322209923
    dt= 8702404915.549507 timestep= 1091 max Sigma= 452.5720548138517 total t = 8615743268081.792 j = 93 Sigma[Inner] = 169.5591317942787
    dt= 8706432020.014633 timestep= 1101 max Sigma= 454.3022494818534 total t = 8702789486953.1 j = 93 Sigma[Inner] = 176.55498285771495
    dt= 8710434377.94024 timestep= 1111 max Sigma= 456.02525670613886 total t = 8789875840314.597 j = 93 Sigma[Inner] = 182.76081641316685
    dt= 8714412522.98953 timestep= 1121 max Sigma= 457.7411751114075 total t = 8877002083652.377 j = 93 Sigma[Inner] = 188.2814407918687
    dt= 8718366969.496061 timestep= 1131 max Sigma= 459.4501009094443 total t = 8964167977682.05 j = 93 Sigma[Inner] = 193.2109031682512
    dt= 8722298213.526703 timestep= 1141 max Sigma= 461.1521279750469 total t = 9051373288161.145 j = 93 Sigma[Inner] = 197.63147990977538
    dt= 8726206734.08132 timestep= 1151 max Sigma= 462.84734791914605 total t = 9138617785712.959 j = 93 Sigma[Inner] = 201.61406504707818
    dt= 8730092994.40279 timestep= 1161 max Sigma= 464.5358501592778 total t = 9225901245663.004 j = 93 Sigma[Inner] = 205.21920394507583
    dt= 8733957443.35915 timestep= 1171 max Sigma= 466.2177219875448 total t = 9313223447889.002 j = 93 Sigma[Inner] = 208.49835321085894
    dt= 371450659.8576244 timestep= 1181 max Sigma= 466.70811343651866 total t = 9330393920031.424 j = 94 Sigma[Inner] = 84.32308492418277
    dt= 375460671.45217055 timestep= 1191 max Sigma= 466.81316087566194 total t = 9335872369564.488 j = 94 Sigma[Inner] = 76.03391772558602
    dt= 2718774081.905042 timestep= 1201 max Sigma= 466.9961101650735 total t = 9347753286988.521 j = 93 Sigma[Inner] = 73.04234660665247
    dt= 4638298843.544774 timestep= 1211 max Sigma= 467.74348022674087 total t = 9388679231050.945 j = 93 Sigma[Inner] = 65.10501614659748
    dt= 5156418224.724904 timestep= 1221 max Sigma= 468.6790781939467 total t = 9438132433412.58 j = 93 Sigma[Inner] = 57.88562304409645
    dt= 5569977670.109865 timestep= 1231 max Sigma= 469.6975954813009 total t = 9491951241620.01 j = 93 Sigma[Inner] = 53.07398569654569
    dt= 8744768253.382927 timestep= 1241 max Sigma= 470.940097183409 total t = 9560462315167.748 j = 93 Sigma[Inner] = 40.49051816443456
    dt= 8748553612.19098 timestep= 1251 max Sigma= 472.59735587401644 total t = 9647930825389.27 j = 93 Sigma[Inner] = 28.71437925255957
    dt= 8752320312.67664 timestep= 1261 max Sigma= 474.24837805486226 total t = 9735437095219.88 j = 93 Sigma[Inner] = 30.36997678116663
    dt= 8756066975.188538 timestep= 1271 max Sigma= 475.8932397543251 total t = 9822980921223.975 j = 93 Sigma[Inner] = 33.386593151665316
    dt= 8759794256.772057 timestep= 1281 max Sigma= 477.53201529923234 total t = 9910562106783.23 j = 93 Sigma[Inner] = 38.65023029791986
    dt= 8763502682.87728 timestep= 1291 max Sigma= 479.164777388179 total t = 9998180461054.65 j = 93 Sigma[Inner] = 46.81071182003669
    dt= 8767192709.541533 timestep= 1301 max Sigma= 480.7915971230318 total t = 10085835798033.09 j = 93 Sigma[Inner] = 57.971674696391844
    dt= 8770864754.171988 timestep= 1311 max Sigma= 482.41254404756916 total t = 10173527936043.742 j = 93 Sigma[Inner] = 71.62857891606768
    dt= 8774519211.229519 timestep= 1321 max Sigma= 484.0276861888434 total t = 10261256697450.582 j = 93 Sigma[Inner] = 86.86740248479808
    dt= 8778156460.38835 timestep= 1331 max Sigma= 485.63709009916073 total t = 10349021908476.115 j = 93 Sigma[Inner] = 102.67103962575992
    dt= 8781776870.772495 timestep= 1341 max Sigma= 487.24082089764335 total t = 10436823399079.14 j = 93 Sigma[Inner] = 118.16887865548831
    dt= 8785380803.03204 timestep= 1351 max Sigma= 488.8389423108588 total t = 10524661002861.998 j = 93 Sigma[Inner] = 132.7557948554103
    dt= 8788968610.164846 timestep= 1361 max Sigma= 490.43151671225394 total t = 10612534556991.127 j = 93 Sigma[Inner] = 146.0962386508675
    dt= 8792540637.591402 timestep= 1371 max Sigma= 492.01860516026534 total t = 10700443902121.422 j = 93 Sigma[Inner] = 158.0665681989833
    dt= 8796097222.806376 timestep= 1381 max Sigma= 493.60026743505466 total t = 10788388882318.795 j = 93 Sigma[Inner] = 168.68267677754838
    dt= 8799638694.841177 timestep= 1391 max Sigma= 495.1765620738689 total t = 10876369344978.016 j = 93 Sigma[Inner] = 178.0384730519812
    dt= 8803165373.71867 timestep= 1401 max Sigma= 496.7475464050612 total t = 10964385140735.037 j = 93 Sigma[Inner] = 186.2628291360908
    dt= 8806677570.03837 timestep= 1411 max Sigma= 498.3132765808344 total t = 11052436123374.479 j = 93 Sigma[Inner] = 193.49325095698106
    dt= 8810175584.78774 timestep= 1421 max Sigma= 499.87380760878983 total t = 11140522149734.125 j = 93 Sigma[Inner] = 199.8616111189511
    dt= 8813659709.4324 timestep= 1431 max Sigma= 501.4291933823706 total t = 11228643079609.082 j = 93 Sigma[Inner] = 205.48751350598667
    dt= 1280012636.087377 timestep= 1441 max Sigma= 502.8554802487323 total t = 11302198825613.896 j = 93 Sigma[Inner] = 103.34338361864417
    dt= 651623681.4075935 timestep= 1451 max Sigma= 502.96694062646145 total t = 11307919893091.465 j = 94 Sigma[Inner] = 126.43011095003862
    dt= 737753651.7350466 timestep= 1461 max Sigma= 503.042628351561 total t = 11312318632397.186 j = 94 Sigma[Inner] = 109.13396556331071
    dt= 259817742.68269634 timestep= 1471 max Sigma= 503.3853120809675 total t = 11331375262989.035 j = 94 Sigma[Inner] = 71.29197943688807
    dt= 890727063.669738 timestep= 1481 max Sigma= 503.47183734219317 total t = 11336941206988.316 j = 94 Sigma[Inner] = 120.01209457385829
    dt= 2896508989.9027557 timestep= 1491 max Sigma= 503.79457793136396 total t = 11357363054439.11 j = 93 Sigma[Inner] = 76.3990853331005
    dt= 3616751222.8068647 timestep= 1501 max Sigma= 504.3728703313856 total t = 11391115662915.084 j = 93 Sigma[Inner] = 70.83625760302803
    


    
![png](output_5_13.png)
    



    
![png](output_5_14.png)
    



    
![png](output_5_15.png)
    


    dt= 3971302449.3927426 timestep= 1511 max Sigma= 505.03410517043557 total t = 11429294979710.111 j = 93 Sigma[Inner] = 66.25336455894188
    dt= 4442157082.187537 timestep= 1521 max Sigma= 505.7563506059303 total t = 11471147421311.986 j = 93 Sigma[Inner] = 62.39056817054382
    dt= 8825850834.295681 timestep= 1531 max Sigma= 506.8827118085137 total t = 11540203141435.22 j = 93 Sigma[Inner] = 28.771732107156243
    dt= 8829277609.061024 timestep= 1541 max Sigma= 508.4154262086171 total t = 11628480504900.732 j = 93 Sigma[Inner] = 29.7078441970235
    dt= 8832691610.558344 timestep= 1551 max Sigma= 509.9432750138074 total t = 11716792068909.271 j = 93 Sigma[Inner] = 31.27028753708076
    dt= 8836092734.765085 timestep= 1561 max Sigma= 511.466305373154 total t = 11805137701565.46 j = 93 Sigma[Inner] = 33.29431903786273
    dt= 8839481531.84772 timestep= 1571 max Sigma= 512.9845635804427 total t = 11893517277284.16 j = 93 Sigma[Inner] = 36.141836866925395
    dt= 8842858408.86067 timestep= 1581 max Sigma= 514.4980950824338 total t = 11981930675114.871 j = 93 Sigma[Inner] = 40.27249707035185
    dt= 8846223699.405466 timestep= 1591 max Sigma= 516.0069444833 total t = 12070377777734.562 j = 93 Sigma[Inner] = 46.12853020815538
    dt= 8849577698.06454 timestep= 1601 max Sigma= 517.5111555574808 total t = 12158858470922.047 j = 93 Sigma[Inner] = 53.99199109502381
    dt= 8852920677.680082 timestep= 1611 max Sigma= 519.0107712663757 total t = 12247372643272.592 j = 93 Sigma[Inner] = 63.87467493291392
    dt= 8856252897.820686 timestep= 1621 max Sigma= 520.5058337766175 total t = 12335920186032.305 j = 93 Sigma[Inner] = 75.49541368914727
    dt= 8859574608.411633 timestep= 1631 max Sigma= 521.996384478755 total t = 12424500992988.326 j = 93 Sigma[Inner] = 88.352469615927
    dt= 8862886050.565855 timestep= 1641 max Sigma= 523.482464005707 total t = 12513114960378.69 j = 93 Sigma[Inner] = 101.8497401634522
    dt= 8866187455.816202 timestep= 1651 max Sigma= 524.9641122506304 total t = 12601761986800.936 j = 93 Sigma[Inner] = 115.41974904727205
    dt= 8869479044.587765 timestep= 1661 max Sigma= 526.4413683840222 total t = 12690441973108.227 j = 93 Sigma[Inner] = 128.60472174094411
    dt= 8872761024.559752 timestep= 1671 max Sigma= 527.9142708700086 total t = 12779154822289.129 j = 93 Sigma[Inner] = 141.08702091740062
    dt= 8876033589.415548 timestep= 1681 max Sigma= 529.3828574818666 total t = 12867900439332.803 j = 93 Sigma[Inner] = 152.68139288438041
    dt= 8879296918.321472 timestep= 1691 max Sigma= 530.847165316893 total t = 12956678731085.455 j = 93 Sigma[Inner] = 163.3077989188557
    dt= 8882551176.30926 timestep= 1701 max Sigma= 532.3072308107801 total t = 13045489606106.521 j = 93 Sigma[Inner] = 172.95988656170664
    dt= 8885796515.582146 timestep= 1711 max Sigma= 533.7630897516638 total t = 13134332974533.758 j = 93 Sigma[Inner] = 181.6774331778171
    dt= 8889033077.63694 timestep= 1721 max Sigma= 535.2147772940043 total t = 13223208747966.03 j = 93 Sigma[Inner] = 189.5256335320959
    dt= 8892260996.006569 timestep= 1731 max Sigma= 536.6623279724276 total t = 13312116839370.9 j = 93 Sigma[Inner] = 196.58102890048286
    dt= 8895480399.381742 timestep= 1741 max Sigma= 538.1057757156168 total t = 13401057163021.977 j = 93 Sigma[Inner] = 202.92268365946111
    dt= 8898691414.86251 timestep= 1751 max Sigma= 539.5451538603028 total t = 13490029634468.342 j = 93 Sigma[Inner] = 208.6270657993489
    dt= 317711084.3432891 timestep= 1761 max Sigma= 540.0787286414402 total t = 13514502095540.674 j = 94 Sigma[Inner] = 96.22471610075232
    dt= 434634250.47884166 timestep= 1771 max Sigma= 540.1410952835878 total t = 13518485471747.936 j = 94 Sigma[Inner] = 128.69534224651176
    dt= 2883061977.1228037 timestep= 1781 max Sigma= 540.3121289285746 total t = 13531539750977.95 j = 93 Sigma[Inner] = 72.9716671072519
    dt= 629168561.7071964 timestep= 1791 max Sigma= 540.4168516786535 total t = 13535781711814.467 j = 94 Sigma[Inner] = 103.93274081483794
    dt= 2359838183.694969 timestep= 1801 max Sigma= 540.6202807856038 total t = 13550135190488.053 j = 93 Sigma[Inner] = 77.96482689510123
    dt= 3443141558.026735 timestep= 1811 max Sigma= 541.0996164153449 total t = 13580982883534.973 j = 93 Sigma[Inner] = 72.62560074824906
    dt= 3813931650.444382 timestep= 1821 max Sigma= 541.6824285698447 total t = 13617585340117.06 j = 93 Sigma[Inner] = 67.90591687539313
    dt= 4119427493.055693 timestep= 1831 max Sigma= 542.3169718578447 total t = 13657391031411.111 j = 93 Sigma[Inner] = 64.01483901752812
    dt= 5765382720.660157 timestep= 1841 max Sigma= 543.0362439835003 total t = 13703877659784.299 j = 93 Sigma[Inner] = 60.54145356615154
    dt= 8909370704.582525 timestep= 1851 max Sigma= 544.3329333314251 total t = 13788033072058.947 j = 93 Sigma[Inner] = 28.694555735020533
    dt= 8912549197.115501 timestep= 1861 max Sigma= 545.7549968038745 total t = 13877144268205.73 j = 93 Sigma[Inner] = 30.101846971498556
    dt= 8915719115.420246 timestep= 1871 max Sigma= 547.1731597299897 total t = 13966287201515.508 j = 93 Sigma[Inner] = 31.840259281873877
    dt= 8918881057.938614 timestep= 1881 max Sigma= 548.5874518993973 total t = 14055461789745.434 j = 93 Sigma[Inner] = 34.20397153860713
    dt= 8922035434.256 timestep= 1891 max Sigma= 549.9979026650776 total t = 14144667955496.316 j = 93 Sigma[Inner] = 37.64267605048424
    dt= 8925182562.619768 timestep= 1901 max Sigma= 551.4045409317633 total t = 14233905624906.66 j = 93 Sigma[Inner] = 42.67045605126759
    dt= 8928322718.324121 timestep= 1911 max Sigma= 552.8073951560053 total t = 14323174727034.354 j = 93 Sigma[Inner] = 49.7167963478339
    dt= 8931456156.907482 timestep= 1921 max Sigma= 554.2064933516995 total t = 14412475193568.85 j = 93 Sigma[Inner] = 58.970236952640796
    dt= 8934583124.784037 timestep= 1931 max Sigma= 555.6018630980437 total t = 14501806958699.383 j = 93 Sigma[Inner] = 70.29108968018014
    dt= 8937703863.220179 timestep= 1941 max Sigma= 556.9935315483896 total t = 14591169959049.574 j = 93 Sigma[Inner] = 83.2381592859262
    dt= 8940818608.582632 timestep= 1951 max Sigma= 558.3815254391722 total t = 14680564133629.615 j = 93 Sigma[Inner] = 97.18916826029792
    dt= 8943927590.49167 timestep= 1961 max Sigma= 559.7658710984662 total t = 14769989423778.344 j = 93 Sigma[Inner] = 111.4908524920197
    dt= 8947031028.96406 timestep= 1971 max Sigma= 561.1465944539361 total t = 14859445773080.523 j = 93 Sigma[Inner] = 125.57796793046197
    dt= 8950129131.370155 timestep= 1981 max Sigma= 562.5237210400976 total t = 14948933127253.932 j = 93 Sigma[Inner] = 139.03377476720433
    dt= 8953222089.846243 timestep= 1991 max Sigma= 563.8972760049302 total t = 15038451434008.08 j = 93 Sigma[Inner] = 151.5974711517011
    dt= 8956310079.617432 timestep= 2001 max Sigma= 565.2672841159545 total t = 15128000642881.75 j = 93 Sigma[Inner] = 163.13970947497216
    


    
![png](output_5_17.png)
    



    
![png](output_5_18.png)
    



    
![png](output_5_19.png)
    


    dt= 8959393258.486738 timestep= 2011 max Sigma= 566.6337697659507 total t = 15217580705070.074 j = 93 Sigma[Inner] = 173.62690051904107
    dt= 8962471767.551294 timestep= 2021 max Sigma= 567.9967569785089 total t = 15307191573253.25 j = 93 Sigma[Inner] = 183.08739434077984
    dt= 8965545733.040585 timestep= 2031 max Sigma= 569.3562694136 total t = 15396833201438.742 j = 93 Sigma[Inner] = 191.58502829952627
    dt= 8968615269.052038 timestep= 2041 max Sigma= 570.71233037332 total t = 15486505544827.03 j = 93 Sigma[Inner] = 199.20074880326732
    dt= 8971680480.891253 timestep= 2051 max Sigma= 572.0649628079278 total t = 15576208559708.293 j = 93 Sigma[Inner] = 206.0209121841937
    dt= 514261147.60007674 timestep= 2061 max Sigma= 573.072511898369 total t = 15634734651971.19 j = 94 Sigma[Inner] = 134.11610502284645
    dt= 238727107.10362864 timestep= 2071 max Sigma= 573.1437561638809 total t = 15639201567919.09 j = 94 Sigma[Inner] = 101.55024084201902
    dt= 1057949441.8439877 timestep= 2081 max Sigma= 573.2153714819075 total t = 15644788658040.143 j = 94 Sigma[Inner] = 90.25361998672828
    dt= 435281840.6094322 timestep= 2091 max Sigma= 573.3189323865577 total t = 15651061887273.83 j = 94 Sigma[Inner] = 76.58728882862587
    dt= 1692161819.992775 timestep= 2101 max Sigma= 573.4401722144588 total t = 15660393808170.973 j = 93 Sigma[Inner] = 80.33604065322932
    dt= 3248240875.346443 timestep= 2111 max Sigma= 573.8176934396591 total t = 15687106857494.672 j = 93 Sigma[Inner] = 74.47770285242744
    dt= 3674847881.650305 timestep= 2121 max Sigma= 574.3368212527081 total t = 15722159050364.783 j = 93 Sigma[Inner] = 69.51884773045707
    dt= 3961726307.4617896 timestep= 2131 max Sigma= 574.9070633427258 total t = 15760524429866.314 j = 93 Sigma[Inner] = 65.51179473321609
    dt= 4268482075.7092805 timestep= 2141 max Sigma= 575.5188888166214 total t = 15801737332619.074 j = 93 Sigma[Inner] = 62.19382218635045
    dt= 8981364112.526814 timestep= 2151 max Sigma= 576.3246466738533 total t = 15860401888150.762 j = 93 Sigma[Inner] = 51.47165376382378
    dt= 8984399109.824717 timestep= 2161 max Sigma= 577.6632365864272 total t = 15950232157556.678 j = 93 Sigma[Inner] = 29.4790138395907
    dt= 8987445494.865635 timestep= 2171 max Sigma= 578.9985112192472 total t = 16040092907661.91 j = 93 Sigma[Inner] = 31.15628037025629
    dt= 8990487469.932945 timestep= 2181 max Sigma= 580.33049167809 total t = 16129984096891.527 j = 93 Sigma[Inner] = 33.27633836127086
    dt= 8993525495.997812 timestep= 2191 max Sigma= 581.6591985109948 total t = 16219905683846.64 j = 93 Sigma[Inner] = 36.21048006746176
    dt= 8996559891.352125 timestep= 2201 max Sigma= 582.9846520118465 total t = 16309857630866.236 j = 93 Sigma[Inner] = 40.466377652768365
    dt= 8999590906.954538 timestep= 2211 max Sigma= 584.3068722105675 total t = 16399839903060.18 j = 93 Sigma[Inner] = 46.56292571630785
    dt= 9002618762.609049 timestep= 2221 max Sigma= 585.625878871527 total t = 16489852467856.549 j = 93 Sigma[Inner] = 54.85849563703244
    dt= 9005643664.072659 timestep= 2231 max Sigma= 586.9416914958005 total t = 16579895294795.053 j = 93 Sigma[Inner] = 65.40546461573648
    dt= 9008665810.34483 timestep= 2241 max Sigma= 588.2543293251218 total t = 16669968355432.473 j = 93 Sigma[Inner] = 77.90728816328433
    dt= 9011685395.563723 timestep= 2251 max Sigma= 589.5638113463968 total t = 16760071623288.809 j = 93 Sigma[Inner] = 91.79751950579946
    dt= 9014702607.829279 timestep= 2261 max Sigma= 590.8701562961529 total t = 16850205073794.06 j = 93 Sigma[Inner] = 106.3920351606848
    dt= 9017717626.386906 timestep= 2271 max Sigma= 592.1733826645873 total t = 16940368684213.234 j = 93 Sigma[Inner] = 121.040138421765
    dt= 9020730618.22155 timestep= 2281 max Sigma= 593.473508699054 total t = 17030562433539.178 j = 93 Sigma[Inner] = 135.22302569850453
    dt= 9023741734.88823 timestep= 2291 max Sigma= 594.7705524069754 total t = 17120786302352.066 j = 93 Sigma[Inner] = 148.588421566632
    dt= 9026751110.197777 timestep= 2301 max Sigma= 596.0645315582641 total t = 17211040272651.467 j = 93 Sigma[Inner] = 160.93835956043435
    dt= 9029758859.148443 timestep= 2311 max Sigma= 597.3554636874154 total t = 17301324327671.895 j = 93 Sigma[Inner] = 172.1946148899869
    dt= 9032765078.259363 timestep= 2321 max Sigma= 598.6433660954713 total t = 17391638451695.363 j = 93 Sigma[Inner] = 182.36051611458362
    dt= 9035769847.250366 timestep= 2331 max Sigma= 599.9282558520512 total t = 17481982629874.78 j = 93 Sigma[Inner] = 191.48876629685412
    dt= 9038773231.849836 timestep= 2341 max Sigma= 601.210149797634 total t = 17572356848080.598 j = 93 Sigma[Inner] = 199.65804003068993
    dt= 9041775287.410652 timestep= 2351 max Sigma= 602.4890645462223 total t = 17662761092780.273 j = 93 Sigma[Inner] = 206.95757739598744
    dt= 327301844.6332422 timestep= 2361 max Sigma= 603.0911737202788 total t = 17696691134258.246 j = 94 Sigma[Inner] = 89.54508591116935
    dt= 361059915.6671599 timestep= 2371 max Sigma= 603.1441375474313 total t = 17700479027485.273 j = 94 Sigma[Inner] = 112.79456453198802
    dt= 270510427.65424335 timestep= 2381 max Sigma= 603.2441169417608 total t = 17707476210095.234 j = 94 Sigma[Inner] = 75.74920319897055
    dt= 635690252.2934297 timestep= 2391 max Sigma= 603.2975014992794 total t = 17711626548850.863 j = 94 Sigma[Inner] = 124.18288835626235
    dt= 2395771002.743388 timestep= 2401 max Sigma= 603.4818142882193 total t = 17726458085836.617 j = 93 Sigma[Inner] = 77.97735149243093
    dt= 3413610015.291917 timestep= 2411 max Sigma= 603.9018727542608 total t = 17757284485351.434 j = 93 Sigma[Inner] = 72.67829278234284
    dt= 3755437757.21076 timestep= 2421 max Sigma= 604.4059449633756 total t = 17793430727805.15 j = 93 Sigma[Inner] = 68.17821376668104
    dt= 4018040752.7921596 timestep= 2431 max Sigma= 604.9509229956448 total t = 17832445257370.18 j = 93 Sigma[Inner] = 64.52231177403142
    dt= 4332856900.034107 timestep= 2441 max Sigma= 605.5332282475268 total t = 17874214484997.305 j = 93 Sigma[Inner] = 61.47379527250427
    dt= 9051034823.393164 timestep= 2451 max Sigma= 606.4214196717281 total t = 17942257146691.332 j = 93 Sigma[Inner] = 28.56486834681925
    dt= 9054033768.336155 timestep= 2461 max Sigma= 607.688319551461 total t = 18032783986854.555 j = 93 Sigma[Inner] = 29.684290167633122
    dt= 9057031524.062767 timestep= 2471 max Sigma= 608.9523217019084 total t = 18123340813757.01 j = 93 Sigma[Inner] = 31.46472473244272
    dt= 9060027634.107962 timestep= 2481 max Sigma= 610.2134413312448 total t = 18213927608781.973 j = 93 Sigma[Inner] = 33.7679181545148
    dt= 9063022469.860735 timestep= 2491 max Sigma= 611.4716934914547 total t = 18304544357655.535 j = 93 Sigma[Inner] = 37.02285672659432
    dt= 9066016282.496185 timestep= 2501 max Sigma= 612.7270930757245 total t = 18395191049081.605 j = 93 Sigma[Inner] = 41.781787743428055
    


    
![png](output_5_21.png)
    



    
![png](output_5_22.png)
    



    
![png](output_5_23.png)
    


    dt= 9069009267.848879 timestep= 2511 max Sigma= 613.9796548095793 total t = 18485867673935.51 j = 93 Sigma[Inner] = 48.56990961827714
    dt= 9072001597.510231 timestep= 2521 max Sigma= 615.2293932486973 total t = 18576574224900.496 j = 93 Sigma[Inner] = 57.69770656802543
    dt= 9074993433.238052 timestep= 2531 max Sigma= 616.4763227798442 total t = 18667310696314.137 j = 93 Sigma[Inner] = 69.12408756985877
    dt= 9077984932.58769 timestep= 2541 max Sigma= 617.7204576231429 total t = 18758077084106.277 j = 93 Sigma[Inner] = 82.44504368427496
    dt= 9080976249.622711 timestep= 2551 max Sigma= 618.9618118347254 total t = 18848873385763.918 j = 93 Sigma[Inner] = 97.00833218653459
    dt= 9083967532.82604 timestep= 2561 max Sigma= 620.2003993092278 total t = 18939699600286.496 j = 93 Sigma[Inner] = 112.08627599623783
    dt= 9086958921.611225 timestep= 2571 max Sigma= 621.4362337818469 total t = 19030555728111.555 j = 93 Sigma[Inner] = 127.02667625895403
    dt= 9089950542.50414 timestep= 2581 max Sigma= 622.669328829838 total t = 19121441771002.92 j = 93 Sigma[Inner] = 141.33752144985922
    dt= 9092942505.837284 timestep= 2591 max Sigma= 623.899697873482 total t = 19212357731902.96 j = 93 Sigma[Inner] = 154.70513623855888
    dt= 9095934903.561396 timestep= 2601 max Sigma= 625.1273541766319 total t = 19303303614757.52 j = 93 Sigma[Inner] = 166.96977751165088
    dt= 9098927808.518072 timestep= 2611 max Sigma= 626.3523108470301 total t = 19394279424326.793 j = 93 Sigma[Inner] = 178.08489318836135
    dt= 9101921275.259447 timestep= 2621 max Sigma= 627.5745808365934 total t = 19485285165997.492 j = 93 Sigma[Inner] = 188.07733238087414
    dt= 9104915342.28071 timestep= 2631 max Sigma= 628.7941769418787 total t = 19576320845611.207 j = 93 Sigma[Inner] = 197.01599822008208
    dt= 9107910035.373293 timestep= 2641 max Sigma= 630.0111118048915 total t = 19667386469321.46 j = 93 Sigma[Inner] = 204.99011059784075
    dt= 887789302.4725136 timestep= 2651 max Sigma= 631.022986393224 total t = 19735059003634.76 j = 94 Sigma[Inner] = 56.11016226132952
    dt= 228659725.2881193 timestep= 2661 max Sigma= 631.0883304884896 total t = 19739306818124.1 j = 94 Sigma[Inner] = 121.45788884567143
    dt= 909267696.8537927 timestep= 2671 max Sigma= 631.1472083645739 total t = 19744409366628.434 j = 94 Sigma[Inner] = 96.7673804751631
    dt= 466846627.2343743 timestep= 2681 max Sigma= 631.2106044203915 total t = 19748728806614.848 j = 94 Sigma[Inner] = 100.21552568828695
    dt= 1718981620.3681347 timestep= 2691 max Sigma= 631.3210104499342 total t = 19758275312289.184 j = 93 Sigma[Inner] = 80.21410729869687
    dt= 3233392201.140494 timestep= 2701 max Sigma= 631.6573989747515 total t = 19785072340481.348 j = 93 Sigma[Inner] = 74.48890226909823
    dt= 3640365476.5802016 timestep= 2711 max Sigma= 632.1144998768845 total t = 19819862286594.633 j = 93 Sigma[Inner] = 69.67346743814538
    dt= 3907138371.665077 timestep= 2721 max Sigma= 632.6145403536626 total t = 19857778913705.438 j = 93 Sigma[Inner] = 65.81967523627512
    dt= 4162522665.7320533 timestep= 2731 max Sigma= 633.1477816679349 total t = 19898226634461.543 j = 93 Sigma[Inner] = 62.660507465826484
    dt= 6317119081.21265 timestep= 2741 max Sigma= 633.7466156670417 total t = 19945569925696.77 j = 93 Sigma[Inner] = 59.80877132149754
    dt= 9119937641.93968 timestep= 2751 max Sigma= 634.8720024825928 total t = 20033442184909.41 j = 93 Sigma[Inner] = 28.75868939120874
    dt= 9122937920.950499 timestep= 2761 max Sigma= 636.0757733038866 total t = 20124658063236.64 j = 93 Sigma[Inner] = 30.394337879919853
    dt= 9125938057.927881 timestep= 2771 max Sigma= 637.2769560125289 total t = 20215903943096.195 j = 93 Sigma[Inner] = 32.418794144392244
    dt= 9129238570.767141 timestep= 2782 max Sigma= 638.5952814804599 total t = 20316309064290.285 j = 93 Sigma[Inner] = 35.51966191550508
    dt= 9132239655.524061 timestep= 2792 max Sigma= 639.7910665741214 total t = 20407617955387.992 j = 93 Sigma[Inner] = 39.7119744339033
    dt= 9135241522.57419 timestep= 2802 max Sigma= 640.9842989856206 total t = 20498956861500.7 j = 93 Sigma[Inner] = 45.81313479243318
    dt= 9138244324.406528 timestep= 2812 max Sigma= 642.1749899263499 total t = 20590325791306.75 j = 93 Sigma[Inner] = 54.25203880691469
    dt= 9141248200.320309 timestep= 2822 max Sigma= 643.3631504953162 total t = 20681724754926.336 j = 93 Sigma[Inner] = 65.12655312266428
    dt= 9144253284.719738 timestep= 2832 max Sigma= 644.5487916799957 total t = 20773153763842.01 j = 93 Sigma[Inner] = 78.13820552516728
    dt= 9147259709.11427 timestep= 2842 max Sigma= 645.7319243578311 total t = 20864612830864.582 j = 93 Sigma[Inner] = 92.67321986719377
    dt= 9150267600.52332 timestep= 2852 max Sigma= 646.9125592977215 total t = 20956101970097.57 j = 93 Sigma[Inner] = 107.9765051955433
    dt= 9153277077.972778 timestep= 2862 max Sigma= 648.0907071611518 total t = 21047621196874.05 j = 93 Sigma[Inner] = 123.32823140447971
    dt= 9156288248.335493 timestep= 2872 max Sigma= 649.2663785027934 total t = 21139170527653.957 j = 93 Sigma[Inner] = 138.1586269182949
    dt= 9159301202.506208 timestep= 2882 max Sigma= 650.439583770564 total t = 21230749979880.867 j = 93 Sigma[Inner] = 152.086860328893
    dt= 9162316012.648033 timestep= 2892 max Sigma= 651.6103333052428 total t = 21322359571805.96 j = 93 Sigma[Inner] = 164.90503420619643
    dt= 9165332730.964146 timestep= 2902 max Sigma= 652.7786373398153 total t = 21413999322292.61 j = 93 Sigma[Inner] = 176.53713628626437
    dt= 9168351390.158949 timestep= 2912 max Sigma= 653.9445059987576 total t = 21505669250617.883 j = 93 Sigma[Inner] = 186.99506265264205
    dt= 9171372005.495407 timestep= 2922 max Sigma= 655.1079492974728 total t = 21597369376287.69 j = 93 Sigma[Inner] = 196.34249372497322
    dt= 9174394578.162333 timestep= 2932 max Sigma= 656.2689771420663 total t = 21689099718879.934 j = 93 Sigma[Inner] = 204.66925609847246
    dt= 886378391.7480042 timestep= 2942 max Sigma= 657.2342491320771 total t = 21757241787346.91 j = 94 Sigma[Inner] = 56.0764250871787
    dt= 227672022.58952263 timestep= 2952 max Sigma= 657.2960144397396 total t = 21761479388878.285 j = 94 Sigma[Inner] = 121.82046442904243
    dt= 898430590.5266875 timestep= 2962 max Sigma= 657.3514534835764 total t = 21766545492401.2 j = 94 Sigma[Inner] = 97.34827302028837
    dt= 461140583.54187554 timestep= 2972 max Sigma= 657.4108167205518 total t = 21770815216697.78 j = 94 Sigma[Inner] = 100.42031724757223
    dt= 1692279243.6758618 timestep= 2982 max Sigma= 657.5136345637786 total t = 21780200302668.31 j = 93 Sigma[Inner] = 80.40919273511525
    dt= 3220572087.955993 timestep= 2992 max Sigma= 657.829531622872 total t = 21806791077362.145 j = 93 Sigma[Inner] = 74.57928475900403
    dt= 3627220809.3415356 timestep= 3002 max Sigma= 658.260991826365 total t = 21841454857833.246 j = 93 Sigma[Inner] = 69.77896211591252
    


    
![png](output_5_25.png)
    



    
![png](output_5_26.png)
    



    
![png](output_5_27.png)
    


    dt= 3890500478.901824 timestep= 3012 max Sigma= 658.7329185808766 total t = 21879223195638.73 j = 93 Sigma[Inner] = 65.94954154695176
    dt= 4136636638.1965346 timestep= 3022 max Sigma= 659.2356499180307 total t = 21919462774127.402 j = 93 Sigma[Inner] = 62.81804856166093
    dt= 5140879146.367643 timestep= 3032 max Sigma= 659.7773109279962 total t = 21963603933075.0 j = 93 Sigma[Inner] = 60.11272596175156
    dt= 9186146333.603209 timestep= 3042 max Sigma= 660.755920465424 total t = 22045703111219.03 j = 93 Sigma[Inner] = 28.2728307619021
    dt= 9189180795.331247 timestep= 3052 max Sigma= 661.9052892006329 total t = 22137581263571.59 j = 93 Sigma[Inner] = 29.846942045468342
    dt= 9192216167.201054 timestep= 3062 max Sigma= 663.0522988514778 total t = 22229489765091.0 j = 93 Sigma[Inner] = 31.771215065618712
    dt= 9195252914.987968 timestep= 3072 max Sigma= 664.1969585075101 total t = 22321428627605.918 j = 93 Sigma[Inner] = 34.362185930129215
    dt= 9198291316.227566 timestep= 3082 max Sigma= 665.3392772087938 total t = 22413397866514.07 j = 93 Sigma[Inner] = 38.13628560137817
    dt= 9201331553.359306 timestep= 3092 max Sigma= 666.4792639286815 total t = 22505397499404.152 j = 93 Sigma[Inner] = 43.70117584022177
    dt= 9204373766.84069 timestep= 3102 max Sigma= 667.6169275665991 total t = 22597427545428.996 j = 93 Sigma[Inner] = 51.56681289667175
    dt= 9207418080.304813 timestep= 3112 max Sigma= 668.7522769457169 total t = 22689488025039.918 j = 93 Sigma[Inner] = 61.942384633295795
    dt= 9210464611.54483 timestep= 3122 max Sigma= 669.8853208128653 total t = 22781578959887.215 j = 93 Sigma[Inner] = 74.62699220747082
    dt= 9213513475.807371 timestep= 3132 max Sigma= 671.0160678393207 total t = 22873700372784.56 j = 93 Sigma[Inner] = 89.05551032904087
    dt= 9216564784.708548 timestep= 3142 max Sigma= 672.144526621711 total t = 22965852287680.246 j = 93 Sigma[Inner] = 104.46500736600474
    dt= 9219618642.763294 timestep= 3152 max Sigma= 673.2707056826218 total t = 23058034729603.133 j = 93 Sigma[Inner] = 120.08815547027815
    dt= 9222675142.96309 timestep= 3162 max Sigma= 674.3946134707014 total t = 23150247724567.574 j = 93 Sigma[Inner] = 135.2931818377189
    dt= 9225734362.531979 timestep= 3172 max Sigma= 675.5162583602174 total t = 23242491299434.273 j = 93 Sigma[Inner] = 149.642376098608
    dt= 9228796359.715801 timestep= 3182 max Sigma= 676.6356486501516 total t = 23334765481733.883 j = 93 Sigma[Inner] = 162.88540771394207
    dt= 9231861172.154133 timestep= 3192 max Sigma= 677.7527925629935 total t = 23427070299466.766 j = 93 Sigma[Inner] = 174.9193939053125
    dt= 9234928817.064865 timestep= 3202 max Sigma= 678.8676982434467 total t = 23519405780896.473 j = 93 Sigma[Inner] = 185.7419423043897
    dt= 9237999293.181715 timestep= 3212 max Sigma= 679.9803737572671 total t = 23611771954354.79 j = 93 Sigma[Inner] = 195.4110838145534
    dt= 9241072584.160881 timestep= 3222 max Sigma= 681.0908270904259 total t = 23704168848074.617 j = 93 Sigma[Inner] = 204.01625841411982
    dt= 1272127763.900982 timestep= 3232 max Sigma= 682.1092795056338 total t = 23781128625487.316 j = 93 Sigma[Inner] = 103.60610416749341
    dt= 264104195.71374238 timestep= 3242 max Sigma= 682.1798408810723 total t = 23786012064924.008 j = 94 Sigma[Inner] = 130.98732792171975
    dt= 727818732.3185834 timestep= 3252 max Sigma= 682.2266917379402 total t = 23790388049357.137 j = 94 Sigma[Inner] = 109.72823521027972
    dt= 447300983.67431617 timestep= 3262 max Sigma= 682.2811679769618 total t = 23794657032188.844 j = 94 Sigma[Inner] = 105.06605780615914
    dt= 1622354352.2234924 timestep= 3272 max Sigma= 682.3750233397899 total t = 23803671465506.805 j = 94 Sigma[Inner] = 80.91034520579522
    dt= 3200248143.8096666 timestep= 3282 max Sigma= 682.6690316948375 total t = 23829816249813.266 j = 93 Sigma[Inner] = 74.75425816316795
    dt= 3613322650.3286777 timestep= 3292 max Sigma= 683.0767494376429 total t = 23864322355947.293 j = 93 Sigma[Inner] = 69.93230499035468
    dt= 3876274803.2229447 timestep= 3302 max Sigma= 683.523228878651 total t = 23901952940316.22 j = 93 Sigma[Inner] = 66.09719535395728
    dt= 4117593125.756231 timestep= 3312 max Sigma= 683.9987311383886 total t = 23942029394377.434 j = 93 Sigma[Inner] = 62.966813821082326
    dt= 5113739678.454946 timestep= 3322 max Sigma= 684.5092693312993 total t = 23985840018840.027 j = 93 Sigma[Inner] = 60.279362947691546
    dt= 9253191422.482687 timestep= 3332 max Sigma= 685.4410369292547 total t = 24068233750387.52 j = 93 Sigma[Inner] = 28.190681983408037
    dt= 9256280693.762472 timestep= 3342 max Sigma= 686.5406337045729 total t = 24160782654470.5 j = 93 Sigma[Inner] = 29.810857756951936
    dt= 9259371608.141424 timestep= 3352 max Sigma= 687.6380543232856 total t = 24253362459857.023 j = 93 Sigma[Inner] = 31.791885232478375
    dt= 9262464622.11669 timestep= 3362 max Sigma= 688.7333061423411 total t = 24345973185652.59 j = 93 Sigma[Inner] = 34.463439049032786
    dt= 9265560002.444338 timestep= 3372 max Sigma= 689.8263964930838 total t = 24438614854433.586 j = 93 Sigma[Inner] = 38.36169869113785
    dt= 9268657920.154186 timestep= 3382 max Sigma= 690.9173326643094 total t = 24531287490854.953 j = 93 Sigma[Inner] = 44.11229227091702
    dt= 9271758504.635433 timestep= 3392 max Sigma= 692.0061218949944 total t = 24623991121023.043 j = 93 Sigma[Inner] = 52.22959520704815
    dt= 9274861869.17575 timestep= 3402 max Sigma= 693.092771371755 total t = 24716725772235.832 j = 93 Sigma[Inner] = 62.90602263916761
    dt= 9277968122.007774 timestep= 3412 max Sigma= 694.1772882284679 total t = 24809491472891.176 j = 93 Sigma[Inner] = 75.9060229615671
    dt= 9281077369.472473 timestep= 3422 max Sigma= 695.2596795467235 total t = 24902288252458.48 j = 93 Sigma[Inner] = 90.62480141206146
    dt= 9284189714.688889 timestep= 3432 max Sigma= 696.3399523563761 total t = 24995116141455.184 j = 93 Sigma[Inner] = 106.26861198399928
    dt= 9287305253.783037 timestep= 3442 max Sigma= 697.4181136357838 total t = 25087975171395.184 j = 93 Sigma[Inner] = 122.05538051951173
    dt= 9290424071.168402 timestep= 3452 max Sigma= 698.4941703115439 total t = 25180865374693.152 j = 93 Sigma[Inner] = 137.35375725319452
    dt= 9293546235.058002 timestep= 3462 max Sigma= 699.5681292576849 total t = 25273786784522.152 j = 93 Sigma[Inner] = 151.7361034110151
    dt= 9296671794.098797 timestep= 3472 max Sigma= 700.6399972943982 total t = 25366739434631.895 j = 93 Sigma[Inner] = 164.96587440622767
    dt= 9299800775.694239 timestep= 3482 max Sigma= 701.7097811864824 total t = 25459723359142.33 j = 93 Sigma[Inner] = 176.95362253494758
    dt= 9302933186.243107 timestep= 3492 max Sigma= 702.7774876417085 total t = 25552738592330.94 j = 93 Sigma[Inner] = 187.70812373141618
    dt= 9306069013.217031 timestep= 3502 max Sigma= 703.8431233093306 total t = 25645785168432.68 j = 93 Sigma[Inner] = 197.29587584976753
    


    
![png](output_5_29.png)
    



    
![png](output_5_30.png)
    



    
![png](output_5_31.png)
    


    dt= 9309208228.766209 timestep= 3512 max Sigma= 704.9066947789308 total t = 25738863121469.45 j = 93 Sigma[Inner] = 205.8123787351916
    dt= 691597743.980616 timestep= 3522 max Sigma= 705.5033209173562 total t = 25782550569110.773 j = 94 Sigma[Inner] = 128.6334841552889
    dt= 345355074.8097221 timestep= 3532 max Sigma= 705.5461434939201 total t = 25785961437042.473 j = 94 Sigma[Inner] = 123.71659851051915
    dt= 213636427.0623125 timestep= 3542 max Sigma= 705.6099110347697 total t = 25791425094822.395 j = 94 Sigma[Inner] = 81.10655019399677
    dt= 645458774.0374664 timestep= 3552 max Sigma= 705.654458918029 total t = 25795766299508.58 j = 94 Sigma[Inner] = 115.14912439562168
    dt= 2435763576.3798647 timestep= 3562 max Sigma= 705.80888538719 total t = 25811111285035.066 j = 93 Sigma[Inner] = 77.83835490754244
    dt= 3401504763.3650055 timestep= 3572 max Sigma= 706.1498921109609 total t = 25842024020244.867 j = 93 Sigma[Inner] = 72.65463881058952
    dt= 3726716684.1251054 timestep= 3582 max Sigma= 706.5549933296052 total t = 25877953176968.58 j = 93 Sigma[Inner] = 68.28981002557019
    dt= 3971322733.670036 timestep= 3592 max Sigma= 706.9914117643733 total t = 25916588644130.812 j = 93 Sigma[Inner] = 64.76838598444861
    dt= 4228523588.8687673 timestep= 3602 max Sigma= 707.4549772945326 total t = 25957664028209.73 j = 93 Sigma[Inner] = 61.85566321824649
    dt= 7820974059.669355 timestep= 3612 max Sigma= 708.0261920290893 total t = 26011608625239.92 j = 93 Sigma[Inner] = 55.49417738714817
    dt= 9321565806.69928 timestep= 3622 max Sigma= 709.0647373721644 total t = 26104810108157.883 j = 93 Sigma[Inner] = 28.692369438727496
    dt= 9324723207.848085 timestep= 3632 max Sigma= 710.118230009267 total t = 26198043130215.316 j = 93 Sigma[Inner] = 30.462704276280995
    dt= 9327882954.111021 timestep= 3642 max Sigma= 711.1696964757269 total t = 26291307738779.242 j = 93 Sigma[Inner] = 32.706946857703215
    dt= 9331045411.340124 timestep= 3652 max Sigma= 712.2191428859842 total t = 26384603959497.082 j = 93 Sigma[Inner] = 35.85935362055067
    dt= 9334210788.534517 timestep= 3662 max Sigma= 713.266575336058 total t = 26477931820712.33 j = 93 Sigma[Inner] = 40.52723080213295
    dt= 9337379224.216808 timestep= 3672 max Sigma= 714.3119998912744 total t = 26571291352423.074 j = 93 Sigma[Inner] = 47.32968099618152
    dt= 9340550827.723654 timestep= 3682 max Sigma= 715.3554225812284 total t = 26664682585829.426 j = 93 Sigma[Inner] = 56.67097594012473
    dt= 9343725698.341736 timestep= 3692 max Sigma= 716.3968493981645 total t = 26758105553159.62 j = 93 Sigma[Inner] = 68.5560173551571
    dt= 9346903932.850428 timestep= 3702 max Sigma= 717.4362862968561 total t = 26851560287617.965 j = 93 Sigma[Inner] = 82.55530376949375
    dt= 9350085626.576418 timestep= 3712 max Sigma= 718.4737391949654 total t = 26945046823369.55 j = 93 Sigma[Inner] = 97.93415207239698
    dt= 9353270870.749388 timestep= 3722 max Sigma= 719.5092139733061 total t = 27038565195513.387 j = 93 Sigma[Inner] = 113.86299821413672
    dt= 9356459748.000662 timestep= 3732 max Sigma= 720.542716475702 total t = 27132115440017.566 j = 93 Sigma[Inner] = 129.60289070838817
    dt= 9359652327.419762 timestep= 3742 max Sigma= 721.5742525083166 total t = 27225697593605.977 j = 93 Sigma[Inner] = 144.60694931960663
    dt= 9362848660.282724 timestep= 3752 max Sigma= 722.60382783847 total t = 27319311693598.42 j = 93 Sigma[Inner] = 158.53822038770545
    dt= 9366048777.24596 timestep= 3762 max Sigma= 723.6314481930671 total t = 27412957777715.66 j = 93 Sigma[Inner] = 171.23633756827692
    dt= 9369252687.444536 timestep= 3772 max Sigma= 724.657119256828 total t = 27506635883866.523 j = 93 Sigma[Inner] = 182.66674340914173
    dt= 9372460379.58625 timestep= 3782 max Sigma= 725.6808466705431 total t = 27600346049937.17 j = 93 Sigma[Inner] = 192.8733734471787
    dt= 9375671824.840603 timestep= 3792 max Sigma= 726.7026360295556 total t = 27694088313601.344 j = 93 Sigma[Inner] = 201.94284856308087
    dt= 9378886981.11792 timestep= 3802 max Sigma= 727.7224928826516 total t = 27787862712167.57 j = 93 Sigma[Inner] = 209.9805879359115
    dt= 443825051.7536236 timestep= 3812 max Sigma= 727.8971197113 total t = 27795005487701.594 j = 94 Sigma[Inner] = 146.20994034307495
    dt= 541235986.8828857 timestep= 3822 max Sigma= 727.9373904235771 total t = 27798811727414.43 j = 94 Sigma[Inner] = 134.22495328929278
    dt= 394523780.8542407 timestep= 3832 max Sigma= 727.9856226910201 total t = 27803107505634.594 j = 94 Sigma[Inner] = 91.7907780854131
    dt= 1337480110.6958632 timestep= 3842 max Sigma= 728.0582684969418 total t = 27810742462028.113 j = 94 Sigma[Inner] = 83.83494833194699
    dt= 3095150674.4293833 timestep= 3852 max Sigma= 728.3020658415169 total t = 27834965663884.16 j = 93 Sigma[Inner] = 75.43434131425938
    dt= 3575482520.185913 timestep= 3862 max Sigma= 728.665606601632 total t = 27868967626366.08 j = 93 Sigma[Inner] = 70.45495903575959
    dt= 3846813498.16509 timestep= 3872 max Sigma= 729.066943143924 total t = 27906276617640.98 j = 93 Sigma[Inner] = 66.52014765866609
    dt= 4086523619.985653 timestep= 3882 max Sigma= 729.4950206958927 total t = 27946057684446.664 j = 93 Sigma[Inner] = 63.31855491889511
    dt= 4457101581.148113 timestep= 3892 max Sigma= 729.9509737082767 total t = 27988585260286.594 j = 93 Sigma[Inner] = 60.60675844151167
    dt= 9388325590.746752 timestep= 3902 max Sigma= 730.7011324476997 total t = 28062964843015.38 j = 93 Sigma[Inner] = 27.654109819601228
    dt= 9391557464.975292 timestep= 3912 max Sigma= 731.7134439976852 total t = 28156865871312.383 j = 93 Sigma[Inner] = 29.197699887388385
    dt= 9394791929.628113 timestep= 3922 max Sigma= 732.7238499705691 total t = 28250799233377.88 j = 93 Sigma[Inner] = 31.129300863146213
    dt= 9398029195.110714 timestep= 3932 max Sigma= 733.7323555476796 total t = 28344764955178.85 j = 93 Sigma[Inner] = 33.69193275535852
    dt= 9401269551.899477 timestep= 3942 max Sigma= 734.7389659120112 total t = 28438763066459.023 j = 93 Sigma[Inner] = 37.41393656541141
    dt= 9404513173.898018 timestep= 3952 max Sigma= 735.7436862334171 total t = 28532793599149.875 j = 93 Sigma[Inner] = 42.944947569330374
    dt= 9407760184.026281 timestep= 3962 max Sigma= 736.746521659971 total t = 28626856586605.266 j = 93 Sigma[Inner] = 50.85226801480307
    dt= 9411010685.306946 timestep= 3972 max Sigma= 737.7474773145484 total t = 28720952063282.027 j = 93 Sigma[Inner] = 61.38927714749815
    dt= 9414264774.603842 timestep= 3982 max Sigma= 738.7465582938206 total t = 28815080064626.67 j = 93 Sigma[Inner] = 74.36155609390671
    dt= 9417522547.010706 timestep= 3992 max Sigma= 739.7437696682208 total t = 28909240627043.637 j = 93 Sigma[Inner] = 89.16970485460062
    dt= 9420784094.923973 timestep= 4002 max Sigma= 740.7391164820948 total t = 29003433787875.703 j = 93 Sigma[Inner] = 104.99385401662506
    


    
![png](output_5_33.png)
    



    
![png](output_5_34.png)
    



    
![png](output_5_35.png)
    


    dt= 9424049504.186 timestep= 4012 max Sigma= 741.7326037535936 total t = 29097659585357.5 j = 93 Sigma[Inner] = 121.01109291195584
    dt= 9427318849.004274 timestep= 4022 max Sigma= 742.7242364740889 total t = 29191918058522.41 j = 93 Sigma[Inner] = 136.55116024182027
    dt= 9430592186.99134 timestep= 4032 max Sigma= 743.7140196070573 total t = 29286209247058.37 j = 93 Sigma[Inner] = 151.15853080277725
    dt= 9433869555.349598 timestep= 4042 max Sigma= 744.7019580865056 total t = 29380533191119.645 j = 93 Sigma[Inner] = 164.5812971468746
    dt= 9437150968.865274 timestep= 4052 max Sigma= 745.6880568150974 total t = 29474889931110.055 j = 93 Sigma[Inner] = 176.72449172191148
    dt= 9440436419.996761 timestep= 4062 max Sigma= 746.6723206621914 total t = 29569279507457.527 j = 93 Sigma[Inner] = 187.59754202508228
    dt= 9443725880.99604 timestep= 4072 max Sigma= 747.6547544620096 total t = 29663701960400.926 j = 93 Sigma[Inner] = 197.2707457175526
    dt= 9447019307.734411 timestep= 4082 max Sigma= 748.6353630121325 total t = 29758157329807.797 j = 93 Sigma[Inner] = 205.8445761744862
    dt= 691817144.9485306 timestep= 4092 max Sigma= 749.1845880026391 total t = 29802395148115.902 j = 94 Sigma[Inner] = 128.58887231935537
    dt= 345840576.56409967 timestep= 4102 max Sigma= 749.223532355927 total t = 29805809641205.5 j = 94 Sigma[Inner] = 123.59633833868477
    dt= 232215021.28044492 timestep= 4112 max Sigma= 749.2881206410613 total t = 29811933372297.266 j = 94 Sigma[Inner] = 78.4272568756973
    dt= 603690923.325263 timestep= 4122 max Sigma= 749.3254874136409 total t = 29815913823422.055 j = 94 Sigma[Inner] = 124.84973285101682
    dt= 2304182816.2835026 timestep= 4132 max Sigma= 749.4527950683382 total t = 29829912136976.555 j = 93 Sigma[Inner] = 78.19546684619753
    dt= 3372365302.6291065 timestep= 4142 max Sigma= 749.7545777744812 total t = 29860145891892.57 j = 93 Sigma[Inner] = 73.00482534809319
    dt= 3711534446.2734675 timestep= 4152 max Sigma= 750.1204897737181 total t = 29895874768165.89 j = 93 Sigma[Inner] = 68.5533033561253
    dt= 3960711147.3724027 timestep= 4162 max Sigma= 750.5157733123726 total t = 29934387095636.73 j = 93 Sigma[Inner] = 64.96196111689959
    dt= 4219091526.8742156 timestep= 4172 max Sigma= 750.9360564998107 total t = 29975365945775.63 j = 93 Sigma[Inner] = 61.99555801648903
    dt= 7165951956.020312 timestep= 4182 max Sigma= 751.42781299108 total t = 30026006883850.215 j = 93 Sigma[Inner] = 58.66588039795645
    dt= 9459618158.077213 timestep= 4192 max Sigma= 752.3612897285931 total t = 30118977390104.48 j = 92 Sigma[Inner] = 28.255289032291664
    dt= 9462932172.35325 timestep= 4202 max Sigma= 753.3331723247578 total t = 30213591796746.633 j = 92 Sigma[Inner] = 30.034211885664348
    dt= 9466248921.402092 timestep= 4212 max Sigma= 754.3032570210618 total t = 30308239358133.117 j = 92 Sigma[Inner] = 32.27188594151021
    dt= 9469568800.467377 timestep= 4222 max Sigma= 755.2715482923672 total t = 30402920103992.438 j = 92 Sigma[Inner] = 35.40178177352777
    dt= 9472892027.893068 timestep= 4232 max Sigma= 756.238050615552 total t = 30497634066920.832 j = 92 Sigma[Inner] = 40.04621870385604
    dt= 9476218742.535418 timestep= 4242 max Sigma= 757.2027684565658 total t = 30592381281206.82 j = 92 Sigma[Inner] = 46.85090951325324
    dt= 9479549050.383057 timestep= 4252 max Sigma= 758.1657062649234 total t = 30687161782321.094 j = 92 Sigma[Inner] = 56.24526418934066
    dt= 9482883046.198185 timestep= 4262 max Sigma= 759.1268684717088 total t = 30781975606721.19 j = 92 Sigma[Inner] = 68.24288035106932
    dt= 9486220822.152023 timestep= 4272 max Sigma= 760.0862594891336 total t = 30876822791794.484 j = 92 Sigma[Inner] = 82.40073113729797
    dt= 9489562469.210363 timestep= 4282 max Sigma= 761.0438837106076 total t = 30971703375844.34 j = 92 Sigma[Inner] = 97.955533651709
    dt= 9492908074.386747 timestep= 4292 max Sigma= 761.9997455107404 total t = 31066617398065.457 j = 92 Sigma[Inner] = 114.0474210378276
    dt= 9496257715.905272 timestep= 4302 max Sigma= 762.9538492449554 total t = 31161564898478.74 j = 92 Sigma[Inner] = 129.9158222726878
    dt= 9499611457.835672 timestep= 4312 max Sigma= 763.9061992485905 total t = 31256545917813.758 j = 92 Sigma[Inner] = 145.00423954474743
    dt= 9502969345.431227 timestep= 4322 max Sigma= 764.8567998354914 total t = 31351560497340.617 j = 92 Sigma[Inner] = 158.9759418723611
    dt= 9506331402.047915 timestep= 4332 max Sigma= 765.8056552962266 total t = 31446608678663.25 j = 92 Sigma[Inner] = 171.67650992665804
    dt= 9509697628.132109 timestep= 4342 max Sigma= 766.7527698961069 total t = 31541690503493.06 j = 92 Sigma[Inner] = 183.07947392984252
    dt= 9513068002.380186 timestep= 4352 max Sigma= 767.6981478732266 total t = 31636806013424.477 j = 92 Sigma[Inner] = 193.23677537566877
    dt= 9516442484.850548 timestep= 4362 max Sigma= 768.6417934367354 total t = 31731955249733.12 j = 92 Sigma[Inner] = 202.24193046373563
    dt= 9519821021.584412 timestep= 4372 max Sigma= 769.5837107655094 total t = 31827138253213.81 j = 92 Sigma[Inner] = 210.20585689319412
    dt= 444248711.68313944 timestep= 4382 max Sigma= 769.7440080289539 total t = 31834281930735.016 j = 94 Sigma[Inner] = 146.0934126033537
    dt= 543294292.351166 timestep= 4392 max Sigma= 769.7807431639333 total t = 31838099013812.574 j = 94 Sigma[Inner] = 133.8486639125638
    dt= 308188468.3255452 timestep= 4402 max Sigma= 769.8351339643787 total t = 31843369474375.484 j = 94 Sigma[Inner] = 77.761544194396
    dt= 1032098813.019702 timestep= 4412 max Sigma= 769.8898807294514 total t = 31849635692650.555 j = 94 Sigma[Inner] = 90.36702037883272
    dt= 2935846161.3797374 timestep= 4422 max Sigma= 770.083893019906 total t = 31871185678981.84 j = 93 Sigma[Inner] = 76.25302133856829
    dt= 3532342820.734709 timestep= 4432 max Sigma= 770.4066876816322 total t = 31904488508844.496 j = 93 Sigma[Inner] = 71.08947750438846
    dt= 3819993422.573243 timestep= 4442 max Sigma= 770.768559517083 total t = 31941470943805.92 j = 93 Sigma[Inner] = 66.99653535535694
    dt= 4064996043.8572593 timestep= 4452 max Sigma= 771.1558112454851 total t = 31981018467538.58 j = 93 Sigma[Inner] = 63.67082872484032
    dt= 4399806713.3642645 timestep= 4462 max Sigma= 771.5685116398325 total t = 32023277533464.85 j = 93 Sigma[Inner] = 60.86753544581189
    dt= 9529442804.731962 timestep= 4472 max Sigma= 772.2529582252841 total t = 32098023177273.926 j = 92 Sigma[Inner] = 27.439802986196014
    dt= 9532839031.127823 timestep= 4482 max Sigma= 773.188270296344 total t = 32193336281218.223 j = 92 Sigma[Inner] = 29.06395232979549
    dt= 9536238139.622252 timestep= 4492 max Sigma= 774.1218741720443 total t = 32288683364285.36 j = 92 Sigma[Inner] = 31.090326156640298
    dt= 9539640298.397247 timestep= 4502 max Sigma= 775.0537737630848 total t = 32384064454890.125 j = 92 Sigma[Inner] = 33.780864013718016
    


    
![png](output_5_37.png)
    



    
![png](output_5_38.png)
    



    
![png](output_5_39.png)
    


    dt= 9543045803.401775 timestep= 4512 max Sigma= 775.9839729938261 total t = 32479479585307.547 j = 92 Sigma[Inner] = 37.69244348565779
    dt= 9546454828.55254 timestep= 4522 max Sigma= 776.912475788515 total t = 32574928790021.098 j = 92 Sigma[Inner] = 43.49824040550584
    dt= 9549867494.78501 timestep= 4532 max Sigma= 777.8392860628683 total t = 32670412104923.906 j = 92 Sigma[Inner] = 51.764695929885235
    dt= 9553283902.742588 timestep= 4542 max Sigma= 778.7644077206538 total t = 32765929566989.188 j = 92 Sigma[Inner] = 62.709431266581014
    dt= 9556704147.18981 timestep= 4552 max Sigma= 779.6878446525868 total t = 32861481214157.273 j = 92 Sigma[Inner] = 76.07670855563346
    dt= 9560128321.568846 timestep= 4562 max Sigma= 780.6096007361517 total t = 32957067085307.895 j = 92 Sigma[Inner] = 91.20455107840361
    dt= 9563556516.941484 timestep= 4572 max Sigma= 781.5296798355986 total t = 33052687220244.56 j = 92 Sigma[Inner] = 107.23196297378774
    dt= 9566988817.83183 timestep= 4582 max Sigma= 782.448085801685 total t = 33148341659649.656 j = 92 Sigma[Inner] = 123.3249713418612
    dt= 9570425296.776937 timestep= 4592 max Sigma= 783.3648224709557 total t = 33244030444989.516 j = 92 Sigma[Inner] = 138.8267779223436
    dt= 9573866009.017105 timestep= 4602 max Sigma= 784.2798936645052 total t = 33339753618364.703 j = 92 Sigma[Inner] = 153.30807614537662
    dt= 9577310988.417774 timestep= 4612 max Sigma= 785.193303186295 total t = 33435511222313.277 j = 92 Sigma[Inner] = 166.54544826746906
    dt= 9580760245.329296 timestep= 4622 max Sigma= 786.105054821185 total t = 33531303299583.473 j = 92 Sigma[Inner] = 178.4685995747202
    dt= 9584213766.683344 timestep= 4632 max Sigma= 787.0151523328789 total t = 33627129892897.223 j = 92 Sigma[Inner] = 189.1057008664546
    dt= 9587671518.253094 timestep= 4642 max Sigma= 787.9235994620019 total t = 33722991044726.72 j = 92 Sigma[Inner] = 198.5400788275661
    dt= 9591133448.718945 timestep= 4652 max Sigma= 788.8303999244944 total t = 33818886797103.91 j = 92 Sigma[Inner] = 206.88065055637065
    dt= 686489658.6779394 timestep= 4662 max Sigma= 789.3304213736674 total t = 33862948195201.85 j = 94 Sigma[Inner] = 129.5475732902413
    dt= 346511994.92742634 timestep= 4672 max Sigma= 789.3658505306101 total t = 33866363911325.11 j = 94 Sigma[Inner] = 123.55871567321893
    dt= 234392223.9270975 timestep= 4682 max Sigma= 789.4249344382629 total t = 33872515714962.33 j = 94 Sigma[Inner] = 78.30888800103062
    dt= 608843353.0006607 timestep= 4692 max Sigma= 789.4592762798296 total t = 33876531420963.48 j = 94 Sigma[Inner] = 124.17055126865338
    dt= 2323549003.8672194 timestep= 4702 max Sigma= 789.5767148492566 total t = 33890700224275.996 j = 93 Sigma[Inner] = 78.14520647431428
    dt= 3383680349.1852016 timestep= 4712 max Sigma= 789.8529993773436 total t = 33921073078068.96 j = 93 Sigma[Inner] = 72.9264644585798
    dt= 3725254246.626265 timestep= 4722 max Sigma= 790.1874666603888 total t = 33956926172982.47 j = 93 Sigma[Inner] = 68.44997486222053
    dt= 3979766236.8261576 timestep= 4732 max Sigma= 790.5490356853446 total t = 33995601722510.58 j = 93 Sigma[Inner] = 64.82795401457763
    dt= 4254927968.934213 timestep= 4742 max Sigma= 790.9341962053675 total t = 34036841490322.934 j = 93 Sigma[Inner] = 61.825005502390205
    dt= 9486013497.454348 timestep= 4752 max Sigma= 791.4399356276157 total t = 34095918429917.2 j = 93 Sigma[Inner] = 51.39145345816052
    dt= 9604621409.915615 timestep= 4762 max Sigma= 792.339289416095 total t = 34191948986711.02 j = 92 Sigma[Inner] = 28.48046899189629
    dt= 9608104962.88391 timestep= 4772 max Sigma= 793.2380953219595 total t = 34288014358112.79 j = 92 Sigma[Inner] = 30.431323585909563
    dt= 9611591492.246092 timestep= 4782 max Sigma= 794.1352760710236 total t = 34384114581016.105 j = 92 Sigma[Inner] = 32.93237155157055
    dt= 7352625690.211802 timestep= 4792 max Sigma= 795.0308351442487 total t = 34477987231674.33 j = 92 Sigma[Inner] = 36.49042438150207
    dt= 7352625690.211726 timestep= 4802 max Sigma= 795.7144747100724 total t = 34551513488576.438 j = 93 Sigma[Inner] = 40.36326115997475
    dt= 7352625690.211765 timestep= 4812 max Sigma= 796.3969803933304 total t = 34625039745478.547 j = 93 Sigma[Inner] = 45.546220052738626
    dt= 7352625690.211785 timestep= 4822 max Sigma= 797.0783545399576 total t = 34698566002380.656 j = 93 Sigma[Inner] = 52.24179778071373
    dt= 7352625690.211796 timestep= 4832 max Sigma= 797.7585994870423 total t = 34772092259282.766 j = 93 Sigma[Inner] = 60.50918930250702
    dt= 7352625690.211802 timestep= 4842 max Sigma= 798.4377175628894 total t = 34845618516184.875 j = 93 Sigma[Inner] = 70.23975395546263
    dt= 7552048908.6078205 timestep= 4852 max Sigma= 799.1220190901769 total t = 34920028793939.906 j = 93 Sigma[Inner] = 81.2805011581711
    dt= 7756644118.471223 timestep= 4862 max Sigma= 799.8261535461592 total t = 34996724630618.652 j = 93 Sigma[Inner] = 93.55310264352866
    dt= 7910043293.937703 timestep= 4872 max Sigma= 800.5456092183406 total t = 35075169822345.434 j = 93 Sigma[Inner] = 106.59594555742144
    dt= 8027655981.694473 timestep= 4882 max Sigma= 801.2762856212272 total t = 35154941642683.508 j = 93 Sigma[Inner] = 119.90971328690028
    dt= 8120127618.3226 timestep= 4892 max Sigma= 802.0153133210846 total t = 35235744131885.875 j = 93 Sigma[Inner] = 133.0537112852585
    dt= 8194690816.60953 timestep= 4902 max Sigma= 802.7606650367327 total t = 35317367968432.99 j = 93 Sigma[Inner] = 145.6884080316797
    dt= 8256265845.004197 timestep= 4912 max Sigma= 803.510885119713 total t = 35399662672466.45 j = 93 Sigma[Inner] = 157.58542256927333
    dt= 8308233857.723432 timestep= 4922 max Sigma= 804.2649081596941 total t = 35482517983450.04 j = 93 Sigma[Inner] = 168.61560892368254
    dt= 8352951804.681954 timestep= 4932 max Sigma= 805.021938457044 total t = 35565851478794.664 j = 93 Sigma[Inner] = 178.7272231500259
    dt= 8392089472.90007 timestep= 4942 max Sigma= 805.7813696215007 total t = 35649600303671.41 j = 93 Sigma[Inner] = 187.92289247959326
    dt= 8426849942.629997 timestep= 4952 max Sigma= 806.542730384626 total t = 35733715588364.14 j = 93 Sigma[Inner] = 196.23994356439167
    dt= 8458114824.442361 timestep= 4962 max Sigma= 807.3056475918519 total t = 35818158629238.63 j = 93 Sigma[Inner] = 203.73552815378184
    dt= 1738334601.952196 timestep= 4972 max Sigma= 808.0698205564585 total t = 35896150031553.03 j = 93 Sigma[Inner] = 199.6770391552796
    dt= 515877663.3134903 timestep= 4982 max Sigma= 808.1341026342998 total t = 35902061402336.016 j = 94 Sigma[Inner] = 146.34074853977856
    dt= 600679867.2872279 timestep= 4992 max Sigma= 808.1711204651624 total t = 35906254872254.57 j = 94 Sigma[Inner] = 124.63848091882248
    dt= 313867300.91992784 timestep= 5002 max Sigma= 808.2274086358069 total t = 35912216269054.79 j = 94 Sigma[Inner] = 75.72079032396475
    


    
![png](output_5_41.png)
    



    
![png](output_5_42.png)
    



    
![png](output_5_43.png)
    


    dt= 1098066968.6971881 timestep= 5012 max Sigma= 808.2793998176227 total t = 35918772487477.516 j = 94 Sigma[Inner] = 88.32150945358082
    dt= 2985061154.9321833 timestep= 5022 max Sigma= 808.4630471961868 total t = 35941053485325.44 j = 93 Sigma[Inner] = 76.01619339848818
    dt= 3558405824.0491276 timestep= 5032 max Sigma= 808.7604066092748 total t = 35974667895872.76 j = 93 Sigma[Inner] = 70.84857561208423
    dt= 3848613519.3073854 timestep= 5042 max Sigma= 809.0927859172924 total t = 36011919217574.17 j = 93 Sigma[Inner] = 66.73731922421302
    dt= 4104888116.405185 timestep= 5052 max Sigma= 809.4488485808608 total t = 36051804097886.12 j = 93 Sigma[Inner] = 63.382321815460394
    dt= 5101848737.864369 timestep= 5062 max Sigma= 809.832726816935 total t = 36095564734346.88 j = 93 Sigma[Inner] = 60.508966307136895
    dt= 8562838255.034059 timestep= 5072 max Sigma= 810.5022185991139 total t = 36173701757248.22 j = 93 Sigma[Inner] = 23.110692167489802
    dt= 8583681008.165477 timestep= 5082 max Sigma= 811.2693807880843 total t = 36259445845216.24 j = 93 Sigma[Inner] = 25.575580536780755
    dt= 8603312205.516893 timestep= 5092 max Sigma= 812.0368829661924 total t = 36345391559314.945 j = 93 Sigma[Inner] = 28.16350660158758
    dt= 8621883305.690386 timestep= 5102 max Sigma= 812.8046165179953 total t = 36431527642096.125 j = 93 Sigma[Inner] = 31.135387695517426
    dt= 8639518728.212284 timestep= 5112 max Sigma= 813.5724859604217 total t = 36517844196452.73 j = 93 Sigma[Inner] = 34.91455710063695
    dt= 8656321898.554544 timestep= 5122 max Sigma= 814.3404066777099 total t = 36604332449706.52 j = 93 Sigma[Inner] = 40.01194802373184
    dt= 8672379754.58557 timestep= 5132 max Sigma= 815.1083031542321 total t = 36690984569563.17 j = 93 Sigma[Inner] = 46.90303207417547
    dt= 8687766114.596596 timestep= 5142 max Sigma= 815.8761075769669 total t = 36777793518546.18 j = 93 Sigma[Inner] = 55.87730906930871
    dt= 8702544232.219034 timestep= 5152 max Sigma= 816.6437587163203 total t = 36864752937420.03 j = 93 Sigma[Inner] = 66.92640801476895
    dt= 8716768760.081028 timestep= 5162 max Sigma= 817.4112010195289 total t = 36951857050759.56 j = 93 Sigma[Inner] = 79.72839547420548
    dt= 8730487276.930141 timestep= 5172 max Sigma= 818.1783838684812 total t = 37039100589648.38 j = 93 Sigma[Inner] = 93.73324898152359
    dt= 8743741488.168037 timestep= 5182 max Sigma= 818.9452609661687 total t = 37126478727775.836 j = 93 Sigma[Inner] = 108.30206057193278
    dt= 8756568179.173132 timestep= 5192 max Sigma= 819.7117898248399 total t = 37213987028124.2 j = 93 Sigma[Inner] = 122.83789948622592
    dt= 8768999979.51773 timestep= 5202 max Sigma= 820.4779313353636 total t = 37301621398107.96 j = 93 Sigma[Inner] = 136.86839940912108
    dt= 8781065981.144436 timestep= 5212 max Sigma= 821.243649402046 total t = 37389378051521.5 j = 93 Sigma[Inner] = 150.0733322595477
    dt= 8792792242.772165 timestep= 5222 max Sigma= 822.0089106306779 total t = 37477253476019.84 j = 93 Sigma[Inner] = 162.2725144862237
    dt= 8804202204.949963 timestep= 5232 max Sigma= 822.7736840602388 total t = 37565244405134.48 j = 93 Sigma[Inner] = 173.39484309706967
    dt= 8815317034.392672 timestep= 5242 max Sigma= 823.5379409307036 total t = 37653347794037.32 j = 93 Sigma[Inner] = 183.44425206058577
    dt= 8826155911.922293 timestep= 5252 max Sigma= 824.3016544809467 total t = 37741560798427.7 j = 93 Sigma[Inner] = 192.4708104500321
    dt= 8836736275.094841 timestep= 5262 max Sigma= 825.0647997719176 total t = 37829880756041.31 j = 93 Sigma[Inner] = 200.54943296555012
    dt= 8847074024.131245 timestep= 5272 max Sigma= 825.827353531204 total t = 37918305170377.42 j = 93 Sigma[Inner] = 207.7656340989352
    dt= 312661999.8411387 timestep= 5282 max Sigma= 826.1762765444875 total t = 37950284848988.086 j = 94 Sigma[Inner] = 99.26445963945778
    dt= 438712199.0996365 timestep= 5292 max Sigma= 826.209201017105 total t = 37954236015593.055 j = 94 Sigma[Inner] = 135.09823666194848
    dt= 208529862.94237638 timestep= 5302 max Sigma= 826.2696491722937 total t = 37961029420072.555 j = 94 Sigma[Inner] = 75.3518451326096
    dt= 696644584.0742838 timestep= 5312 max Sigma= 826.3053542082408 total t = 37965666703106.94 j = 94 Sigma[Inner] = 109.95655674126968
    dt= 2569398934.356889 timestep= 5322 max Sigma= 826.4332984982308 total t = 37982410383594.16 j = 93 Sigma[Inner] = 77.4778261545777
    dt= 3455468148.318675 timestep= 5332 max Sigma= 826.6984545169702 total t = 38014130587778.02 j = 93 Sigma[Inner] = 72.1830108237236
    dt= 3781458239.3710093 timestep= 5342 max Sigma= 827.0089460379515 total t = 38050589220465.914 j = 93 Sigma[Inner] = 67.78610354697864
    dt= 4042476129.397081 timestep= 5352 max Sigma= 827.3437746907048 total t = 38089847359329.164 j = 93 Sigma[Inner] = 64.20822314024352
    dt= 4387356863.725881 timestep= 5362 max Sigma= 827.7019275684477 total t = 38131943085617.555 j = 93 Sigma[Inner] = 61.205914397766435
    dt= 8878863469.117506 timestep= 5372 max Sigma= 828.2803549480749 total t = 38203942760760.13 j = 93 Sigma[Inner] = 28.736662606729695
    dt= 8888330739.76216 timestep= 5382 max Sigma= 829.0401844408743 total t = 38292783616645.414 j = 93 Sigma[Inner] = 30.126201280760657
    dt= 8897617382.744719 timestep= 5392 max Sigma= 829.7993237272414 total t = 38381718145623.72 j = 93 Sigma[Inner] = 31.86761430602425
    dt= 8906733283.774635 timestep= 5402 max Sigma= 830.5577571676998 total t = 38470744593718.87 j = 93 Sigma[Inner] = 34.1358235569829
    dt= 8915687882.464155 timestep= 5412 max Sigma= 831.315470148112 total t = 38559861306284.02 j = 93 Sigma[Inner] = 37.34652741070095
    dt= 8924489711.923056 timestep= 5422 max Sigma= 832.0724489910945 total t = 38649066717883.03 j = 93 Sigma[Inner] = 42.026318820256776
    dt= 8933146588.957037 timestep= 5432 max Sigma= 832.8286808812189 total t = 38738359344350.1 j = 93 Sigma[Inner] = 48.67271638134292
    dt= 8941665733.742706 timestep= 5442 max Sigma= 833.5841538039659 total t = 38827737776326.25 j = 93 Sigma[Inner] = 57.581135427811695
    dt= 8950053851.609632 timestep= 5452 max Sigma= 834.338856493695 total t = 38917200673765.805 j = 93 Sigma[Inner] = 68.71806094759607
    dt= 8958317193.157976 timestep= 5462 max Sigma= 835.0927783880021 total t = 39006746761132.12 j = 93 Sigma[Inner] = 81.70790230971001
    dt= 8966461601.049274 timestep= 5472 max Sigma= 835.8459095869076 total t = 39096374823116.445 j = 93 Sigma[Inner] = 95.93573455187303
    dt= 8974492547.94381 timestep= 5482 max Sigma= 836.5982408158749 total t = 39186083700773.23 j = 93 Sigma[Inner] = 110.70644287946176
    dt= 8982415168.13996 timestep= 5492 max Sigma= 837.3497633919642 total t = 39275872287998.555 j = 93 Sigma[Inner] = 125.3875926801486
    dt= 8990234284.507719 timestep= 5502 max Sigma= 838.1004691926228 total t = 39365739528298.06 j = 93 Sigma[Inner] = 139.49336990414076
    


    
![png](output_5_45.png)
    



    
![png](output_5_46.png)
    



    
![png](output_5_47.png)
    


    dt= 8997954431.813225 timestep= 5512 max Sigma= 838.8503506267178 total t = 39455684411803.516 j = 93 Sigma[Inner] = 152.70651811484842
    dt= 9005579877.258057 timestep= 5522 max Sigma= 839.5994006075159 total t = 39545705972506.79 j = 93 Sigma[Inner] = 164.858751439794
    dt= 9013114638.882362 timestep= 5532 max Sigma= 840.3476125273775 total t = 39635803285686.914 j = 93 Sigma[Inner] = 175.89373742633052
    dt= 9020562502.345322 timestep= 5542 max Sigma= 841.0949802339861 total t = 39725975465511.93 j = 93 Sigma[Inner] = 185.82941448943396
    dt= 9027927036.476904 timestep= 5552 max Sigma= 841.8414980079747 total t = 39816221662801.54 j = 93 Sigma[Inner] = 194.72747489828353
    dt= 9035935790.996916 timestep= 5563 max Sigma= 842.6616795644798 total t = 39915576998730.766 j = 93 Sigma[Inner] = 203.41719751993836
    dt= 1751998168.272499 timestep= 5573 max Sigma= 843.4063956713167 total t = 39998684882097.82 j = 93 Sigma[Inner] = 198.90355025955216
    dt= 517312541.79495263 timestep= 5583 max Sigma= 843.4677505618698 total t = 40004905834783.41 j = 94 Sigma[Inner] = 146.08801272958962
    dt= 605179268.416567 timestep= 5593 max Sigma= 843.501720821175 total t = 40009122194882.46 j = 94 Sigma[Inner] = 124.00814482590935
    dt= 316804564.81086636 timestep= 5603 max Sigma= 843.5537350211162 total t = 40015155925082.49 j = 94 Sigma[Inner] = 75.58922882568714
    dt= 1118470151.4947414 timestep= 5613 max Sigma= 843.6018018818866 total t = 40021800695751.336 j = 94 Sigma[Inner] = 87.76781590564752
    dt= 3000309721.037701 timestep= 5623 max Sigma= 843.7713485094811 total t = 40044298507551.27 j = 93 Sigma[Inner] = 75.94137781903697
    dt= 3569284613.368028 timestep= 5633 max Sigma= 844.0438994311089 total t = 40078027085309.95 j = 93 Sigma[Inner] = 70.76134374008011
    dt= 3861711427.6541243 timestep= 5643 max Sigma= 844.3484270097906 total t = 40115397461146.64 j = 93 Sigma[Inner] = 66.63170387592322
    dt= 4124180271.8761344 timestep= 5653 max Sigma= 844.674890205044 total t = 40155441512085.17 j = 93 Sigma[Inner] = 63.255965176835936
    dt= 5559879644.740506 timestep= 5663 max Sigma= 845.0330036230843 total t = 40200554649050.56 j = 93 Sigma[Inner] = 60.311127695659785
    dt= 9064867313.184536 timestep= 5673 max Sigma= 845.697191270775 total t = 40285173013710.96 j = 93 Sigma[Inner] = 28.7452946559215
    dt= 9071789396.460968 timestep= 5683 max Sigma= 846.4383487849328 total t = 40375859813010.19 j = 93 Sigma[Inner] = 30.343724804494546
    dt= 9078646853.157495 timestep= 5693 max Sigma= 847.1786265934232 total t = 40466615474968.695 j = 93 Sigma[Inner] = 32.30271214455384
    dt= 9085442717.072187 timestep= 5703 max Sigma= 847.9180217107418 total t = 40557439370445.73 j = 93 Sigma[Inner] = 34.94031134434202
    dt= 9092179576.965223 timestep= 5713 max Sigma= 848.6565314786922 total t = 40648330898038.47 j = 93 Sigma[Inner] = 38.752502903970566
    dt= 9098859740.052065 timestep= 5723 max Sigma= 849.3941535323078 total t = 40739289480582.62 j = 93 Sigma[Inner] = 44.306875797817625
    dt= 9105485323.211342 timestep= 5733 max Sigma= 850.130885776563 total t = 40830314562885.0 j = 93 Sigma[Inner] = 52.0662569188327
    dt= 9112058303.118387 timestep= 5743 max Sigma= 850.8667263688918 total t = 40921405610119.53 j = 93 Sigma[Inner] = 62.20734548291458
    dt= 9118580545.250147 timestep= 5753 max Sigma= 851.6016737049553 total t = 41012562106597.02 j = 93 Sigma[Inner] = 74.52843899595658
    dt= 9125053821.53834 timestep= 5763 max Sigma= 852.3357264063217 total t = 41103783554757.15 j = 93 Sigma[Inner] = 88.495910726328
    dt= 9131479821.576895 timestep= 5773 max Sigma= 853.0688833093309 total t = 41195069474298.72 j = 93 Sigma[Inner] = 103.39616285732184
    dt= 9137860159.938414 timestep= 5783 max Sigma= 853.8011434547223 total t = 41286419401398.95 j = 93 Sigma[Inner] = 118.51063541263588
    dt= 9144196381.02249 timestep= 5793 max Sigma= 854.532506077767 total t = 41377832887991.66 j = 93 Sigma[Inner] = 133.24348668939436
    dt= 9150489962.31854 timestep= 5803 max Sigma= 855.2629705987486 total t = 41469309501084.38 j = 93 Sigma[Inner] = 147.1767929438613
    dt= 9156742316.7022 timestep= 5813 max Sigma= 855.9925366136872 total t = 41560848822102.586 j = 93 Sigma[Inner] = 160.0666899722717
    dt= 9162954794.239685 timestep= 5823 max Sigma= 856.7212038852591 total t = 41652450446253.66 j = 93 Sigma[Inner] = 171.80830621357632
    dt= 9169128683.869183 timestep= 5833 max Sigma= 857.448972333887 total t = 41744113981907.984 j = 93 Sigma[Inner] = 182.3930768259747
    dt= 9175265215.23285 timestep= 5843 max Sigma= 858.1758420290113 total t = 41835839049997.63 j = 93 Sigma[Inner] = 191.87149016766222
    dt= 9181365560.836678 timestep= 5853 max Sigma= 858.9018131805634 total t = 41927625283434.84 j = 93 Sigma[Inner] = 200.32557870754988
    dt= 9187430838.624668 timestep= 5863 max Sigma= 859.6268861306794 total t = 42019472326554.56 j = 93 Sigma[Inner] = 207.85069712268447
    dt= 324867222.6301002 timestep= 5873 max Sigma= 859.9614582313012 total t = 42053047464691.19 j = 94 Sigma[Inner] = 89.89815620164609
    dt= 374396567.90332973 timestep= 5883 max Sigma= 859.9911436014224 total t = 42056864432463.414 j = 94 Sigma[Inner] = 119.82462513910983
    dt= 235068193.2243901 timestep= 5893 max Sigma= 860.0472828070373 total t = 42063850627586.35 j = 94 Sigma[Inner] = 75.54956712911478
    dt= 633246500.1097149 timestep= 5903 max Sigma= 860.0766891750412 total t = 42067981672318.75 j = 94 Sigma[Inner] = 123.96490094394157
    dt= 2389683275.9138856 timestep= 5913 max Sigma= 860.1792859015111 total t = 42082763916257.34 j = 93 Sigma[Inner] = 77.99251846742504
    dt= 3407443750.0381546 timestep= 5923 max Sigma= 860.4135146870108 total t = 42113532306264.42 j = 93 Sigma[Inner] = 72.71418804890224
    dt= 3747350580.607856 timestep= 5933 max Sigma= 860.6946667997897 total t = 42149606699663.89 j = 93 Sigma[Inner] = 68.23429391430192
    dt= 4006878781.430489 timestep= 5943 max Sigma= 860.9985971637735 total t = 42188525294261.055 j = 93 Sigma[Inner] = 64.59852253368038
    dt= 4307296636.376248 timestep= 5953 max Sigma= 861.3230739771154 total t = 42230132047437.23 j = 93 Sigma[Inner] = 61.57053622790072
    dt= 9205324382.566525 timestep= 5963 max Sigma= 861.7853590233881 total t = 42293938826368.95 j = 93 Sigma[Inner] = 42.66921766224314
    dt= 9211256249.909464 timestep= 5973 max Sigma= 862.5068582050732 total t = 42386024685319.016 j = 93 Sigma[Inner] = 29.229224130088788
    dt= 9217164891.804201 timestep= 5983 max Sigma= 863.2274628321508 total t = 42478169771155.41 j = 93 Sigma[Inner] = 31.027489466722407
    dt= 9223043030.476242 timestep= 5993 max Sigma= 863.947173848127 total t = 42570373774406.99 j = 93 Sigma[Inner] = 33.34208990024777
    dt= 9228891988.61084 timestep= 6003 max Sigma= 864.665992240366 total t = 42662636397584.31 j = 93 Sigma[Inner] = 36.61531609575582
    


    
![png](output_5_49.png)
    



    
![png](output_5_50.png)
    



    
![png](output_5_51.png)
    


    dt= 9234712842.622908 timestep= 6013 max Sigma= 865.383919104048 total t = 42754957354951.25 j = 93 Sigma[Inner] = 41.42933034850184
    dt= 9240506528.57857 timestep= 6023 max Sigma= 866.1009556236289 total t = 42847336370702.8 j = 93 Sigma[Inner] = 48.34504185326213
    dt= 9246273897.366575 timestep= 6033 max Sigma= 866.8171030607316 total t = 42939773177893.63 j = 93 Sigma[Inner] = 57.69560494492622
    dt= 9252015744.327837 timestep= 6043 max Sigma= 867.532362745461 total t = 43032267517764.11 j = 93 Sigma[Inner] = 69.43412729780779
    dt= 9257732825.547575 timestep= 6053 max Sigma= 868.2467360696148 total t = 43124819139282.31 j = 93 Sigma[Inner] = 83.12177399343362
    dt= 9263425866.83557 timestep= 6063 max Sigma= 868.9602244809737 total t = 43217427798805.31 j = 93 Sigma[Inner] = 98.05675347466641
    dt= 9269095568.458961 timestep= 6073 max Sigma= 869.6728294782197 total t = 43310093259805.16 j = 93 Sigma[Inner] = 113.46568130272802
    dt= 9274742607.277637 timestep= 6083 max Sigma= 870.3845526062158 total t = 43402815292626.82 j = 93 Sigma[Inner] = 128.66665806625758
    dt= 9280367637.267115 timestep= 6093 max Sigma= 871.0953954514802 total t = 43495593674258.086 j = 93 Sigma[Inner] = 143.15638268974982
    dt= 9285971289.10199 timestep= 6103 max Sigma= 871.8053596377599 total t = 43588428188098.9 j = 93 Sigma[Inner] = 156.6240142430853
    dt= 9291554169.313923 timestep= 6113 max Sigma= 872.5144468216506 total t = 43681318623723.945 j = 93 Sigma[Inner] = 168.92051608272902
    dt= 9297116859.431133 timestep= 6123 max Sigma= 873.2226586882466 total t = 43774264776636.336 j = 93 Sigma[Inner] = 180.01277015521367
    dt= 9302659915.40848 timestep= 6133 max Sigma= 873.9299969468332 total t = 43867266448014.08 j = 93 Sigma[Inner] = 189.94066922361142
    dt= 9308183867.556522 timestep= 6143 max Sigma= 874.6364633266497 total t = 43960323444453.51 j = 93 Sigma[Inner] = 198.78435524397383
    dt= 9313689221.076632 timestep= 6153 max Sigma= 875.3420595727715 total t = 44053435577715.28 j = 93 Sigma[Inner] = 206.6421159200062
    dt= 326338457.8001506 timestep= 6163 max Sigma= 875.6730421434177 total t = 44088181899129.41 j = 94 Sigma[Inner] = 89.67836516535873
    dt= 375500003.98183066 timestep= 6173 max Sigma= 875.701646147771 total t = 44092012630345.64 j = 94 Sigma[Inner] = 119.62098049714612
    dt= 235198616.16098022 timestep= 6183 max Sigma= 875.755829869222 total t = 44099036379576.96 j = 94 Sigma[Inner] = 75.4633907203146
    dt= 634518545.8145726 timestep= 6193 max Sigma= 875.7840959925372 total t = 44103173396970.086 j = 94 Sigma[Inner] = 123.93529467299992
    dt= 2392721222.397969 timestep= 6203 max Sigma= 875.8828029025888 total t = 44117985850525.54 j = 93 Sigma[Inner] = 77.98567323745785
    dt= 3407832414.677264 timestep= 6213 max Sigma= 876.1077913412944 total t = 44148768284550.39 j = 93 Sigma[Inner] = 72.70709593794193
    dt= 3747154335.630228 timestep= 6223 max Sigma= 876.3777174039383 total t = 44184843070658.75 j = 93 Sigma[Inner] = 68.2309002071866
    dt= 4006167528.6310835 timestep= 6233 max Sigma= 876.6694847155954 total t = 44223757006964.4 j = 93 Sigma[Inner] = 64.59899882802922
    dt= 4305039070.28229 timestep= 6243 max Sigma= 876.9809359710333 total t = 44265350657945.1 j = 93 Sigma[Inner] = 61.57515385896056
    dt= 9329874596.314592 timestep= 6253 max Sigma= 877.4256023649584 total t = 44329389295345.71 j = 93 Sigma[Inner] = 42.557078320870396
    dt= 9335306830.452023 timestep= 6263 max Sigma= 878.1277694566161 total t = 44422717897780.35 j = 93 Sigma[Inner] = 29.030747836631342
    dt= 9340729520.374418 timestep= 6273 max Sigma= 878.829075376221 total t = 44516100805416.625 j = 93 Sigma[Inner] = 30.886516014700227
    dt= 9346135112.452667 timestep= 6283 max Sigma= 879.5295218632942 total t = 44609537845093.28 j = 93 Sigma[Inner] = 33.28142058150055
    dt= 9351524446.768198 timestep= 6293 max Sigma= 880.2291106117615 total t = 44703028850687.47 j = 93 Sigma[Inner] = 36.68246644215433
    dt= 9356898151.369507 timestep= 6303 max Sigma= 880.927843333719 total t = 44796573663200.46 j = 93 Sigma[Inner] = 41.69909597717182
    dt= 9362256740.430752 timestep= 6313 max Sigma= 881.6257217457129 total t = 44890172129231.26 j = 93 Sigma[Inner] = 48.90670461883768
    dt= 9367600665.037308 timestep= 6323 max Sigma= 882.3227475607385 total t = 44983824100143.81 j = 93 Sigma[Inner] = 58.62607336307616
    dt= 9372930340.128075 timestep= 6333 max Sigma= 883.0189224832616 total t = 45077529431598.26 j = 93 Sigma[Inner] = 70.76968262384966
    dt= 9378246158.968512 timestep= 6343 max Sigma= 883.7142482058878 total t = 45171287983276.39 j = 93 Sigma[Inner] = 84.84402639892674
    dt= 9383548500.75964 timestep= 6353 max Sigma= 884.4087264069434 total t = 45265099618710.945 j = 93 Sigma[Inner] = 100.09900842416995
    dt= 9388837734.233809 timestep= 6363 max Sigma= 885.1023587485588 total t = 45358964205167.35 j = 93 Sigma[Inner] = 115.73338599830932
    dt= 9394114218.782276 timestep= 6373 max Sigma= 885.7951468750032 total t = 45452881613546.79 j = 93 Sigma[Inner] = 131.0601883126693
    dt= 9399378304.05456 timestep= 6383 max Sigma= 886.4870924111252 total t = 45546851718291.54 j = 93 Sigma[Inner] = 145.5875166536111
    dt= 9404630328.690819 timestep= 6393 max Sigma= 887.1781969608028 total t = 45640874397281.086 j = 93 Sigma[Inner] = 159.02365237410396
    dt= 9409870618.704586 timestep= 6403 max Sigma= 887.8684621053576 total t = 45734949531713.25 j = 93 Sigma[Inner] = 171.2395624850081
    dt= 9415099485.931044 timestep= 6413 max Sigma= 888.5578894019234 total t = 45829077005969.15 j = 93 Sigma[Inner] = 182.21933765572882
    dt= 9420317226.855669 timestep= 6423 max Sigma= 889.246480381782 total t = 45923256707464.6 j = 93 Sigma[Inner] = 192.01611178804973
    dt= 9425524122.03143 timestep= 6433 max Sigma= 889.9342365487016 total t = 46017488526492.82 j = 93 Sigma[Inner] = 200.71950824495826
    dt= 9430720436.185078 timestep= 6443 max Sigma= 890.621159377325 total t = 46111772356065.13 j = 93 Sigma[Inner] = 208.4342664103122
    dt= 312589443.2398276 timestep= 6453 max Sigma= 890.8708812273978 total t = 46136969686872.44 j = 94 Sigma[Inner] = 97.84009388874506
    dt= 436887281.48518485 timestep= 6463 max Sigma= 890.8986345226779 total t = 46140909324054.34 j = 94 Sigma[Inner] = 135.24324670681062
    dt= 207562066.8041096 timestep= 6473 max Sigma= 890.9493863796433 total t = 46147657702736.66 j = 94 Sigma[Inner] = 75.44770474172462
    dt= 692034461.9284251 timestep= 6483 max Sigma= 890.9794012014262 total t = 46152269255943.734 j = 94 Sigma[Inner] = 110.42068855729715
    dt= 2558742860.917023 timestep= 6493 max Sigma= 891.086751081884 total t = 46168899158151.56 j = 93 Sigma[Inner] = 77.50998565439187
    dt= 3449494622.3742332 timestep= 6503 max Sigma= 891.3103053197532 total t = 46200546659176.14 j = 93 Sigma[Inner] = 72.22802505277268
    


    
![png](output_5_53.png)
    



    
![png](output_5_54.png)
    



    
![png](output_5_55.png)
    


    dt= 3774398330.330739 timestep= 6513 max Sigma= 891.5723110797165 total t = 46236940801115.35 j = 93 Sigma[Inner] = 67.84191728913417
    dt= 4032463051.1999617 timestep= 6523 max Sigma= 891.8547956701769 total t = 46276114425453.39 j = 93 Sigma[Inner] = 64.27714932539672
    dt= 4355432227.067261 timestep= 6533 max Sigma= 892.1565963407004 total t = 46318045061759.41 j = 93 Sigma[Inner] = 61.29265657785492
    dt= 9445778673.151688 timestep= 6543 max Sigma= 892.6157816048598 total t = 46386501181004.59 j = 93 Sigma[Inner] = 39.46422842735344
    dt= 9450934850.7907 timestep= 6553 max Sigma= 893.299458511906 total t = 46480987313865.39 j = 93 Sigma[Inner] = 27.42898058526463
    dt= 9456085233.1472 timestep= 6563 max Sigma= 893.982309054226 total t = 46575524998271.72 j = 93 Sigma[Inner] = 28.994432727874628
    dt= 9461225355.966656 timestep= 6573 max Sigma= 894.6643344608583 total t = 46670114129449.414 j = 93 Sigma[Inner] = 31.2009816645802
    dt= 9466355838.086231 timestep= 6583 max Sigma= 895.3455359159782 total t = 46764754608416.72 j = 93 Sigma[Inner] = 34.56827940478118
    dt= 9471477110.509975 timestep= 6593 max Sigma= 896.0259145915124 total t = 46859446341246.164 j = 93 Sigma[Inner] = 39.750137636323096
    dt= 9476589506.124744 timestep= 6603 max Sigma= 896.7054716367467 total t = 46954189237729.03 j = 93 Sigma[Inner] = 47.33871410195933
    dt= 9481693306.073618 timestep= 6613 max Sigma= 897.3842081729288 total t = 47048983210673.82 j = 93 Sigma[Inner] = 57.62235384270224
    dt= 9486788764.2512 timestep= 6623 max Sigma= 898.0621252904876 total t = 47143828175536.125 j = 93 Sigma[Inner] = 70.43228561926708
    dt= 9491876120.29759 timestep= 6633 max Sigma= 898.7392240476565 total t = 47238724050224.6 j = 93 Sigma[Inner] = 85.17220209763803
    dt= 9496955606.208765 timestep= 6643 max Sigma= 899.4155054698438 total t = 47333670754998.61 j = 93 Sigma[Inner] = 101.0034649031728
    dt= 9502027449.177305 timestep= 6653 max Sigma= 900.0909705493833 total t = 47428668212409.82 j = 93 Sigma[Inner] = 117.07334172916583
    dt= 9507091872.099842 timestep= 6663 max Sigma= 900.7656202454407 total t = 47523716347258.516 j = 93 Sigma[Inner] = 132.68270467103616
    dt= 9512149092.648966 timestep= 6673 max Sigma= 901.4394554839382 total t = 47618815086546.37 j = 93 Sigma[Inner] = 147.35524153717716
    dt= 9517199321.560898 timestep= 6683 max Sigma= 902.1124771574165 total t = 47713964359415.22 j = 93 Sigma[Inner] = 160.82778517921287
    dt= 9522242760.65886 timestep= 6693 max Sigma= 902.784686124796 total t = 47809164097066.9 j = 93 Sigma[Inner] = 173.0016713820366
    dt= 9527279601.030827 timestep= 6703 max Sigma= 903.4560832110315 total t = 47904414232663.74 j = 93 Sigma[Inner] = 183.88728783940684
    dt= 9532310021.67488 timestep= 6713 max Sigma= 904.126669206683 total t = 47999714701213.34 j = 93 Sigma[Inner] = 193.55810049955187
    dt= 9537334188.812218 timestep= 6723 max Sigma= 904.7964448674426 total t = 48095065439443.28 j = 93 Sigma[Inner] = 202.11836656628358
    dt= 2192163528.6435127 timestep= 6733 max Sigma= 905.4654109136638 total t = 48183116196945.75 j = 93 Sigma[Inner] = 209.68284332500525
    dt= 519027764.7940389 timestep= 6743 max Sigma= 905.5211336408946 total t = 48189397314547.36 j = 94 Sigma[Inner] = 145.79992316986022
    dt= 607450325.1504152 timestep= 6753 max Sigma= 905.5501406069585 total t = 48193626878704.484 j = 94 Sigma[Inner] = 123.69342657128394
    dt= 317549884.80716676 timestep= 6763 max Sigma= 905.5946521699535 total t = 48199692195030.03 j = 94 Sigma[Inner] = 75.52816237111436
    dt= 1124204238.038091 timestep= 6773 max Sigma= 905.6357118332912 total t = 48206361878851.52 j = 94 Sigma[Inner] = 87.61652451145511
    dt= 3004487137.8004937 timestep= 6783 max Sigma= 905.7804871887136 total t = 48228919749990.33 j = 93 Sigma[Inner] = 75.92069492546281
    dt= 3572317918.9780164 timestep= 6793 max Sigma= 906.0127690470971 total t = 48262679230038.67 j = 93 Sigma[Inner] = 70.7375077119058
    dt= 3865665932.2236924 timestep= 6803 max Sigma= 906.2723189346963 total t = 48300084233944.38 j = 93 Sigma[Inner] = 66.60208009395903
    dt= 4130735009.2412496 timestep= 6813 max Sigma= 906.550688288944 total t = 48340179589170.11 j = 93 Sigma[Inner] = 63.21842241514546
    dt= 5566956295.667972 timestep= 6823 max Sigma= 906.8564370859556 total t = 48385396736306.625 j = 93 Sigma[Inner] = 60.25904645534833
    dt= 9557173834.174864 timestep= 6833 max Sigma= 907.4425715316004 total t = 48473411176659.71 j = 92 Sigma[Inner] = 27.87275035790556
    dt= 9562172423.048094 timestep= 6843 max Sigma= 908.1083351720393 total t = 48569010413220.805 j = 92 Sigma[Inner] = 29.673885365826777
    dt= 9567164198.904432 timestep= 6853 max Sigma= 908.7732924286626 total t = 48664659597508.06 j = 92 Sigma[Inner] = 31.886781018459665
    dt= 9572149843.228401 timestep= 6863 max Sigma= 909.4374437333801 total t = 48760358665395.9 j = 92 Sigma[Inner] = 34.90523472204607
    dt= 9577129782.263906 timestep= 6873 max Sigma= 910.1007895142739 total t = 48856107558065.734 j = 92 Sigma[Inner] = 39.32887867348365
    dt= 9582104306.28884 timestep= 6883 max Sigma= 910.7633301821897 total t = 48951906220139.71 j = 92 Sigma[Inner] = 45.808152618026334
    dt= 9587073635.552813 timestep= 6893 max Sigma= 911.4250661240985 total t = 49047754598719.414 j = 92 Sigma[Inner] = 54.807200662614925
    dt= 9592037954.696373 timestep= 6903 max Sigma= 912.0859976999807 total t = 49143652642892.87 j = 92 Sigma[Inner] = 66.3920792376941
    dt= 9596997430.999569 timestep= 6913 max Sigma= 912.7461252415646 total t = 49239600303488.35 j = 92 Sigma[Inner] = 80.16872084936614
    dt= 9601952223.925974 timestep= 6923 max Sigma= 913.4054490520472 total t = 49335597532958.5 j = 92 Sigma[Inner] = 95.40304592150754
    dt= 9606902489.683018 timestep= 6933 max Sigma= 914.063969406314 total t = 49431644285330.37 j = 92 Sigma[Inner] = 111.24135641018903
    dt= 9611848382.747065 timestep= 6943 max Sigma= 914.7216865513808 total t = 49527740516183.26 j = 92 Sigma[Inner] = 126.91326168879826
    dt= 9616790055.483534 timestep= 6953 max Sigma= 915.3786007068803 total t = 49623886182630.49 j = 92 Sigma[Inner] = 141.84630880709435
    dt= 9621727656.625727 timestep= 6963 max Sigma= 916.0347120654912 total t = 49720081243290.63 j = 92 Sigma[Inner] = 155.68883671841922
    dt= 9626661329.205004 timestep= 6973 max Sigma= 916.6900207932479 total t = 49816325658240.04 j = 92 Sigma[Inner] = 168.27562101556322
    dt= 9631591208.41833 timestep= 6983 max Sigma= 917.3445270297109 total t = 49912619388943.93 j = 92 Sigma[Inner] = 179.57372031987026
    dt= 9636517419.818779 timestep= 6993 max Sigma= 917.9982308880093 total t = 50008962398167.75 j = 92 Sigma[Inner] = 189.63187981472163
    dt= 9641440078.10318 timestep= 7003 max Sigma= 918.6511324547813 total t = 50105354649873.47 j = 92 Sigma[Inner] = 198.5424481905251
    


    
![png](output_5_57.png)
    



    
![png](output_5_58.png)
    



    
![png](output_5_59.png)
    


    dt= 9646359286.651838 timestep= 7013 max Sigma= 919.3032317900658 total t = 50201796109107.82 j = 92 Sigma[Inner] = 206.41622456946573
    dt= 599920495.8027025 timestep= 7023 max Sigma= 919.723922908012 total t = 50255053623559.45 j = 94 Sigma[Inner] = 165.77782337788622
    dt= 282826300.03077984 timestep= 7033 max Sigma= 919.7537782299416 total t = 50259161079126.39 j = 94 Sigma[Inner] = 98.99845500832768
    dt= 1334374244.9809957 timestep= 7043 max Sigma= 919.7908108775038 total t = 50265701338031.84 j = 93 Sigma[Inner] = 82.75031951474917
    dt= 498409686.58815014 timestep= 7053 max Sigma= 919.830375895411 total t = 50270729994907.07 j = 94 Sigma[Inner] = 89.24015053907767
    dt= 1865478652.4145222 timestep= 7063 max Sigma= 919.8918248779235 total t = 50281206716515.88 j = 93 Sigma[Inner] = 79.35464836083123
    dt= 3287203778.128335 timestep= 7073 max Sigma= 920.0698899578798 total t = 50309034194774.23 j = 93 Sigma[Inner] = 74.05113064658633
    dt= 3686669030.671849 timestep= 7083 max Sigma= 920.3048710694495 total t = 50344298670747.305 j = 93 Sigma[Inner] = 69.2390450216132
    dt= 3963507947.9931774 timestep= 7093 max Sigma= 920.5617998406086 total t = 50382721985133.87 j = 93 Sigma[Inner] = 65.34608640010116
    dt= 4259750207.7518926 timestep= 7103 max Sigma= 920.836978695256 total t = 50423903355897.875 j = 93 Sigma[Inner] = 62.120603509166074
    dt= 9467813687.794687 timestep= 7113 max Sigma= 921.19906456103 total t = 50482954136313.06 j = 93 Sigma[Inner] = 51.48282721843488
    dt= 9665564438.387594 timestep= 7123 max Sigma= 921.8467284015305 total t = 50579587700833.63 j = 92 Sigma[Inner] = 28.447255923971575
    dt= 9670470605.181965 timestep= 7133 max Sigma= 922.4948864128227 total t = 50676270332780.12 j = 92 Sigma[Inner] = 30.440687796247328
    dt= 9675372708.071487 timestep= 7143 max Sigma= 923.1422421504138 total t = 50773002003498.78 j = 92 Sigma[Inner] = 32.985483506761575
    dt= 9680271258.595325 timestep= 7153 max Sigma= 923.7887954366364 total t = 50869782675388.54 j = 92 Sigma[Inner] = 36.58703106024872
    dt= 9685166568.895449 timestep= 7163 max Sigma= 924.434546092879 total t = 50966612314756.94 j = 92 Sigma[Inner] = 41.90614920268373
    dt= 9690058849.397392 timestep= 7173 max Sigma= 925.0794939300571 total t = 51063490890417.38 j = 92 Sigma[Inner] = 49.551411169013384
    dt= 9694948259.032183 timestep= 7183 max Sigma= 925.7236387440754 total t = 51160418372975.13 j = 92 Sigma[Inner] = 59.82771724075736
    dt= 9699834931.625082 timestep= 7193 max Sigma= 926.3669803138685 total t = 51257394734471.18 j = 92 Sigma[Inner] = 72.57671273274552
    dt= 9704718989.83568 timestep= 7203 max Sigma= 927.0095184007972 total t = 51354419948215.24 j = 92 Sigma[Inner] = 87.20589286869179
    dt= 9709600552.227034 timestep= 7213 max Sigma= 927.6512527487495 total t = 51451493988717.266 j = 92 Sigma[Inner] = 102.87833442690241
    dt= 9714479736.284868 timestep= 7223 max Sigma= 928.2921830845734 total t = 51548616831665.63 j = 92 Sigma[Inner] = 118.74568093951795
    dt= 9719356658.9162 timestep= 7233 max Sigma= 928.9323091186271 total t = 51645788453920.7 j = 92 Sigma[Inner] = 134.1173194738906
    dt= 9724231435.373367 timestep= 7243 max Sigma= 929.5716305453033 total t = 51743008833504.2 j = 92 Sigma[Inner] = 148.52805364129372
    dt= 9729104177.290684 timestep= 7253 max Sigma= 930.2101470434475 total t = 51840277949572.305 j = 92 Sigma[Inner] = 161.72599494948574
    dt= 9733974990.390379 timestep= 7263 max Sigma= 930.8478582766327 total t = 51937595782366.89 j = 92 Sigma[Inner] = 173.62216382063903
    dt= 9738843972.315361 timestep= 7273 max Sigma= 931.4847638932799 total t = 52034962313144.266 j = 92 Sigma[Inner] = 184.2343410891647
    dt= 9743711210.94009 timestep= 7283 max Sigma= 932.1208635266481 total t = 52132377524084.586 j = 92 Sigma[Inner] = 193.64114596598836
    dt= 9748576783.391167 timestep= 7293 max Sigma= 932.7561567947347 total t = 52229841398188.125 j = 92 Sigma[Inner] = 201.9501348159214
    dt= 9753440755.886238 timestep= 7303 max Sigma= 933.3906433001335 total t = 52327353919166.164 j = 92 Sigma[Inner] = 209.277958149632
    dt= 396890853.6007717 timestep= 7313 max Sigma= 933.5606434630602 total t = 52344152812866.57 j = 94 Sigma[Inner] = 120.95292990137271
    dt= 478275415.31702554 timestep= 7323 max Sigma= 933.5859853686111 total t = 52348134503888.625 j = 94 Sigma[Inner] = 139.51583479089635
    dt= 260562280.3001354 timestep= 7333 max Sigma= 933.6306050293712 total t = 52354784731528.61 j = 95 Sigma[Inner] = 75.37566540177025
    dt= 817519932.3110762 timestep= 7343 max Sigma= 933.661176492277 total t = 52360047814219.24 j = 94 Sigma[Inner] = 100.36430753987983
    dt= 2753516465.5729513 timestep= 7353 max Sigma= 933.7718449855698 total t = 52379023102179.78 j = 93 Sigma[Inner] = 76.8851031353475
    dt= 3519396856.984879 timestep= 7363 max Sigma= 933.9793215134142 total t = 52411747358497.52 j = 93 Sigma[Inner] = 71.54424323130382
    dt= 3840917218.1129603 timestep= 7373 max Sigma= 934.2176382920047 total t = 52448800458372.516 j = 93 Sigma[Inner] = 67.17047788967925
    dt= 4120582179.8848963 timestep= 7383 max Sigma= 934.4747207132365 total t = 52488731450316.71 j = 93 Sigma[Inner] = 63.58394607082089
    dt= 5551127427.090943 timestep= 7393 max Sigma= 934.7581244604295 total t = 52533905998488.33 j = 93 Sigma[Inner] = 60.450850608511104
    dt= 9768089791.992079 timestep= 7403 max Sigma= 935.2991717610967 total t = 52621728356223.664 j = 93 Sigma[Inner] = 26.42633680789044
    dt= 9772951159.47033 timestep= 7413 max Sigma= 935.9304136450739 total t = 52719435994158.625 j = 93 Sigma[Inner] = 27.909948030632368
    dt= 9777809827.781328 timestep= 7423 max Sigma= 936.5608467135049 total t = 52817192230371.02 j = 93 Sigma[Inner] = 29.84505470538586
    dt= 9782666384.260536 timestep= 7433 max Sigma= 937.1904704090185 total t = 52914997041279.52 j = 93 Sigma[Inner] = 32.67536212982339
    dt= 9787521179.667107 timestep= 7443 max Sigma= 937.8192841892654 total t = 53012850407846.56 j = 93 Sigma[Inner] = 37.05061958281551
    dt= 9792374430.272455 timestep= 7453 max Sigma= 938.4472875146813 total t = 53110752313727.86 j = 93 Sigma[Inner] = 43.6619872151068
    dt= 9797226284.357458 timestep= 7463 max Sigma= 939.0744798424089 total t = 53208702744329.48 j = 93 Sigma[Inner] = 52.97895516623017
    dt= 9802076856.835972 timestep= 7473 max Sigma= 939.7008606234969 total t = 53306701686336.45 j = 93 Sigma[Inner] = 65.01964682378224
    dt= 9806926247.54703 timestep= 7483 max Sigma= 940.3264293018128 total t = 53404749127489.414 j = 93 Sigma[Inner] = 79.3007251815318
    dt= 9811774550.756575 timestep= 7493 max Sigma= 940.9511853138465 total t = 53502845056491.31 j = 93 Sigma[Inner] = 94.99251643774865
    dt= 9816621859.614923 timestep= 7503 max Sigma= 941.5751280889604 total t = 53600989462979.46 j = 93 Sigma[Inner] = 111.17250090974599
    


    
![png](output_5_61.png)
    



    
![png](output_5_62.png)
    



    
![png](output_5_63.png)
    


    dt= 9821468267.527344 timestep= 7513 max Sigma= 942.1982570498184 total t = 53699182337523.86 j = 93 Sigma[Inner] = 127.04165128044666
    dt= 9826313867.575563 timestep= 7523 max Sigma= 942.8205716128366 total t = 53797423671627.86 j = 93 Sigma[Inner] = 142.03307987451706
    dt= 9831158750.767275 timestep= 7533 max Sigma= 943.4420711885493 total t = 53895713457715.99 j = 93 Sigma[Inner] = 155.82026789356792
    dt= 9836003003.725552 timestep= 7543 max Sigma= 944.0627551818397 total t = 53994051689100.75 j = 93 Sigma[Inner] = 168.26921813546684
    dt= 9840846706.326365 timestep= 7553 max Sigma= 944.682622992016 total t = 54092438359925.56 j = 93 Sigma[Inner] = 179.37598082300278
    dt= 9845689929.691118 timestep= 7563 max Sigma= 945.3016740127404 total t = 54190873465085.94 j = 93 Sigma[Inner] = 189.21257932927736
    dt= 9850532734.82427 timestep= 7573 max Sigma= 945.9199076318489 total t = 54289357000133.75 j = 93 Sigma[Inner] = 197.88844307454374
    dt= 9855375172.060236 timestep= 7583 max Sigma= 946.5373232311049 total t = 54387888961172.305 j = 93 Sigma[Inner] = 205.52611455612723
    dt= 1141014972.8830888 timestep= 7593 max Sigma= 947.049715482765 total t = 54461078538461.45 j = 94 Sigma[Inner] = 54.2534928067788
    dt= 276427415.988983 timestep= 7603 max Sigma= 947.0846902236207 total t = 54465809505762.78 j = 94 Sigma[Inner] = 108.26202931469746
    dt= 838328133.4065139 timestep= 7613 max Sigma= 947.1109780261927 total t = 54470577539201.17 j = 94 Sigma[Inner] = 101.2301343563542
    dt= 334405203.7993032 timestep= 7623 max Sigma= 947.1677779094657 total t = 54479162813196.234 j = 94 Sigma[Inner] = 73.33077476452189
    dt= 1308693622.7222862 timestep= 7633 max Sigma= 947.2084414698303 total t = 54486645121982.14 j = 94 Sigma[Inner] = 83.67907174204204
    dt= 3115755555.1037316 timestep= 7643 max Sigma= 947.3486475791067 total t = 54510897200472.68 j = 93 Sigma[Inner] = 75.33665314394644
    dt= 3634357370.006329 timestep= 7653 max Sigma= 947.5605531058167 total t = 54545356231465.7 j = 93 Sigma[Inner] = 70.160773501236
    dt= 3937664101.4224014 timestep= 7663 max Sigma= 947.7961507118441 total t = 54583419546275.99 j = 93 Sigma[Inner] = 65.98135364438433
    dt= 4251415193.5858254 timestep= 7673 max Sigma= 948.0499284430488 total t = 54624436487746.0 j = 93 Sigma[Inner] = 62.517691389464396
    dt= 8035765504.632124 timestep= 7683 max Sigma= 948.3759192154234 total t = 54680550548015.266 j = 93 Sigma[Inner] = 54.147856975079925
    dt= 9874568537.330866 timestep= 7693 max Sigma= 948.9778935123823 total t = 54779150882277.48 j = 92 Sigma[Inner] = 27.471399256873244
    dt= 9879412370.008814 timestep= 7703 max Sigma= 949.5912369703652 total t = 54877923210098.01 j = 92 Sigma[Inner] = 29.330451061033145
    dt= 9884254888.365223 timestep= 7713 max Sigma= 950.2037587247505 total t = 54976743968492.39 j = 92 Sigma[Inner] = 31.73501618632433
    dt= 9889096573.87919 timestep= 7723 max Sigma= 950.815458059016 total t = 55075613147198.7 j = 92 Sigma[Inner] = 35.21623840924264
    dt= 9893937701.229797 timestep= 7733 max Sigma= 951.4263342738527 total t = 55174530739517.445 j = 92 Sigma[Inner] = 40.47355245876643
    dt= 9898778440.470673 timestep= 7743 max Sigma= 952.0363866777651 total t = 55273496740861.85 j = 92 Sigma[Inner] = 48.150304341121554
    dt= 9903618909.456179 timestep= 7753 max Sigma= 952.6456145825053 total t = 55372511148028.16 j = 92 Sigma[Inner] = 58.55894270715061
    dt= 9908459201.295595 timestep= 7763 max Sigma= 953.2540173009925 total t = 55471573958838.98 j = 92 Sigma[Inner] = 71.51044121725276
    dt= 9913299398.804203 timestep= 7773 max Sigma= 953.8615941465355 total t = 55570685171983.01 j = 92 Sigma[Inner] = 86.35566692252404
    dt= 9918139581.798912 timestep= 7783 max Sigma= 954.4683444327263 total t = 55669844786956.65 j = 92 Sigma[Inner] = 102.20067202641711
    dt= 9922979830.182346 timestep= 7793 max Sigma= 955.0742674736491 total t = 55769052804053.63 j = 92 Sigma[Inner] = 118.16012340113187
    dt= 9927820224.402525 timestep= 7803 max Sigma= 955.679362584195 total t = 55868309224370.2 j = 92 Sigma[Inner] = 133.5322799051105
    dt= 9932660844.264896 timestep= 7813 max Sigma= 956.2836290803455 total t = 55967614049804.94 j = 92 Sigma[Inner] = 147.8607292414132
    dt= 9937501766.807404 timestep= 7823 max Sigma= 956.8870662793493 total t = 56066967283041.27 j = 92 Sigma[Inner] = 160.9124573474102
    dt= 9942343063.818003 timestep= 7833 max Sigma= 957.4896734997535 total t = 56166368927506.19 j = 92 Sigma[Inner] = 172.61941254025524
    dt= 9947184799.47576 timestep= 7843 max Sigma= 958.0914500612788 total t = 56265818987304.484 j = 92 Sigma[Inner] = 183.017839812641
    dt= 9952027028.488607 timestep= 7853 max Sigma= 958.6923952845648 total t = 56365317467131.945 j = 92 Sigma[Inner] = 192.20071664842342
    dt= 9956869794.976683 timestep= 7863 max Sigma= 959.292508490819 total t = 56464864372173.76 j = 92 Sigma[Inner] = 200.28584857144568
    dt= 9961713132.22138 timestep= 7873 max Sigma= 959.8917890014247 total t = 56564459707996.516 j = 92 Sigma[Inner] = 207.3967007386259
    dt= 694866762.4833275 timestep= 7883 max Sigma= 960.2231893042554 total t = 56610346917329.22 j = 94 Sigma[Inner] = 128.50933090674891
    dt= 328845524.1816174 timestep= 7893 max Sigma= 960.2481848228246 total t = 56614143499833.35 j = 94 Sigma[Inner] = 99.27716403297296
    dt= 1866202746.9090376 timestep= 7903 max Sigma= 960.2890112093256 total t = 56622480517847.87 j = 93 Sigma[Inner] = 76.2919043355946
    dt= 528876453.67052513 timestep= 7913 max Sigma= 960.3336107058326 total t = 56628572220562.79 j = 94 Sigma[Inner] = 81.66855674479542
    dt= 2032867274.7119782 timestep= 7923 max Sigma= 960.3943572014433 total t = 56640196579346.42 j = 93 Sigma[Inner] = 78.72142761273102
    dt= 3355366754.012893 timestep= 7933 max Sigma= 960.5602657951075 total t = 56669168878536.375 j = 93 Sigma[Inner] = 73.52308915821604
    dt= 3754978392.251796 timestep= 7943 max Sigma= 960.7732940154966 total t = 56705092024476.72 j = 93 Sigma[Inner] = 68.66126852828872
    dt= 4056760563.7538557 timestep= 7953 max Sigma= 961.0065114163192 total t = 56744311205118.64 j = 93 Sigma[Inner] = 64.66953551733167
    dt= 5052161922.256628 timestep= 7963 max Sigma= 961.2606268436822 total t = 56787743746736.25 j = 93 Sigma[Inner] = 61.27084926848651
    dt= 9976570127.809008 timestep= 7973 max Sigma= 961.725677166701 total t = 56870416623597.63 j = 92 Sigma[Inner] = 26.616033171749567
    dt= 9981419341.292244 timestep= 7983 max Sigma= 962.3215619752212 total t = 56970208995767.305 j = 92 Sigma[Inner] = 28.392924497561584
    dt= 9986267925.028236 timestep= 7993 max Sigma= 962.9166108117057 total t = 57070049856755.984 j = 92 Sigma[Inner] = 30.565385170533666
    dt= 9991116313.207825 timestep= 8003 max Sigma= 963.5108229269991 total t = 57169939202129.01 j = 92 Sigma[Inner] = 33.53631182456973
    


    
![png](output_5_65.png)
    



    
![png](output_5_66.png)
    



    
![png](output_5_67.png)
    


    dt= 9995964853.363585 timestep= 8013 max Sigma= 964.1041975995788 total t = 57269877032008.73 j = 92 Sigma[Inner] = 37.957066697616824
    dt= 10000813745.722116 timestep= 8023 max Sigma= 964.6967341241944 total t = 57369863349100.1 j = 92 Sigma[Inner] = 44.53661848401014
    dt= 10005663116.560947 timestep= 8033 max Sigma= 965.2884318054232 total t = 57469898157661.4 j = 92 Sigma[Inner] = 53.765853961947265
    dt= 10010513056.22411 timestep= 8043 max Sigma= 965.879289954565 total t = 57569981462993.77 j = 92 Sigma[Inner] = 65.67768575107154
    dt= 10015363639.146862 timestep= 8053 max Sigma= 966.4693078882733 total t = 57670113271202.64 j = 92 Sigma[Inner] = 79.79334953504558
    dt= 10020214934.27736 timestep= 8063 max Sigma= 967.0584849280949 total t = 57770293589101.66 j = 92 Sigma[Inner] = 95.28189135207778
    dt= 10025067010.036657 timestep= 8073 max Sigma= 967.6468204004623 total t = 57870522424188.586 j = 92 Sigma[Inner] = 111.22073064973394
    dt= 10029919935.954067 timestep= 8083 max Sigma= 968.2343136368801 total t = 57970799784651.16 j = 92 Sigma[Inner] = 126.81592458521857
    dt= 10034773782.194977 timestep= 8093 max Sigma= 968.8209639741415 total t = 58071125679376.93 j = 92 Sigma[Inner] = 141.51029151379086
    dt= 10039628617.79568 timestep= 8103 max Sigma= 969.4067707544716 total t = 58171500117950.71 j = 92 Sigma[Inner] = 154.98887361368912
    dt= 10044484508.242678 timestep= 8113 max Sigma= 969.9917333255466 total t = 58271923110630.266 j = 92 Sigma[Inner] = 167.12848657912977
    dt= 10049341512.930494 timestep= 8123 max Sigma= 970.5758510403608 total t = 58372394668296.75 j = 92 Sigma[Inner] = 177.93381695286254
    dt= 10054199682.93419 timestep= 8133 max Sigma= 971.1591232569524 total t = 58472914802381.14 j = 92 Sigma[Inner] = 187.48299142603182
    dt= 10059059059.417389 timestep= 8143 max Sigma= 971.7415493380145 total t = 58573483524771.57 j = 92 Sigma[Inner] = 195.88926203658573
    dt= 10063919672.867334 timestep= 8153 max Sigma= 972.3231286504381 total t = 58674100847709.46 j = 92 Sigma[Inner] = 203.27718694742077
    dt= 10068781543.221308 timestep= 8163 max Sigma= 972.9038605648345 total t = 58774766783682.51 j = 92 Sigma[Inner] = 209.76925358936688
    dt= 453510609.9058593 timestep= 8173 max Sigma= 973.0028940742698 total t = 58782337627780.09 j = 94 Sigma[Inner] = 144.14789118016577
    dt= 537006606.6178796 timestep= 8183 max Sigma= 973.0263141972688 total t = 58786486480068.73 j = 94 Sigma[Inner] = 134.33582565166674
    dt= 222622620.67077863 timestep= 8193 max Sigma= 973.0886358501141 total t = 58796991580287.77 j = 94 Sigma[Inner] = 72.90000377648259
    dt= 818506580.1001041 timestep= 8203 max Sigma= 973.1156009135417 total t = 58802269535017.92 j = 94 Sigma[Inner] = 99.9975329952914
    dt= 2791893962.594989 timestep= 8213 max Sigma= 973.2152448900413 total t = 58821548013502.08 j = 93 Sigma[Inner] = 76.75989912389275
    dt= 3564453766.368508 timestep= 8223 max Sigma= 973.4014090068622 total t = 58854666653740.39 j = 93 Sigma[Inner] = 71.28154893831824
    dt= 3907762330.4763575 timestep= 8233 max Sigma= 973.615768204972 total t = 58892279843384.71 j = 93 Sigma[Inner] = 66.75359696161902
    dt= 4247961400.7501464 timestep= 8243 max Sigma= 973.8486256199363 total t = 58933136477510.125 j = 93 Sigma[Inner] = 62.9850941162907
    dt= 9635473399.997578 timestep= 8253 max Sigma= 974.166816460829 total t = 58993938617785.94 j = 93 Sigma[Inner] = 50.533538865002555
    dt= 10084220598.615826 timestep= 8263 max Sigma= 974.742305118058 total t = 59094758928490.49 j = 93 Sigma[Inner] = 27.627725449232745
    dt= 10089090371.692171 timestep= 8273 max Sigma= 975.3194915975986 total t = 59195627918299.44 j = 93 Sigma[Inner] = 29.70058011518898
    dt= 10093960372.577404 timestep= 8283 max Sigma= 975.895827590148 total t = 59296545606614.43 j = 93 Sigma[Inner] = 32.3839528441074
    dt= 10098831035.15965 timestep= 8293 max Sigma= 976.4713124369725 total t = 59397511998319.555 j = 93 Sigma[Inner] = 36.26433649514443
    dt= 10103702599.03442 timestep= 8303 max Sigma= 977.0459455020198 total t = 59498527101460.33 j = 93 Sigma[Inner] = 42.06476310392959
    dt= 10108575206.037807 timestep= 8313 max Sigma= 977.6197261636044 total t = 59599590925885.17 j = 93 Sigma[Inner] = 50.38581007110979
    dt= 10113448949.782917 timestep= 8323 max Sigma= 978.1926538103004 total t = 59700703482566.84 j = 93 Sigma[Inner] = 61.4293848889567
    dt= 10118811465.694012 timestep= 8334 max Sigma= 978.8218882698933 total t = 59811983594742.3 j = 93 Sigma[Inner] = 76.32079943455132
    dt= 10123687818.793467 timestep= 8344 max Sigma= 979.3930226311977 total t = 59913198528262.734 j = 93 Sigma[Inner] = 91.52468356683215
    dt= 8150190816.398483 timestep= 8354 max Sigma= 979.907678628635 total t = 60002599301649.836 j = 93 Sigma[Inner] = 105.80896963955617
    dt= 8150190816.398492 timestep= 8364 max Sigma= 980.3659486089135 total t = 60084101209813.82 j = 93 Sigma[Inner] = 118.4584063693644
    dt= 8150190816.398495 timestep= 8374 max Sigma= 980.8234873800767 total t = 60165603117977.805 j = 93 Sigma[Inner] = 130.6622443365264
    dt= 8150190816.398495 timestep= 8384 max Sigma= 981.2802954531439 total t = 60247105026141.79 j = 93 Sigma[Inner] = 142.18899439538444
    dt= 8150190816.3985 timestep= 8394 max Sigma= 981.7363733402702 total t = 60328606934305.77 j = 93 Sigma[Inner] = 152.90455822295095
    dt= 8231560400.065429 timestep= 8404 max Sigma= 982.1922714168544 total t = 60410288701860.64 j = 93 Sigma[Inner] = 162.7631080612335
    dt= 8443700437.654959 timestep= 8414 max Sigma= 982.657039277765 total t = 60493823682789.69 j = 93 Sigma[Inner] = 171.92997441399953
    dt= 8601692463.544533 timestep= 8424 max Sigma= 983.1314083285278 total t = 60579166786645.82 j = 93 Sigma[Inner] = 180.3981415274261
    dt= 8721836390.412277 timestep= 8434 max Sigma= 983.6127497550879 total t = 60665870359603.09 j = 93 Sigma[Inner] = 188.1309425320038
    dt= 8815560398.640888 timestep= 8444 max Sigma= 984.0992305186404 total t = 60753622339865.66 j = 93 Sigma[Inner] = 195.1384354668803
    dt= 8890634403.655655 timestep= 8454 max Sigma= 984.5895651721058 total t = 60842203759869.6 j = 93 Sigma[Inner] = 201.45799963647383
    dt= 8952319940.57155 timestep= 8464 max Sigma= 985.0828392700153 total t = 60931458736432.78 j = 93 Sigma[Inner] = 207.14099134980194
    dt= 335691615.8985156 timestep= 8474 max Sigma= 985.3170015753315 total t = 60965238679579.24 j = 94 Sigma[Inner] = 88.15624943488197
    dt= 351489724.19812435 timestep= 8484 max Sigma= 985.3405837605895 total t = 60969526462634.77 j = 94 Sigma[Inner] = 91.84624653542195
    dt= 2322396393.011451 timestep= 8494 max Sigma= 985.384979366739 total t = 60979540703581.78 j = 93 Sigma[Inner] = 73.99051062780529
    dt= 515575333.71206576 timestep= 8504 max Sigma= 985.4621966086029 total t = 60991726481021.58 j = 94 Sigma[Inner] = 71.59780055888722
    


    
![png](output_5_69.png)
    



    
![png](output_5_70.png)
    



    
![png](output_5_71.png)
    


    dt= 1931012294.2624853 timestep= 8514 max Sigma= 985.5142184233999 total t = 61002571159873.98 j = 93 Sigma[Inner] = 78.77587525491249
    dt= 3354045760.319112 timestep= 8524 max Sigma= 985.6636494089335 total t = 61031088725647.0 j = 93 Sigma[Inner] = 73.6746137424823
    dt= 3780681747.929655 timestep= 8534 max Sigma= 985.8601024219406 total t = 61067157686862.07 j = 93 Sigma[Inner] = 68.64468032617589
    dt= 4113821817.480427 timestep= 8544 max Sigma= 986.0765281856201 total t = 61106785803028.79 j = 93 Sigma[Inner] = 64.4750701998063
    dt= 5749282358.771938 timestep= 8554 max Sigma= 986.3236342388357 total t = 61153323829463.93 j = 93 Sigma[Inner] = 60.72709556391743
    dt= 9101657348.567736 timestep= 8564 max Sigma= 986.7744407874401 total t = 61238693949161.12 j = 93 Sigma[Inner] = 28.4165595179067
    dt= 9134899914.035467 timestep= 8574 max Sigma= 987.2746906389076 total t = 61329896276830.516 j = 93 Sigma[Inner] = 29.909254897666468
    dt= 9164960872.696114 timestep= 8584 max Sigma= 987.7757732911583 total t = 61421412963137.25 j = 93 Sigma[Inner] = 31.755998197139043
    dt= 9192438659.80323 timestep= 8594 max Sigma= 988.277521723407 total t = 61513215625874.86 j = 93 Sigma[Inner] = 34.27369450571828
    dt= 9217785094.645853 timestep= 8604 max Sigma= 988.7797988788109 total t = 61605281019772.22 j = 93 Sigma[Inner] = 37.94299253178134
    dt= 9241346977.205986 timestep= 8614 max Sigma= 989.2824905178571 total t = 61697589811943.86 j = 93 Sigma[Inner] = 43.30118316476824
    dt= 9263394753.930256 timestep= 8624 max Sigma= 989.7855000671167 total t = 61790125697998.95 j = 93 Sigma[Inner] = 50.772594661995306
    dt= 9284142528.466908 timestep= 8634 max Sigma= 990.288744849763 total t = 61882874754367.73 j = 93 Sigma[Inner] = 60.50046577230915
    dt= 9303762224.773779 timestep= 8644 max Sigma= 990.7921532863598 total t = 61975824956719.44 j = 93 Sigma[Inner] = 72.26796180733994
    dt= 9322393747.160057 timestep= 8654 max Sigma= 991.295662786429 total t = 62068965816768.02 j = 93 Sigma[Inner] = 85.55031966297125
    dt= 9340152364.517391 timestep= 8664 max Sigma= 991.7992181384666 total t = 62162288104589.6 j = 93 Sigma[Inner] = 99.66257825797288
    dt= 9357134145.525457 timestep= 8674 max Sigma= 992.3027702643324 total t = 62255783633505.32 j = 93 Sigma[Inner] = 113.92452122393341
    dt= 9373420008.66963 timestep= 8684 max Sigma= 992.8062752433813 total t = 62349445091315.28 j = 93 Sigma[Inner] = 127.77838697884319
    dt= 9389078776.179188 timestep= 8694 max Sigma= 993.3096935387306 total t = 62443265906288.47 j = 93 Sigma[Inner] = 140.83781227527714
    dt= 9404169503.517195 timestep= 8704 max Sigma= 993.8129893767887 total t = 62537240139521.12 j = 93 Sigma[Inner] = 152.88182014565973
    dt= 9418743276.16473 timestep= 8714 max Sigma= 994.3161302443064 total t = 62631362397527.55 j = 93 Sigma[Inner] = 163.82030157439883
    dt= 9432844610.512596 timestep= 8724 max Sigma= 994.8190864765379 total t = 62725627760526.39 j = 93 Sigma[Inner] = 173.65300799999056
    dt= 9446512557.499908 timestep= 8734 max Sigma= 995.3218309167629 total t = 62820031723031.91 j = 93 Sigma[Inner] = 182.43410014114127
    dt= 9459781580.83362 timestep= 8744 max Sigma= 995.8243386322639 total t = 62914570144191.49 j = 93 Sigma[Inner] = 190.24613859045454
    dt= 9472682262.610197 timestep= 8754 max Sigma= 996.3265866753898 total t = 63009239205918.695 j = 93 Sigma[Inner] = 197.18297553465007
    dt= 9485241875.542887 timestep= 8764 max Sigma= 996.8285538809464 total t = 63104035377321.1 j = 93 Sigma[Inner] = 203.33944569505093
    dt= 9497484851.1521 timestep= 8774 max Sigma= 997.3302206931093 total t = 63198955384257.305 j = 93 Sigma[Inner] = 208.80568538127295
    dt= 459252963.3163844 timestep= 8784 max Sigma= 997.4206454584997 total t = 63207044068945.82 j = 94 Sigma[Inner] = 143.0748773574072
    dt= 528423894.70086807 timestep= 8794 max Sigma= 997.445131391881 total t = 63211752264189.73 j = 94 Sigma[Inner] = 138.48143668338938
    dt= 3524652556.332847 timestep= 8804 max Sigma= 997.5248420565049 total t = 63229852977497.62 j = 93 Sigma[Inner] = 72.38906677439499
    dt= 688674203.8308394 timestep= 8814 max Sigma= 997.581722551393 total t = 63237798105990.31 j = 94 Sigma[Inner] = 74.71486533648019
    dt= 2461427476.3927293 timestep= 8824 max Sigma= 997.6525449962764 total t = 63252997933963.55 j = 93 Sigma[Inner] = 77.72858356810806
    dt= 3503720776.7316003 timestep= 8834 max Sigma= 997.8136453368586 total t = 63284595933905.49 j = 93 Sigma[Inner] = 72.19429799965151
    dt= 3892569074.0654664 timestep= 8844 max Sigma= 998.0080540696732 total t = 63321882845827.95 j = 93 Sigma[Inner] = 67.35245867475356
    dt= 4274984678.1231728 timestep= 8854 max Sigma= 998.2213707947208 total t = 63362783372596.62 j = 93 Sigma[Inner] = 63.28542274575537
    dt= 9525287018.648409 timestep= 8864 max Sigma= 998.5135248849126 total t = 63423579058323.56 j = 93 Sigma[Inner] = 50.57479732025554
    dt= 9536550643.350647 timestep= 8874 max Sigma= 999.0140550868937 total t = 63518893771707.79 j = 93 Sigma[Inner] = 28.31395650050733
    dt= 9547650351.61043 timestep= 8884 max Sigma= 999.5142150518359 total t = 63614320516459.87 j = 93 Sigma[Inner] = 30.12229428526526
    dt= 9558527401.969372 timestep= 8894 max Sigma= 1000.013992116018 total t = 63709857021417.79 j = 93 Sigma[Inner] = 32.44587862502013
    dt= 9569195814.18495 timestep= 8904 max Sigma= 1000.5133740831932 total t = 63805501138449.016 j = 93 Sigma[Inner] = 35.73347740852736
    dt= 9579668165.818108 timestep= 8914 max Sigma= 1001.0123496566594 total t = 63901250851407.42 j = 93 Sigma[Inner] = 40.560373468916524
    dt= 9589955867.361204 timestep= 8924 max Sigma= 1001.5109083655924 total t = 63997104263321.2 j = 93 Sigma[Inner] = 47.453799326316144
    dt= 9600069347.941624 timestep= 8934 max Sigma= 1002.0090405036201 total t = 64093059585798.02 j = 93 Sigma[Inner] = 56.686868060312364
    dt= 9610018189.713367 timestep= 8944 max Sigma= 1002.5067370758425 total t = 64189115129982.54 j = 93 Sigma[Inner] = 68.14539687962692
    dt= 9619811231.22718 timestep= 8954 max Sigma= 1003.0039897520633 total t = 64285269298675.39 j = 93 Sigma[Inner] = 81.34349168813453
    dt= 9629456650.364622 timestep= 8964 max Sigma= 1003.5007908248358 total t = 64381520579367.28 j = 93 Sigma[Inner] = 95.57080088742535
    dt= 9638962032.95251 timestep= 8974 max Sigma= 1003.997133171371 total t = 64477867538019.59 j = 93 Sigma[Inner] = 110.08372240518464
    dt= 9648334430.879896 timestep= 8984 max Sigma= 1004.493010218633 total t = 64574308813471.0 j = 93 Sigma[Inner] = 124.2542460699963
    dt= 9657580412.129086 timestep= 8994 max Sigma= 1004.9884159111158 total t = 64670843112379.17 j = 93 Sigma[Inner] = 137.6394495069205
    dt= 9666706104.31417 timestep= 9004 max Sigma= 1005.4833446809135 total t = 64767469204626.016 j = 93 Sigma[Inner] = 149.9824151339925
    


    
![png](output_5_73.png)
    



    
![png](output_5_74.png)
    



    
![png](output_5_75.png)
    


    dt= 9675717232.857635 timestep= 9014 max Sigma= 1005.9777914197725 total t = 64864185919128.0 j = 93 Sigma[Inner] = 161.17555539167384
    dt= 9684619154.667776 timestep= 9024 max Sigma= 1006.4717514528604 total t = 64960992140002.88 j = 93 Sigma[Inner] = 171.21400643243877
    dt= 9693416888.014225 timestep= 9034 max Sigma= 1006.965220514054 total t = 65057886803051.76 j = 93 Sigma[Inner] = 180.15471299884808
    dt= 9702115139.18515 timestep= 9044 max Sigma= 1007.4581947225626 total t = 65154868892521.7 j = 93 Sigma[Inner] = 188.0864409664041
    dt= 9710718326.42126 timestep= 9054 max Sigma= 1007.9506705607491 total t = 65251937438119.68 j = 93 Sigma[Inner] = 195.1102383863282
    dt= 9719230601.545732 timestep= 9064 max Sigma= 1008.442644853027 total t = 65349091512252.78 j = 93 Sigma[Inner] = 201.32784402277827
    dt= 9727655869.640154 timestep= 9074 max Sigma= 1008.9341147457478 total t = 65446330227473.734 j = 93 Sigma[Inner] = 206.83544249258546
    dt= 523223132.54171693 timestep= 9084 max Sigma= 1009.2992195586368 total t = 65509476301071.93 j = 94 Sigma[Inner] = 133.00385193927855
    dt= 307753105.2533616 timestep= 9094 max Sigma= 1009.3265706777009 total t = 65514685386138.82 j = 94 Sigma[Inner] = 86.20037447465619
    dt= 1047803573.7271333 timestep= 9104 max Sigma= 1009.3506084100429 total t = 65520193321181.32 j = 94 Sigma[Inner] = 90.80447821439195
    dt= 268609682.0783281 timestep= 9114 max Sigma= 1009.4749317951353 total t = 65544080135201.93 j = 94 Sigma[Inner] = 69.73231883314917
    dt= 979952941.3504313 timestep= 9124 max Sigma= 1009.5026571167164 total t = 65550293973565.69 j = 94 Sigma[Inner] = 117.9988796175944
    dt= 3027361979.9527426 timestep= 9134 max Sigma= 1009.6036988761438 total t = 65572399269309.055 j = 93 Sigma[Inner] = 75.8454357034038
    dt= 3673736892.8376007 timestep= 9144 max Sigma= 1009.7740030958655 total t = 65606870160261.55 j = 93 Sigma[Inner] = 70.29185635044959
    dt= 4040590825.917573 timestep= 9154 max Sigma= 1009.9673163937869 total t = 65645658135437.336 j = 93 Sigma[Inner] = 65.6936449811271
    dt= 5434059237.484979 timestep= 9164 max Sigma= 1010.184133507439 total t = 65690177945492.67 j = 93 Sigma[Inner] = 61.68262924927261
    dt= 9755523597.08287 timestep= 9174 max Sigma= 1010.5920634910877 total t = 65775733664561.086 j = 92 Sigma[Inner] = 27.25749274377545
    dt= 9763613638.43896 timestep= 9184 max Sigma= 1011.0812963369473 total t = 65873333456086.805 j = 92 Sigma[Inner] = 29.04092028642183
    dt= 9771632234.256721 timestep= 9194 max Sigma= 1011.5700144072154 total t = 65971013752203.7 j = 92 Sigma[Inner] = 31.213648759231646
    dt= 9779582834.517014 timestep= 9204 max Sigma= 1012.0582164037479 total t = 66068773857638.375 j = 92 Sigma[Inner] = 34.149913372034796
    dt= 9787468459.809893 timestep= 9214 max Sigma= 1012.5459012586807 total t = 66166613109375.66 j = 92 Sigma[Inner] = 38.41482348464938
    dt= 9795291803.01512 timestep= 9224 max Sigma= 1013.03306810683 total t = 66264530872698.61 j = 92 Sigma[Inner] = 44.60465139339195
    dt= 9803055326.456736 timestep= 9234 max Sigma= 1013.5197162646361 total t = 66362526538491.99 j = 92 Sigma[Inner] = 53.12438765327164
    dt= 9810761317.579544 timestep= 9244 max Sigma= 1014.0058452130861 total t = 66460599521267.1 j = 92 Sigma[Inner] = 64.00209434235622
    dt= 9818411922.667116 timestep= 9254 max Sigma= 1014.4914545830179 total t = 66558749257611.49 j = 92 Sigma[Inner] = 76.84726395055061
    dt= 9826009168.555586 timestep= 9264 max Sigma= 1014.9765441419643 total t = 66656975204904.04 j = 92 Sigma[Inner] = 90.97158745406176
    dt= 9833554977.51244 timestep= 9274 max Sigma= 1015.4611137820615 total t = 66755276840206.58 j = 92 Sigma[Inner] = 105.5920051125078
    dt= 9841051178.01722 timestep= 9284 max Sigma= 1015.9451635087719 total t = 66853653659280.09 j = 92 Sigma[Inner] = 120.01131090188464
    dt= 9848499512.928213 timestep= 9294 max Sigma= 1016.4286934302446 total t = 66952105175693.586 j = 92 Sigma[Inner] = 133.717066578441
    dt= 9855901645.87129 timestep= 9304 max Sigma= 1016.9117037472254 total t = 67050630920004.13 j = 92 Sigma[Inner] = 146.39872876954755
    dt= 9863259166.35374 timestep= 9314 max Sigma= 1017.394194743441 total t = 67149230438993.65 j = 92 Sigma[Inner] = 157.9147240020264
    dt= 9870573593.936747 timestep= 9324 max Sigma= 1017.8761667764086 total t = 67247903294951.305 j = 92 Sigma[Inner] = 168.24265212703307
    dt= 9877846381.715244 timestep= 9334 max Sigma= 1018.3576202686419 total t = 67346649064993.67 j = 92 Sigma[Inner] = 177.43314531664404
    dt= 9885078919.306992 timestep= 9344 max Sigma= 1018.83855569922 total t = 67445467340416.82 j = 92 Sigma[Inner] = 185.57522643405218
    dt= 9892272535.521423 timestep= 9354 max Sigma= 1019.3189735957153 total t = 67544357726076.48 j = 92 Sigma[Inner] = 192.77348233652606
    dt= 9899428500.850294 timestep= 9364 max Sigma= 1019.7988745264588 total t = 67643319839793.68 j = 92 Sigma[Inner] = 199.13445457801095
    dt= 9906548029.892479 timestep= 9374 max Sigma= 1020.2782590931487 total t = 67742353311784.96 j = 92 Sigma[Inner] = 204.7592871899126
    dt= 1316434052.0627406 timestep= 9384 max Sigma= 1020.7198962140264 total t = 67825149623008.57 j = 93 Sigma[Inner] = 109.05263330741907
    dt= 664840065.1137058 timestep= 9394 max Sigma= 1020.7511021406608 total t = 67830961594025.34 j = 94 Sigma[Inner] = 124.86388216617694
    dt= 693125511.4041417 timestep= 9404 max Sigma= 1020.7744488749507 total t = 67835826104301.71 j = 94 Sigma[Inner] = 113.74458926266176
    dt= 4034874859.1660576 timestep= 9414 max Sigma= 1020.8799872485447 total t = 67861034728260.516 j = 93 Sigma[Inner] = 70.74647139727708
    dt= 4826820816.293563 timestep= 9424 max Sigma= 1021.095322888714 total t = 67906468898228.25 j = 93 Sigma[Inner] = 62.01257454900052
    dt= 5223914336.570067 timestep= 9434 max Sigma= 1021.3370538223548 total t = 67957022891914.734 j = 93 Sigma[Inner] = 56.29459745732525
    dt= 5576929198.045201 timestep= 9444 max Sigma= 1021.5961295096472 total t = 68011181769460.28 j = 93 Sigma[Inner] = 52.39908237109034
    dt= 8568229554.9371195 timestep= 9454 max Sigma= 1021.88503505233 total t = 68074235101009.89 j = 93 Sigma[Inner] = 49.30731469284067
    dt= 9937080326.086775 timestep= 9464 max Sigma= 1022.3556346784524 total t = 68173574578856.89 j = 93 Sigma[Inner] = 26.4380415026141
    dt= 9944026829.087057 timestep= 9474 max Sigma= 1022.8322694389811 total t = 68272983614167.77 j = 93 Sigma[Inner] = 28.478223186259342
    dt= 9950942270.510855 timestep= 9484 max Sigma= 1023.308392129314 total t = 68372461942399.52 j = 93 Sigma[Inner] = 32.351588675669724
    dt= 9957828020.57302 timestep= 9494 max Sigma= 1023.7840033562503 total t = 68472009260733.51 j = 93 Sigma[Inner] = 39.40521189349467
    dt= 9964685204.302465 timestep= 9504 max Sigma= 1024.2591037302445 total t = 68571625278596.22 j = 93 Sigma[Inner] = 50.482812224654566
    


    
![png](output_5_77.png)
    



    
![png](output_5_78.png)
    



    
![png](output_5_79.png)
    


    dt= 9971514800.013918 timestep= 9514 max Sigma= 1024.7336938533083 total t = 68671309715799.38 j = 93 Sigma[Inner] = 65.38675507151582
    dt= 9978317692.557491 timestep= 9524 max Sigma= 1025.2077743107939 total t = 68771062301391.836 j = 93 Sigma[Inner] = 82.9516528699149
    dt= 9985094703.051582 timestep= 9534 max Sigma= 1025.6813456653856 total t = 68870882772902.19 j = 93 Sigma[Inner] = 101.5946431529057
    dt= 9991846606.132555 timestep= 9544 max Sigma= 1026.1544084524246 total t = 68970770875803.73 j = 93 Sigma[Inner] = 119.89031615399833
    dt= 9998574140.386774 timestep= 9554 max Sigma= 1026.6269631760833 total t = 69070726363111.14 j = 93 Sigma[Inner] = 136.87345489173174
    dt= 10005278014.928013 timestep= 9564 max Sigma= 1027.09901030614 total t = 69170748995058.56 j = 93 Sigma[Inner] = 152.06125663329658
    dt= 10011958913.70052 timestep= 9574 max Sigma= 1027.5705502751891 total t = 69270838538829.97 j = 93 Sigma[Inner] = 165.32999486895218
    dt= 10018617498.369757 timestep= 9584 max Sigma= 1028.0415834761957 total t = 69370994768324.27 j = 93 Sigma[Inner] = 176.76832102786534
    dt= 10025254410.286688 timestep= 9594 max Sigma= 1028.5121102603302 total t = 69471217463944.016 j = 93 Sigma[Inner] = 186.5645466931441
    dt= 10031870271.81318 timestep= 9604 max Sigma= 1028.9821309350452 total t = 69571506412399.91 j = 93 Sigma[Inner] = 194.93723088737434
    dt= 10038465687.193262 timestep= 9614 max Sigma= 1029.4516457623533 total t = 69671861406526.055 j = 93 Sigma[Inner] = 202.09892993660392
    dt= 10045041243.101143 timestep= 9624 max Sigma= 1029.9206549572934 total t = 69772282245102.06 j = 93 Sigma[Inner] = 208.2404238264772
    dt= 408028483.2857023 timestep= 9634 max Sigma= 1030.0480423769416 total t = 69789949321428.09 j = 94 Sigma[Inner] = 118.33296321517346
    dt= 483113577.9226328 timestep= 9644 max Sigma= 1030.0693189079573 total t = 69794586332649.45 j = 94 Sigma[Inner] = 129.2013477302547
    dt= 3190447639.199683 timestep= 9654 max Sigma= 1030.1272374190987 total t = 69809713731594.4 j = 93 Sigma[Inner] = 72.67797011400589
    dt= 642888370.2012405 timestep= 9664 max Sigma= 1030.1753110096713 total t = 69817477049481.24 j = 94 Sigma[Inner] = 75.33413075890233
    dt= 2390103954.578297 timestep= 9674 max Sigma= 1030.2347781664655 total t = 69831981479804.0 j = 93 Sigma[Inner] = 77.91167068512337
    dt= 3493508888.7976017 timestep= 9684 max Sigma= 1030.3752376926495 total t = 69863227745794.57 j = 93 Sigma[Inner] = 72.37417207256439
    dt= 3897464349.023657 timestep= 9694 max Sigma= 1030.5469417201534 total t = 69900500746617.41 j = 93 Sigma[Inner] = 67.44427168796946
    dt= 4310547492.866396 timestep= 9704 max Sigma= 1030.7361422555152 total t = 69941566776800.71 j = 93 Sigma[Inner] = 63.2783327458348
    dt= 10060378113.889307 timestep= 9714 max Sigma= 1031.0175728353006 total t = 70007838112795.97 j = 92 Sigma[Inner] = 42.001269975219714
    dt= 10066875185.971064 timestep= 9724 max Sigma= 1031.4848921680282 total t = 70108477556144.73 j = 92 Sigma[Inner] = 27.957116850623922
    dt= 10073373999.142094 timestep= 9734 max Sigma= 1031.9517062825191 total t = 70209182066943.586 j = 92 Sigma[Inner] = 30.0996684756888
    dt= 10079854644.059265 timestep= 9744 max Sigma= 1032.41801518097 total t = 70309951465034.8 j = 92 Sigma[Inner] = 32.8736172118998
    dt= 10086318072.631351 timestep= 9754 max Sigma= 1032.8838187111658 total t = 70410785574209.94 j = 92 Sigma[Inner] = 36.843243178158644
    dt= 10092765015.91988 timestep= 9764 max Sigma= 1033.34911668429 total t = 70511684226461.08 j = 92 Sigma[Inner] = 42.66984929117396
    dt= 10099196077.728096 timestep= 9774 max Sigma= 1033.8139088670787 total t = 70612647260336.14 j = 92 Sigma[Inner] = 50.86843985972916
    dt= 10105611784.466787 timestep= 9784 max Sigma= 1034.278194977428 total t = 70713674519965.78 j = 92 Sigma[Inner] = 61.571266254890034
    dt= 10112012612.4639 timestep= 9794 max Sigma= 1034.741974681905 total t = 70814765854452.77 j = 92 Sigma[Inner] = 74.43744537734037
    dt= 10118399003.393787 timestep= 9804 max Sigma= 1035.2052475943383 total t = 70915921117463.1 j = 92 Sigma[Inner] = 88.7600564406544
    dt= 10124771373.283472 timestep= 9814 max Sigma= 1035.6680132750632 total t = 71017140166932.73 j = 92 Sigma[Inner] = 103.69197492376053
    dt= 10131130117.921888 timestep= 9824 max Sigma= 1036.1302712305796 total t = 71118422864843.27 j = 92 Sigma[Inner] = 118.46184861469125
    dt= 10137475616.154871 timestep= 9834 max Sigma= 1036.592020913482 total t = 71219769077038.81 j = 92 Sigma[Inner] = 132.49850683898046
    dt= 10143808231.861542 timestep= 9844 max Sigma= 1037.053261722577 total t = 71321178673068.4 j = 92 Sigma[Inner] = 145.4568713121022
    dt= 10150128315.061926 timestep= 9854 max Sigma= 1037.5139930031373 total t = 71422651526043.06 j = 92 Sigma[Inner] = 157.18206089545774
    dt= 10156436202.437428 timestep= 9864 max Sigma= 1037.9742140472458 total t = 71524187512500.34 j = 92 Sigma[Inner] = 167.65233599120145
    dt= 10162732217.468195 timestep= 9874 max Sigma= 1038.4339240942115 total t = 71625786512272.1 j = 92 Sigma[Inner] = 176.92619140371735
    dt= 10169016670.354183 timestep= 9884 max Sigma= 1038.8931223310328 total t = 71727448408351.95 j = 92 Sigma[Inner] = 185.10309093153282
    dt= 10175289857.864109 timestep= 9894 max Sigma= 1039.3518078929 total t = 71829173086761.84 j = 92 Sigma[Inner] = 192.29805759316352
    dt= 10181552063.234732 timestep= 9904 max Sigma= 1039.8099798637343 total t = 71930960436416.98 j = 92 Sigma[Inner] = 198.62688174632547
    dt= 10187803556.216845 timestep= 9914 max Sigma= 1040.26763727676 total t = 72032810348990.64 j = 92 Sigma[Inner] = 204.19838845593318
    dt= 10194044593.334547 timestep= 9924 max Sigma= 1040.7247791151203 total t = 72134722718780.61 j = 92 Sigma[Inner] = 209.11099289745263
    dt= 412079382.8881794 timestep= 9934 max Sigma= 1040.846648551003 total t = 72152138427673.12 j = 94 Sigma[Inner] = 117.4198180005329
    dt= 492355359.07856923 timestep= 9944 max Sigma= 1040.8674027646248 total t = 72156851987312.52 j = 94 Sigma[Inner] = 128.72772054591204
    dt= 3268301263.1925764 timestep= 9954 max Sigma= 1040.9252599414856 total t = 72172546083005.11 j = 93 Sigma[Inner] = 72.59223900632585
    dt= 4683975911.092549 timestep= 9964 max Sigma= 1041.1102039251039 total t = 72215273720500.97 j = 93 Sigma[Inner] = 64.15813128250755
    dt= 5140351507.884238 timestep= 9974 max Sigma= 1041.3296846921305 total t = 72264794333738.33 j = 93 Sigma[Inner] = 57.59437327199301
    dt= 5507038467.799962 timestep= 9984 max Sigma= 1041.5668049159258 total t = 72318214657970.36 j = 93 Sigma[Inner] = 53.20520095689158
    dt= 7530354401.754094 timestep= 9994 max Sigma= 1041.824460972635 total t = 72377940774085.62 j = 93 Sigma[Inner] = 49.926622472609296
    dt= 10214977425.0329 timestep= 10004 max Sigma= 1042.2604068029318 total t = 72478383361776.4 j = 92 Sigma[Inner] = 25.953342270130676
    


    
![png](output_5_81.png)
    



    
![png](output_5_82.png)
    



    
![png](output_5_83.png)
    


    dt= 10221178520.603468 timestep= 10014 max Sigma= 1042.7152832897902 total t = 72580567250995.62 j = 92 Sigma[Inner] = 28.041463750828864
    dt= 10227369205.486982 timestep= 10024 max Sigma= 1043.1696380254973 total t = 72682813093242.3 j = 92 Sigma[Inner] = 31.98037557590373
    dt= 10233550142.133766 timestep= 10034 max Sigma= 1043.6234696518864 total t = 72785120788281.61 j = 92 Sigma[Inner] = 39.20936634340334
    dt= 10239721783.022902 timestep= 10044 max Sigma= 1044.0767767836355 total t = 72887490241245.08 j = 92 Sigma[Inner] = 50.62278373112517
    dt= 10245884465.334265 timestep= 10054 max Sigma= 1044.5295580023055 total t = 72989921361099.19 j = 92 Sigma[Inner] = 65.97208446680014
    dt= 10252038461.738285 timestep= 10064 max Sigma= 1044.9818118536703 total t = 73092414059797.16 j = 92 Sigma[Inner] = 83.97041386673071
    dt= 10258184008.200256 timestep= 10074 max Sigma= 1045.4335368468112 total t = 73194968251800.28 j = 92 Sigma[Inner] = 102.92282957410492
    dt= 10264321319.642824 timestep= 10084 max Sigma= 1045.8847314541918 total t = 73297583853804.08 j = 92 Sigma[Inner] = 121.3525406938248
    dt= 10270450599.012802 timestep= 10094 max Sigma= 1046.335394112287 total t = 73400260784581.81 j = 92 Sigma[Inner] = 138.30276345741493
    dt= 10276572042.64054 timestep= 10104 max Sigma= 1046.7855232225306 total t = 73502998964896.47 j = 92 Sigma[Inner] = 153.33115737808959
    dt= 10282685843.424843 timestep= 10114 max Sigma= 1047.2351171524451 total t = 73605798317453.47 j = 92 Sigma[Inner] = 166.3604055765802
    dt= 10288792192.67563 timestep= 10124 max Sigma= 1047.6841742368815 total t = 73708658766877.39 j = 92 Sigma[Inner] = 177.51777112980918
    dt= 10294891281.081068 timestep= 10134 max Sigma= 1048.1326927793123 total t = 73811580239701.98 j = 92 Sigma[Inner] = 187.01876447115575
    dt= 10300983299.075771 timestep= 10144 max Sigma= 1048.5806710531426 total t = 73914562664366.38 j = 92 Sigma[Inner] = 195.09907100080338
    dt= 10307068436.78873 timestep= 10154 max Sigma= 1049.0281073030174 total t = 74017605971212.5 j = 92 Sigma[Inner] = 201.98101478211885
    dt= 10313146883.700039 timestep= 10164 max Sigma= 1049.4749997461097 total t = 74120710092480.42 j = 92 Sigma[Inner] = 207.86028104319593
    dt= 322209522.2598999 timestep= 10174 max Sigma= 1049.6388384494055 total t = 74148564055747.92 j = 94 Sigma[Inner] = 94.7361831783142
    dt= 409025700.7330447 timestep= 10184 max Sigma= 1049.6583912815563 total t = 74153169361406.6 j = 94 Sigma[Inner] = 103.89525304091022
    dt= 2846438591.777969 timestep= 10194 max Sigma= 1049.7028810906113 total t = 74165889239137.67 j = 93 Sigma[Inner] = 72.86890197209247
    dt= 4605489873.295248 timestep= 10204 max Sigma= 1049.8730965165148 total t = 74207004675449.2 j = 93 Sigma[Inner] = 65.10627715592982
    dt= 5082722256.189889 timestep= 10214 max Sigma= 1050.0823087478134 total t = 74255891945776.1 j = 93 Sigma[Inner] = 58.29265234116436
    dt= 5442985314.043136 timestep= 10224 max Sigma= 1050.3088506238514 total t = 74308718576350.31 j = 93 Sigma[Inner] = 53.76219105744903
    dt= 5921761665.316149 timestep= 10234 max Sigma= 1050.5514383448337 total t = 74365433546374.73 j = 93 Sigma[Inner] = 50.46504932336659
    dt= 10333090585.90468 timestep= 10244 max Sigma= 1050.941496410048 total t = 74460381538490.88 j = 92 Sigma[Inner] = 25.443241762577134
    dt= 10339145729.346287 timestep= 10254 max Sigma= 1051.3860344983937 total t = 74563745753737.88 j = 92 Sigma[Inner] = 27.37074017682385
    dt= 10345193898.29994 timestep= 10264 max Sigma= 1051.8300190547172 total t = 74667170481388.47 j = 92 Sigma[Inner] = 30.935140477097637
    dt= 10351235779.663713 timestep= 10274 max Sigma= 1052.2734481071893 total t = 74770655655690.27 j = 92 Sigma[Inner] = 37.63530847335251
    dt= 10357271835.044128 timestep= 10284 max Sigma= 1052.7163196824836 total t = 74874201216446.67 j = 92 Sigma[Inner] = 48.54901489538469
    dt= 10363302396.471956 timestep= 10294 max Sigma= 1053.1586317990698 total t = 74977807107302.58 j = 92 Sigma[Inner] = 63.60953802777178
    dt= 10369327722.374092 timestep= 10304 max Sigma= 1053.6003824639615 total t = 75081473274785.38 j = 92 Sigma[Inner] = 81.59662294518918
    dt= 10375348027.712793 timestep= 10314 max Sigma= 1054.041569671314 total t = 75185199667749.64 j = 92 Sigma[Inner] = 100.76110403640803
    dt= 10381363500.668657 timestep= 10324 max Sigma= 1054.4821914020233 total t = 75288986237042.14 j = 92 Sigma[Inner] = 119.51906510793393
    dt= 10387374312.162615 timestep= 10334 max Sigma= 1054.9222456238772 total t = 75392832935290.6 j = 92 Sigma[Inner] = 136.81783430946228
    dt= 10393380621.468283 timestep= 10344 max Sigma= 1055.3617302920106 total t = 75496739716764.34 j = 92 Sigma[Inner] = 152.15651587144552
    dt= 10399382579.617323 timestep= 10354 max Sigma= 1055.8006433495448 total t = 75600706537277.97 j = 92 Sigma[Inner] = 165.43358855002964
    dt= 10405380331.50178 timestep= 10364 max Sigma= 1056.2389827283182 total t = 75704733354121.23 j = 92 Sigma[Inner] = 176.77363518484285
    dt= 10411374017.162357 timestep= 10374 max Sigma= 1056.6767463496803 total t = 75808820126005.38 j = 92 Sigma[Inner] = 186.39965034247774
    dt= 10417363772.536882 timestep= 10384 max Sigma= 1057.1139321253002 total t = 75912966813018.97 j = 92 Sigma[Inner] = 194.55799986406848
    dt= 10423349729.834768 timestep= 10394 max Sigma= 1057.550537957988 total t = 76017173376589.3 j = 92 Sigma[Inner] = 201.48148838977264
    dt= 10429332017.650335 timestep= 10404 max Sigma= 1057.9865617424998 total t = 76121439779445.86 j = 92 Sigma[Inner] = 207.3747862556071
    dt= 705141111.5986341 timestep= 10414 max Sigma= 1058.2270559566603 total t = 76169307596939.5 j = 94 Sigma[Inner] = 126.69048720172633
    dt= 328102163.2047342 timestep= 10424 max Sigma= 1058.2474179896888 total t = 76173809929995.84 j = 94 Sigma[Inner] = 81.69391262609403
    dt= 1864500500.1238654 timestep= 10434 max Sigma= 1058.2754813351985 total t = 76182071873315.22 j = 93 Sigma[Inner] = 76.08666793654459
    dt= 4468398329.142911 timestep= 10444 max Sigma= 1058.4201282168706 total t = 76219353542200.08 j = 93 Sigma[Inner] = 66.81529311559188
    dt= 5014420489.528628 timestep= 10454 max Sigma= 1058.6178740051646 total t = 76267342676897.83 j = 93 Sigma[Inner] = 59.3598668132129
    dt= 5388260289.672325 timestep= 10464 max Sigma= 1058.8338478277058 total t = 76319580110010.03 j = 93 Sigma[Inner] = 54.43876127431178
    dt= 5820340323.625629 timestep= 10474 max Sigma= 1059.0653229248474 total t = 76375653187690.05 j = 93 Sigma[Inner] = 50.92546448375787
    dt= 10449085359.633718 timestep= 10484 max Sigma= 1059.424687220652 total t = 76466775082954.95 j = 92 Sigma[Inner] = 25.160741445284362
    dt= 10455056590.997486 timestep= 10494 max Sigma= 1059.858178901286 total t = 76571298782106.77 j = 92 Sigma[Inner] = 27.057127904619218
    dt= 10461023496.31088 timestep= 10504 max Sigma= 1060.291077370719 total t = 76675882169282.86 j = 92 Sigma[Inner] = 30.505159942280617
    


    
![png](output_5_85.png)
    



    
![png](output_5_86.png)
    



    
![png](output_5_87.png)
    


    dt= 10466986685.054163 timestep= 10514 max Sigma= 1060.7233804466089 total t = 76780525204655.55 j = 92 Sigma[Inner] = 37.03323548454603
    dt= 10472946564.045357 timestep= 10524 max Sigma= 1061.1550859615486 total t = 76885227853444.42 j = 92 Sigma[Inner] = 47.796511030075756
    dt= 10478903404.26754 timestep= 10534 max Sigma= 1061.5861917560544 total t = 76989990084124.25 j = 92 Sigma[Inner] = 62.790872941908546
    dt= 10484857400.409733 timestep= 10544 max Sigma= 1062.016695674895 total t = 77094811867424.95 j = 92 Sigma[Inner] = 80.8024651436912
    dt= 10490808702.927189 timestep= 10554 max Sigma= 1062.4465955653252 total t = 77199693175760.89 j = 92 Sigma[Inner] = 100.04393003648708
    dt= 10496757435.80982 timestep= 10564 max Sigma= 1062.8758892763478 total t = 77304633982894.62 j = 92 Sigma[Inner] = 118.88480528499244
    dt= 10502703706.752363 timestep= 10574 max Sigma= 1063.3045746585349 total t = 77409634263732.6 j = 92 Sigma[Inner] = 136.24180283285165
    dt= 10508647613.176352 timestep= 10584 max Sigma= 1063.732649564178 total t = 77514693994197.92 j = 92 Sigma[Inner] = 151.603191007855
    dt= 10514589245.907087 timestep= 10594 max Sigma= 1064.1601118476153 total t = 77619813151149.3 j = 92 Sigma[Inner] = 164.8696312898279
    dt= 10520528691.461283 timestep= 10604 max Sigma= 1064.5869593656694 total t = 77724991712328.48 j = 92 Sigma[Inner] = 176.17331073813386
    dt= 10526466033.4613 timestep= 10614 max Sigma= 1065.013189978143 total t = 77830229656325.62 j = 92 Sigma[Inner] = 185.7455206876471
    dt= 10532401353.464794 timestep= 10624 max Sigma= 1065.4388015483412 total t = 77935526962555.22 j = 92 Sigma[Inner] = 193.83959299401
    dt= 10538334731.383875 timestep= 10634 max Sigma= 1065.8637919436044 total t = 78040883611238.08 j = 92 Sigma[Inner] = 200.69349850172452
    dt= 10544266245.612282 timestep= 10644 max Sigma= 1066.2881590358338 total t = 78146299583385.78 j = 92 Sigma[Inner] = 206.51546015826284
    dt= 526060915.73967904 timestep= 10654 max Sigma= 1066.6020886558983 total t = 78214398199917.23 j = 94 Sigma[Inner] = 132.61056993551435
    dt= 314590782.55093724 timestep= 10664 max Sigma= 1066.6240427124812 total t = 78219654598057.55 j = 94 Sigma[Inner] = 85.45730915698726
    dt= 1088232209.6369653 timestep= 10674 max Sigma= 1066.6435841878956 total t = 78225295720478.5 j = 94 Sigma[Inner] = 89.33845042734585
    dt= 4301155289.130569 timestep= 10684 max Sigma= 1066.758877423559 total t = 78257235007571.05 j = 93 Sigma[Inner] = 68.63605999208424
    dt= 4949185891.715814 timestep= 10694 max Sigma= 1066.9448871380685 total t = 78304262189848.31 j = 93 Sigma[Inner] = 60.45761482448945
    dt= 5345114322.518103 timestep= 10704 max Sigma= 1067.1505665417944 total t = 78355990059362.66 j = 93 Sigma[Inner] = 55.08088042335863
    dt= 5774106322.616108 timestep= 10714 max Sigma= 1067.3716520107164 total t = 78411652875372.47 j = 93 Sigma[Inner] = 51.30586116113145
    dt= 10564047394.495127 timestep= 10724 max Sigma= 1067.7004969307854 total t = 78498705638571.69 j = 92 Sigma[Inner] = 24.86411128477069
    dt= 10569975035.363596 timestep= 10734 max Sigma= 1068.1221346197397 total t = 78604378716386.27 j = 92 Sigma[Inner] = 26.71754999997889
    dt= 10575900025.569695 timestep= 10744 max Sigma= 1068.5431378562755 total t = 78710111056228.67 j = 92 Sigma[Inner] = 30.028119236911465
    dt= 10581822800.734873 timestep= 10754 max Sigma= 1068.9635045090304 total t = 78815902633387.38 j = 92 Sigma[Inner] = 36.33066636783405
    dt= 10587743747.776184 timestep= 10764 max Sigma= 1069.3832324716032 total t = 78921753427995.12 j = 92 Sigma[Inner] = 46.85483968368511
    dt= 10593663111.628115 timestep= 10774 max Sigma= 1069.8023196559793 total t = 79027663423202.92 j = 92 Sigma[Inner] = 61.674192520919284
    dt= 10599581058.454105 timestep= 10784 max Sigma= 1070.2207639885162 total t = 79133632604140.72 j = 92 Sigma[Inner] = 79.60399198791369
    dt= 10605497709.539062 timestep= 10794 max Sigma= 1070.6385634078747 total t = 79239660957333.0 j = 92 Sigma[Inner] = 98.83572867804564
    dt= 10611413159.975407 timestep= 10804 max Sigma= 1071.0557158639938 total t = 79345748470361.73 j = 92 Sigma[Inner] = 117.69864681099976
    dt= 10617327489.289759 timestep= 10814 max Sigma= 1071.4722193176453 total t = 79451895131667.61 j = 92 Sigma[Inner] = 135.07692311623055
    dt= 10623240767.685865 timestep= 10824 max Sigma= 1071.8880717403135 total t = 79558100930431.39 j = 92 Sigma[Inner] = 150.44262286871896
    dt= 10629153059.820534 timestep= 10834 max Sigma= 1072.3032711142632 total t = 79664365856502.95 j = 92 Sigma[Inner] = 163.6930350559779
    dt= 10635064427.125872 timestep= 10844 max Sigma= 1072.7178154327157 total t = 79770689900359.52 j = 92 Sigma[Inner] = 174.96352220715025
    dt= 10640974929.222782 timestep= 10854 max Sigma= 1073.1317027000894 total t = 79877073053081.8 j = 92 Sigma[Inner] = 184.49053063012053
    dt= 10646884624.728916 timestep= 10864 max Sigma= 1073.5449309322673 total t = 79983515306340.73 j = 92 Sigma[Inner] = 192.5322455582916
    dt= 10652793571.642332 timestep= 10874 max Sigma= 1073.9574981568817 total t = 80090016652389.84 j = 92 Sigma[Inner] = 199.3304124594545
    dt= 10658701827.423037 timestep= 10884 max Sigma= 1074.3694024135893 total t = 80196577084059.69 j = 92 Sigma[Inner] = 205.09589470573306
    dt= 1797574068.2736456 timestep= 10894 max Sigma= 1074.7806417543406 total t = 80294329559371.4 j = 92 Sigma[Inner] = 196.4214519111362
    dt= 419553304.70092195 timestep= 10904 max Sigma= 1074.8642374964138 total t = 80314653018273.95 j = 94 Sigma[Inner] = 116.0656863594334
    dt= 438195195.6365529 timestep= 10914 max Sigma= 1074.8834054256913 total t = 80319649313881.58 j = 94 Sigma[Inner] = 100.77640463093164
    dt= 3011957595.1793327 timestep= 10924 max Sigma= 1074.9264831127866 total t = 80333411508307.03 j = 93 Sigma[Inner] = 72.72634080883326
    dt= 4663524290.656818 timestep= 10934 max Sigma= 1075.0814839978161 total t = 80375341101160.2 j = 93 Sigma[Inner] = 64.60988405338182
    dt= 5152591888.935422 timestep= 10944 max Sigma= 1075.26997787674 total t = 80424855923724.34 j = 93 Sigma[Inner] = 57.71882543791992
    dt= 5552538134.495492 timestep= 10954 max Sigma= 1075.4747152067591 total t = 80478562011930.3 j = 93 Sigma[Inner] = 53.09141229550061
    dt= 10365075479.752987 timestep= 10964 max Sigma= 1075.7142931592984 total t = 80545825325522.48 j = 93 Sigma[Inner] = 47.41675892097591
    dt= 10683935509.067915 timestep= 10974 max Sigma= 1076.1221420102474 total t = 80652638107503.0 j = 92 Sigma[Inner] = 25.8468152011383
    dt= 10689843766.583578 timestep= 10984 max Sigma= 1076.530517796232 total t = 80759509959425.08 j = 92 Sigma[Inner] = 28.5056476443547
    dt= 10695750634.393784 timestep= 10994 max Sigma= 1076.9382186921232 total t = 80866440885778.25 j = 92 Sigma[Inner] = 33.579447226023206
    dt= 10701656584.546602 timestep= 11004 max Sigma= 1077.3452427999723 total t = 80973430875467.39 j = 92 Sigma[Inner] = 42.55469521828774
    


    
![png](output_5_89.png)
    



    
![png](output_5_90.png)
    



    
![png](output_5_91.png)
    


    dt= 10707561901.738657 timestep= 11014 max Sigma= 1077.7515882483356 total t = 81080479920993.22 j = 92 Sigma[Inner] = 55.98094624136052
    dt= 9003807903.568768 timestep= 11024 max Sigma= 1078.1507863441138 total t = 81184169289830.89 j = 93 Sigma[Inner] = 72.75799325926606
    dt= 9003807903.568768 timestep= 11034 max Sigma= 1078.4911241907644 total t = 81274207368866.52 j = 93 Sigma[Inner] = 88.65454977419337
    dt= 9003807903.56877 timestep= 11044 max Sigma= 1078.8308220716904 total t = 81364245447902.14 j = 93 Sigma[Inner] = 104.80531243868774
    dt= 9003807903.56877 timestep= 11054 max Sigma= 1079.1698798115694 total t = 81454283526937.77 j = 93 Sigma[Inner] = 120.33698900481043
    dt= 9003807903.56877 timestep= 11064 max Sigma= 1079.508297242298 total t = 81544321605973.39 j = 93 Sigma[Inner] = 134.6913487797427
    dt= 9003807903.56877 timestep= 11074 max Sigma= 1079.8460742030668 total t = 81634359685009.02 j = 93 Sigma[Inner] = 147.59858189176413
    dt= 9206573066.663607 timestep= 11084 max Sigma= 1080.1866760796627 total t = 81725526861243.45 j = 93 Sigma[Inner] = 159.10883441589883
    dt= 9364095382.27169 timestep= 11094 max Sigma= 1080.5335230010053 total t = 81818496503301.56 j = 93 Sigma[Inner] = 169.34115966612305
    dt= 9483359464.354668 timestep= 11104 max Sigma= 1080.8848747664424 total t = 81912819541407.52 j = 93 Sigma[Inner] = 178.3121064787988
    dt= 9584073156.89358 timestep= 11115 max Sigma= 1081.2750885380358 total t = 82017764741213.31 j = 93 Sigma[Inner] = 186.83437093162783
    dt= 9656357918.569248 timestep= 11125 max Sigma= 1081.632284840724 total t = 82114015562375.95 j = 93 Sigma[Inner] = 193.49554581174783
    dt= 9715677994.782555 timestep= 11135 max Sigma= 1081.9911970818803 total t = 82210914447360.58 j = 93 Sigma[Inner] = 199.25927661392362
    dt= 9765548891.493279 timestep= 11145 max Sigma= 1082.3513934146301 total t = 82308352180603.05 j = 93 Sigma[Inner] = 204.25080175924762
    dt= 9808392419.978315 timestep= 11155 max Sigma= 1082.7125555334185 total t = 82406248319118.36 j = 93 Sigma[Inner] = 208.5822298189533
    dt= 346719077.40631676 timestep= 11165 max Sigma= 1082.8465756258372 total t = 82433151011686.9 j = 94 Sigma[Inner] = 72.8450611351809
    dt= 355939690.64418936 timestep= 11175 max Sigma= 1082.8662382938444 total t = 82438498227334.28 j = 94 Sigma[Inner] = 72.91839646929931
    dt= 2160297340.755653 timestep= 11185 max Sigma= 1082.8939172413225 total t = 82447817840127.02 j = 93 Sigma[Inner] = 74.36135060556995
    dt= 4551982034.279855 timestep= 11195 max Sigma= 1083.0277833089137 total t = 82486572233204.97 j = 93 Sigma[Inner] = 66.12468450750417
    dt= 5107080614.194331 timestep= 11205 max Sigma= 1083.2053909431877 total t = 82535415167439.23 j = 93 Sigma[Inner] = 58.57744276498033
    dt= 5527910631.1799345 timestep= 11215 max Sigma= 1083.399960979417 total t = 82588792968279.61 j = 93 Sigma[Inner] = 53.535507856495194
    dt= 9895457745.746244 timestep= 11225 max Sigma= 1083.6282887689338 total t = 82655381937935.05 j = 93 Sigma[Inner] = 47.61005085837703
    dt= 9923987385.574453 timestep= 11235 max Sigma= 1083.9912704331723 total t = 82754494825017.48 j = 92 Sigma[Inner] = 26.865871686709724
    dt= 9950329727.270348 timestep= 11245 max Sigma= 1084.3544827391495 total t = 82853881348168.0 j = 92 Sigma[Inner] = 29.058319524458163
    dt= 9974718611.19133 timestep= 11255 max Sigma= 1084.7178410809695 total t = 82953520251652.77 j = 92 Sigma[Inner] = 33.15518681361985
    dt= 9997473571.409046 timestep= 11265 max Sigma= 1085.0812741866596 total t = 83053393826333.86 j = 92 Sigma[Inner] = 40.400843462149865
    dt= 10018843341.682526 timestep= 11275 max Sigma= 1085.44472181178 total t = 83153487150706.33 j = 92 Sigma[Inner] = 51.47347014735413
    dt= 10039024554.980253 timestep= 11285 max Sigma= 1085.8081324030227 total t = 83253787491214.66 j = 92 Sigma[Inner] = 66.04921933052248
    dt= 10058174935.530418 timestep= 11295 max Sigma= 1086.1714613699196 total t = 83354283857514.03 j = 92 Sigma[Inner] = 82.94724376406347
    dt= 10076422732.612988 timestep= 11305 max Sigma= 1086.5346697878078 total t = 83454966667820.17 j = 92 Sigma[Inner] = 100.66779809250296
    dt= 10093873556.839447 timestep= 11315 max Sigma= 1086.8977234106953 total t = 83555827493534.75 j = 92 Sigma[Inner] = 117.90679897541268
    dt= 10110615398.577522 timestep= 11325 max Sigma= 1087.2605919096907 total t = 83656858861710.81 j = 92 Sigma[Inner] = 133.80593603988916
    dt= 10126722358.567377 timestep= 11335 max Sigma= 1087.6232482776293 total t = 83758054100260.7 j = 92 Sigma[Inner] = 147.9531767535331
    dt= 10142257455.351574 timestep= 11345 max Sigma= 1087.9856683576045 total t = 83859407215143.89 j = 92 Sigma[Inner] = 160.2612983590341
    dt= 10157274763.23195 timestep= 11355 max Sigma= 1088.3478304649234 total t = 83960912791773.23 j = 92 Sigma[Inner] = 170.8318468137113
    dt= 10171821059.262772 timestep= 11365 max Sigma= 1088.7097150802442 total t = 84062565914977.2 j = 92 Sigma[Inner] = 179.85266263563852
    dt= 10185937106.233295 timestep= 11375 max Sigma= 1089.0713045975315 total t = 84164362103342.06 j = 92 Sigma[Inner] = 187.53545013261424
    dt= 10199658662.874012 timestep= 11385 max Sigma= 1089.4325831146077 total t = 84266297254819.89 j = 92 Sigma[Inner] = 194.08352365475915
    dt= 10213017287.518826 timestep= 11395 max Sigma= 1089.7935362571354 total t = 84368367601256.42 j = 92 Sigma[Inner] = 199.67807088857745
    dt= 10226040983.782705 timestep= 11405 max Sigma= 1090.1541510290494 total t = 84470569670052.0 j = 92 Sigma[Inner] = 204.4743493314896
    dt= 10238754724.202644 timestep= 11415 max Sigma= 1090.5144156841038 total t = 84572900251582.95 j = 92 Sigma[Inner] = 208.60260494612606
    dt= 349208683.9338371 timestep= 11425 max Sigma= 1090.6469149758466 total t = 84600699655349.1 j = 94 Sigma[Inner] = 72.61187797296718
    dt= 367806495.75320065 timestep= 11435 max Sigma= 1090.667011224759 total t = 84606437752472.05 j = 94 Sigma[Inner] = 71.07385582998752
    dt= 2240702067.0727057 timestep= 11445 max Sigma= 1090.6942122250332 total t = 84616053306397.9 j = 93 Sigma[Inner] = 73.98269820198716
    dt= 4579936419.388634 timestep= 11455 max Sigma= 1090.8234982562287 total t = 84655210019863.16 j = 93 Sigma[Inner] = 65.91405866984078
    dt= 5143163881.103909 timestep= 11465 max Sigma= 1090.9939827256153 total t = 84704367601887.61 j = 93 Sigma[Inner] = 58.303075437042224
    dt= 5591468296.820235 timestep= 11475 max Sigma= 1091.1811554509347 total t = 84758225928977.33 j = 93 Sigma[Inner] = 53.19224094156529
    dt= 10269873023.305983 timestep= 11485 max Sigma= 1091.4309721875568 total t = 84834284824843.38 j = 92 Sigma[Inner] = 25.503324004751434
    dt= 10281654852.260794 timestep= 11495 max Sigma= 1091.789914648079 total t = 84937048526292.02 j = 92 Sigma[Inner] = 26.812813858982842
    dt= 10293215172.300478 timestep= 11505 max Sigma= 1092.1484667813102 total t = 85039928837304.33 j = 92 Sigma[Inner] = 29.548637685066563
    


    
![png](output_5_93.png)
    



    
![png](output_5_94.png)
    



    
![png](output_5_95.png)
    


    dt= 10304563689.87429 timestep= 11515 max Sigma= 1092.5066220274425 total t = 85142923574798.67 j = 92 Sigma[Inner] = 34.67218947996513
    dt= 10315713955.55483 timestep= 11525 max Sigma= 1092.8643745414602 total t = 85246030696494.89 j = 92 Sigma[Inner] = 43.40704651797382
    dt= 10326678183.97271 timestep= 11535 max Sigma= 1093.2217191522711 total t = 85349248288053.5 j = 92 Sigma[Inner] = 56.12838326408749
    dt= 10337467475.339663 timestep= 11545 max Sigma= 1093.5786513055002 total t = 85452574550999.39 j = 92 Sigma[Inner] = 72.08986840817472
    dt= 10348091975.363487 timestep= 11555 max Sigma= 1093.9351670127835 total t = 85556007792496.03 j = 92 Sigma[Inner] = 89.80997456308954
    dt= 10358560997.154638 timestep= 11565 max Sigma= 1094.29126280584 total t = 85659546416495.94 j = 92 Sigma[Inner] = 107.72159542087749
    dt= 10368883118.313183 timestep= 11575 max Sigma= 1094.6469356942735 total t = 85763188915965.25 j = 92 Sigma[Inner] = 124.6366037579663
    dt= 10379066260.854246 timestep= 11585 max Sigma= 1095.0021831264128 total t = 85866933865980.11 j = 92 Sigma[Inner] = 139.88321620810513
    dt= 10389117758.619194 timestep= 11595 max Sigma= 1095.35700295273 total t = 85970779917550.62 j = 92 Sigma[Inner] = 153.22058381896656
    dt= 10399044415.132069 timestep= 11605 max Sigma= 1095.711393391508 total t = 86074725792064.08 j = 92 Sigma[Inner] = 164.6826152779248
    dt= 10408852553.881983 timestep= 11615 max Sigma= 1096.065352996521 total t = 86178770276264.16 j = 92 Sigma[Inner] = 174.44211023621776
    dt= 10418548062.42276 timestep= 11625 max Sigma= 1096.4188806265297 total t = 86282912217697.67 j = 92 Sigma[Inner] = 182.72166399441548
    dt= 10428136431.311502 timestep= 11635 max Sigma= 1096.7719754164618 total t = 86387150520573.25 j = 92 Sigma[Inner] = 189.74518641708528
    dt= 10437622788.666536 timestep= 11645 max Sigma= 1097.1246367501496 total t = 86491484141984.78 j = 92 Sigma[Inner] = 195.71589739893383
    dt= 10447011930.961157 timestep= 11655 max Sigma= 1097.4768642345387 total t = 86595912088459.33 j = 92 Sigma[Inner] = 200.8089892482484
    dt= 10456308350.554504 timestep= 11665 max Sigma= 1097.8286576752776 total t = 86700433412794.56 j = 92 Sigma[Inner] = 205.1714360689679
    dt= 10465516260.376354 timestep= 11675 max Sigma= 1098.1800170536335 total t = 86805047211155.58 j = 92 Sigma[Inner] = 208.9247960348613
    dt= 373133498.0409172 timestep= 11685 max Sigma= 1098.2759265580733 total t = 86823547595189.94 j = 94 Sigma[Inner] = 84.26308896809773
    dt= 382270945.1039672 timestep= 11695 max Sigma= 1098.2955829719956 total t = 86829419213650.52 j = 94 Sigma[Inner] = 74.08377715706816
    dt= 2788847019.3262134 timestep= 11705 max Sigma= 1098.3286375858015 total t = 86841685750949.44 j = 93 Sigma[Inner] = 72.94150226907497
    dt= 4681494986.5704365 timestep= 11715 max Sigma= 1098.4609788836076 total t = 86883073430268.36 j = 93 Sigma[Inner] = 64.81475461360075
    dt= 5225908382.144176 timestep= 11725 max Sigma= 1098.6265889651715 total t = 86933085129154.22 j = 93 Sigma[Inner] = 57.43035514100771
    dt= 5727774091.128954 timestep= 11735 max Sigma= 1098.8084357331402 total t = 86987960459647.58 j = 93 Sigma[Inner] = 52.41818178370583
    dt= 10488776698.272322 timestep= 11745 max Sigma= 1099.0807944749242 total t = 87074266806565.6 j = 92 Sigma[Inner] = 25.153055369631538
    dt= 10497703819.150244 timestep= 11755 max Sigma= 1099.4306061197315 total t = 87179203733987.14 j = 92 Sigma[Inner] = 27.024890600697063
    dt= 10506557303.363792 timestep= 11765 max Sigma= 1099.7799854018385 total t = 87284229525766.89 j = 92 Sigma[Inner] = 30.275642090482265
    dt= 10515340506.689392 timestep= 11775 max Sigma= 1100.1289328678954 total t = 87389343463006.7 j = 92 Sigma[Inner] = 36.30795822978493
    dt= 10524056670.416388 timestep= 11785 max Sigma= 1100.4774491347025 total t = 87494544861036.0 j = 92 Sigma[Inner] = 46.22559789361795
    dt= 10532708715.314812 timestep= 11795 max Sigma= 1100.8255348683088 total t = 87599833065747.02 j = 92 Sigma[Inner] = 60.09172626697271
    dt= 10541299323.777584 timestep= 11805 max Sigma= 1101.1731907659414 total t = 87705207450877.42 j = 92 Sigma[Inner] = 76.84762473490669
    dt= 10549830989.128984 timestep= 11815 max Sigma= 1101.5204175407403 total t = 87810667415917.23 j = 92 Sigma[Inner] = 94.86483703204107
    dt= 10558306046.922726 timestep= 11825 max Sigma= 1101.8672159084292 total t = 87916212384401.17 j = 92 Sigma[Inner] = 112.61454956562136
    dt= 10566726696.085339 timestep= 11835 max Sigma= 1102.2135865754406 total t = 88021841802453.53 j = 92 Sigma[Inner] = 129.04925208668723
    dt= 10575095014.141737 timestep= 11845 max Sigma= 1102.5595302282163 total t = 88127555137509.31 j = 92 Sigma[Inner] = 143.64997712419054
    dt= 10583412968.851782 timestep= 11855 max Sigma= 1102.9050475235122 total t = 88233351877166.55 j = 92 Sigma[Inner] = 156.29263808476944
    dt= 10591682427.564466 timestep= 11865 max Sigma= 1103.2501390795944 total t = 88339231528141.6 j = 92 Sigma[Inner] = 167.08262048984537
    dt= 10599905165.039164 timestep= 11875 max Sigma= 1103.594805468252 total t = 88445193615309.17 j = 92 Sigma[Inner] = 176.2283992536219
    dt= 10608082870.175154 timestep= 11885 max Sigma= 1103.9390472075647 total t = 88551237680814.11 j = 92 Sigma[Inner] = 183.96547252215368
    dt= 10616217151.916826 timestep= 11895 max Sigma= 1104.282864755379 total t = 88657363283245.94 j = 92 Sigma[Inner] = 190.51817713975845
    dt= 10624309544.50399 timestep= 11905 max Sigma= 1104.6262585034465 total t = 88763569996868.61 j = 92 Sigma[Inner] = 196.08424998729328
    dt= 10632361512.180466 timestep= 11915 max Sigma= 1104.9692287721928 total t = 88869857410899.89 j = 92 Sigma[Inner] = 200.83118361514974
    dt= 10640374453.442682 timestep= 11925 max Sigma= 1105.3117758060764 total t = 88976225128834.98 j = 92 Sigma[Inner] = 204.89792247495726
    dt= 10648349704.891884 timestep= 11935 max Sigma= 1105.6538997695066 total t = 89082672767810.62 j = 92 Sigma[Inner] = 208.3985322201115
    dt= 376945024.85445946 timestep= 11945 max Sigma= 1105.7483395558004 total t = 89101820884627.28 j = 94 Sigma[Inner] = 83.76668051545332
    dt= 387485224.28530616 timestep= 11955 max Sigma= 1105.7674545531302 total t = 89107788425981.27 j = 94 Sigma[Inner] = 73.63732988140542
    dt= 2844481182.5133495 timestep= 11965 max Sigma= 1105.799960753278 total t = 89120377114120.06 j = 93 Sigma[Inner] = 72.8745444236753
    dt= 4709659392.118096 timestep= 11975 max Sigma= 1105.9277726712476 total t = 89162096885915.16 j = 93 Sigma[Inner] = 64.60879157497813
    dt= 5266750740.889894 timestep= 11985 max Sigma= 1106.0873304297018 total t = 89212448712622.25 j = 93 Sigma[Inner] = 57.14741256682908
    dt= 5846963361.10154 timestep= 11995 max Sigma= 1106.263261500622 total t = 89267986661127.77 j = 93 Sigma[Inner] = 52.04628589046457
    dt= 10669240933.830784 timestep= 12005 max Sigma= 1106.5560275085147 total t = 89364384919817.84 j = 92 Sigma[Inner] = 25.37916087005651
    


    
![png](output_5_97.png)
    



    
![png](output_5_98.png)
    



    
![png](output_5_99.png)
    


    dt= 10677093014.84081 timestep= 12015 max Sigma= 1106.8966100129408 total t = 89471120543953.61 j = 92 Sigma[Inner] = 27.57391788308923
    dt= 10684911549.248245 timestep= 12025 max Sigma= 1107.2367691631125 total t = 89577934503026.69 j = 92 Sigma[Inner] = 31.470489484196126
    dt= 10692698108.10368 timestep= 12035 max Sigma= 1107.576504583454 total t = 89684826470396.44 j = 92 Sigma[Inner] = 38.53640249224021
    dt= 10700454010.939947 timestep= 12045 max Sigma= 1107.9158158171697 total t = 89791796133740.16 j = 92 Sigma[Inner] = 49.67773447239473
    dt= 10708180405.97896 timestep= 12055 max Sigma= 1108.254702319687 total t = 89898843192925.12 j = 92 Sigma[Inner] = 64.60857290751679
    dt= 10715878328.49756 timestep= 12065 max Sigma= 1108.593163454909 total t = 90005967358644.19 j = 92 Sigma[Inner] = 81.99095657430911
    dt= 10723548733.947382 timestep= 12075 max Sigma= 1108.9311984931874 total t = 90113168351483.44 j = 92 Sigma[Inner] = 100.1175474038473
    dt= 10731192517.411776 timestep= 12085 max Sigma= 1109.2688066103683 total t = 90220445901239.23 j = 92 Sigma[Inner] = 117.55226290428922
    dt= 10738810525.52371 timestep= 12095 max Sigma= 1109.605986887553 total t = 90327799746384.5 j = 92 Sigma[Inner] = 133.40936987140418
    dt= 10746403564.132952 timestep= 12105 max Sigma= 1109.9427383113596 total t = 90435229633629.12 j = 92 Sigma[Inner] = 147.3182754149056
    dt= 10753972403.514856 timestep= 12115 max Sigma= 1110.279059774569 total t = 90542735317542.23 j = 92 Sigma[Inner] = 159.2568231561267
    dt= 10761517782.112528 timestep= 12125 max Sigma= 1110.6149500770703 total t = 90650316560217.95 j = 92 Sigma[Inner] = 169.3869994999082
    dt= 10769040409.368784 timestep= 12135 max Sigma= 1110.9504079270541 total t = 90757973130973.38 j = 92 Sigma[Inner] = 177.94195468117482
    dt= 10776540967.964544 timestep= 12145 max Sigma= 1111.28543194242 total t = 90865704806071.05 j = 92 Sigma[Inner] = 185.16317523715816
    dt= 10784020115.646807 timestep= 12155 max Sigma= 1111.6200206523577 total t = 90973511368461.83 j = 92 Sigma[Inner] = 191.27155042460225
    dt= 10791478486.755589 timestep= 12165 max Sigma= 1111.954172499088 total t = 91081392607544.28 j = 92 Sigma[Inner] = 196.45739406252957
    dt= 10798916693.518122 timestep= 12175 max Sigma= 1112.2878858397396 total t = 91189348318937.64 j = 92 Sigma[Inner] = 200.87968174170138
    dt= 10806335327.157442 timestep= 12185 max Sigma= 1112.6211589483382 total t = 91297378304267.08 j = 92 Sigma[Inner] = 204.66911993354327
    dt= 10813734958.850946 timestep= 12195 max Sigma= 1112.9539900179007 total t = 91405482370959.03 j = 92 Sigma[Inner] = 207.932381319854
    dt= 1076336695.3835595 timestep= 12205 max Sigma= 1113.1075084274858 total t = 91445678736897.28 j = 94 Sigma[Inner] = 85.70689010538969
    dt= 341365116.07916135 timestep= 12215 max Sigma= 1113.1326344911688 total t = 91453121416873.47 j = 94 Sigma[Inner] = 65.78142232337925
    dt= 1458490873.897862 timestep= 12225 max Sigma= 1113.1503433499388 total t = 91460002924296.7 j = 93 Sigma[Inner] = 80.7210622869979
    dt= 4490652865.56538 timestep= 12235 max Sigma= 1113.2502653315635 total t = 91495571299740.55 j = 93 Sigma[Inner] = 67.21166867059877
    dt= 5156782259.61548 timestep= 12245 max Sigma= 1113.3983602400383 total t = 91544495388020.81 j = 93 Sigma[Inner] = 58.782425491911106
    dt= 5698218651.849916 timestep= 12255 max Sigma= 1113.5635462086595 total t = 91598915244809.81 j = 93 Sigma[Inner] = 53.06282103569782
    dt= 10832836567.783138 timestep= 12265 max Sigma= 1113.8154614727016 total t = 91686318903817.02 j = 92 Sigma[Inner] = 24.827810781963915
    dt= 10840175734.310278 timestep= 12275 max Sigma= 1114.146687083912 total t = 91794687648552.02 j = 92 Sigma[Inner] = 26.910408820379523
    dt= 10847497278.715193 timestep= 12285 max Sigma= 1114.4774614664666 total t = 91903129688794.27 j = 92 Sigma[Inner] = 30.473703760452693
    dt= 10854801827.911047 timestep= 12295 max Sigma= 1114.807782368516 total t = 92011644850267.14 j = 92 Sigma[Inner] = 37.013709200863495
    dt= 10862090172.870504 timestep= 12305 max Sigma= 1115.1376474850624 total t = 92120232967531.62 j = 92 Sigma[Inner] = 47.605505703669905
    dt= 10869362950.831194 timestep= 12315 max Sigma= 1115.4670544561613 total t = 92228893882146.42 j = 92 Sigma[Inner] = 62.14033012356342
    dt= 10876620706.154095 timestep= 12325 max Sigma= 1115.7960008660411 total t = 92337627441495.36 j = 92 Sigma[Inner] = 79.35508514592179
    dt= 10883863924.103949 timestep= 12335 max Sigma= 1116.1244842436813 total t = 92446433498060.23 j = 92 Sigma[Inner] = 97.50733907006493
    dt= 10891093050.07739 timestep= 12345 max Sigma= 1116.4525020641984 total t = 92555311908944.36 j = 92 Sigma[Inner] = 115.07691495241775
    dt= 10898308500.873568 timestep= 12355 max Sigma= 1116.7800517506842 total t = 92664262535540.67 j = 92 Sigma[Inner] = 131.10163529620868
    dt= 10905510671.531532 timestep= 12365 max Sigma= 1117.1071306762988 total t = 92773285243286.0 j = 92 Sigma[Inner] = 145.1644039377344
    dt= 10912699939.64493 timestep= 12375 max Sigma= 1117.4337361665116 total t = 92882379901468.77 j = 92 Sigma[Inner] = 157.2237405134447
    dt= 10919876668.204803 timestep= 12385 max Sigma= 1117.759865501416 total t = 92991546383071.11 j = 92 Sigma[Inner] = 167.4388356452028
    dt= 10927041207.55544 timestep= 12395 max Sigma= 1118.0855159180855 total t = 93100784564634.19 j = 92 Sigma[Inner] = 176.04749753275445
    dt= 10934193896.792711 timestep= 12405 max Sigma= 1118.4106846129407 total t = 93210094326139.77 j = 92 Sigma[Inner] = 183.29799142525528
    dt= 10941335064.793133 timestep= 12415 max Sigma= 1118.7353687441164 total t = 93319475550903.58 j = 92 Sigma[Inner] = 189.41765724836526
    dt= 10948465030.98341 timestep= 12425 max Sigma= 1119.0595654338053 total t = 93428928125477.64 j = 92 Sigma[Inner] = 194.6020975789596
    dt= 10955584105.917858 timestep= 12435 max Sigma= 1119.3832717705823 total t = 93538451939559.03 j = 92 Sigma[Inner] = 199.0143188657546
    dt= 10962692591.708532 timestep= 12445 max Sigma= 1119.7064848116918 total t = 93648046885903.61 j = 92 Sigma[Inner] = 202.78797576027685
    dt= 10969790782.341595 timestep= 12455 max Sigma= 1120.0292015852995 total t = 93757712860243.48 j = 92 Sigma[Inner] = 206.0318423674818
    dt= 10976878963.908123 timestep= 12465 max Sigma= 1120.3514190926976 total t = 93867449761207.0 j = 92 Sigma[Inner] = 208.83423863958782
    dt= 460103565.2775064 timestep= 12475 max Sigma= 1120.4086831074078 total t = 93876460152696.02 j = 94 Sigma[Inner] = 117.186326653977
    dt= 413187839.78937733 timestep= 12485 max Sigma= 1120.4367307759983 total t = 93885980855358.78 j = 94 Sigma[Inner] = 64.7052605274319
    dt= 2639358328.7944865 timestep= 12495 max Sigma= 1120.463514730983 total t = 93897345316128.03 j = 93 Sigma[Inner] = 72.62023194416251
    dt= 4706907702.401573 timestep= 12505 max Sigma= 1120.5778617901747 total t = 93938444225235.62 j = 93 Sigma[Inner] = 64.88209962233803
    


    
![png](output_5_101.png)
    



    
![png](output_5_102.png)
    



    
![png](output_5_103.png)
    


    dt= 5306145023.10498 timestep= 12515 max Sigma= 1120.724111406646 total t = 93989008774943.12 j = 93 Sigma[Inner] = 57.08366598788003
    dt= 6475731428.015215 timestep= 12525 max Sigma= 1120.887334621992 total t = 94046001869985.14 j = 93 Sigma[Inner] = 51.676777321648295
    dt= 10994987147.150661 timestep= 12535 max Sigma= 1121.1747567830937 total t = 94148970670361.36 j = 92 Sigma[Inner] = 25.15382631014743
    dt= 11002045089.242544 timestep= 12545 max Sigma= 1121.4951705887256 total t = 94258959369109.5 j = 92 Sigma[Inner] = 27.622588431500706
    dt= 11009093044.942966 timestep= 12555 max Sigma= 1121.8150712409204 total t = 94369018591662.28 j = 92 Sigma[Inner] = 32.05432775728276
    dt= 11016131732.458134 timestep= 12565 max Sigma= 1122.134455601502 total t = 94479148242303.77 j = 92 Sigma[Inner] = 39.956472561761856
    dt= 11023161666.034533 timestep= 12575 max Sigma= 1122.4533205298505 total t = 94589348231309.03 j = 92 Sigma[Inner] = 52.06242163437389
    dt= 11030183239.320974 timestep= 12585 max Sigma= 1122.7716628800501 total t = 94699618473378.48 j = 92 Sigma[Inner] = 67.79083465749856
    dt= 11037196774.740788 timestep= 12595 max Sigma= 1123.0894795000263 total t = 94809958886727.62 j = 92 Sigma[Inner] = 85.57675925667822
    dt= 11044202550.94617 timestep= 12605 max Sigma= 1123.406767231761 total t = 94920369392538.77 j = 92 Sigma[Inner] = 103.65946536972055
    dt= 11051200818.447224 timestep= 12615 max Sigma= 1123.7235229120952 total t = 95030849914616.3 j = 92 Sigma[Inner] = 120.691512172449
    dt= 11058191808.749504 timestep= 12625 max Sigma= 1124.039743373837 total t = 95141400379159.47 j = 92 Sigma[Inner] = 135.92872303939757
    dt= 11065175739.860287 timestep= 12635 max Sigma= 1124.3554254470332 total t = 95252020714604.8 j = 92 Sigma[Inner] = 149.12715028851892
    dt= 11072152819.719528 timestep= 12645 max Sigma= 1124.6705659603172 total t = 95362710851511.31 j = 92 Sigma[Inner] = 160.35087448711332
    dt= 11079123248.413187 timestep= 12655 max Sigma= 1124.9851617422876 total t = 95473470722472.77 j = 92 Sigma[Inner] = 169.80982545286042
    dt= 11086087219.647251 timestep= 12665 max Sigma= 1125.299209622882 total t = 95584300262047.66 j = 92 Sigma[Inner] = 177.75828306521078
    dt= 11093044921.752375 timestep= 12675 max Sigma= 1125.6127064347402 total t = 95695199406701.34 j = 92 Sigma[Inner] = 184.44305422826054
    dt= 11099996538.373466 timestep= 12685 max Sigma= 1125.9256490145383 total t = 95806168094756.16 j = 92 Sigma[Inner] = 190.08220756516386
    dt= 11106942248.935713 timestep= 12695 max Sigma= 1126.2380342042843 total t = 95917206266347.53 j = 92 Sigma[Inner] = 194.85966940926997
    dt= 11113882228.943878 timestep= 12705 max Sigma= 1126.549858852578 total t = 96028313863383.23 j = 92 Sigma[Inner] = 198.92692785768227
    dt= 11120816650.154915 timestep= 12715 max Sigma= 1126.8611198158271 total t = 96139490829505.45 j = 92 Sigma[Inner] = 202.407319388564
    dt= 11127745680.655071 timestep= 12725 max Sigma= 1127.1718139594168 total t = 96250737110053.7 j = 92 Sigma[Inner] = 205.40079794003233
    dt= 11134669484.868803 timestep= 12735 max Sigma= 1127.4819381588325 total t = 96362052652028.27 j = 92 Sigma[Inner] = 207.98832582901443
    dt= 1100250518.4793732 timestep= 12745 max Sigma= 1127.624481494036 total t = 96403272392282.88 j = 94 Sigma[Inner] = 85.12797261062522
    dt= 1012129798.5224712 timestep= 12755 max Sigma= 1127.6528938694212 total t = 96413408405777.98 j = 92 Sigma[Inner] = 64.24062434306512
    dt= 1158486373.1063867 timestep= 12765 max Sigma= 1127.6692928304299 total t = 96419457012303.72 j = 94 Sigma[Inner] = 85.8912278949425
    dt= 4444754524.41903 timestep= 12775 max Sigma= 1127.7534518625762 total t = 96453044764045.9 j = 93 Sigma[Inner] = 67.83858451328278
    dt= 5175958872.442954 timestep= 12785 max Sigma= 1127.8870561585024 total t = 96501921549294.62 j = 93 Sigma[Inner] = 58.961137107904385
    dt= 5817838787.859075 timestep= 12795 max Sigma= 1128.0376871443043 total t = 96556906517586.02 j = 93 Sigma[Inner] = 52.87085461111732
    dt= 11152890328.387478 timestep= 12805 max Sigma= 1128.2968356607096 total t = 96655884884699.3 j = 92 Sigma[Inner] = 24.96936285032059
    dt= 11159799677.855944 timestep= 12815 max Sigma= 1128.6048606842646 total t = 96767451794344.12 j = 92 Sigma[Inner] = 27.495111579831203
    dt= 11166703324.459538 timestep= 12825 max Sigma= 1128.912301513172 total t = 96879087765605.52 j = 92 Sigma[Inner] = 31.8959001011418
    dt= 11173601873.654007 timestep= 12835 max Sigma= 1129.2191550714322 total t = 96990792744877.56 j = 92 Sigma[Inner] = 39.732293419687714
    dt= 11180495749.02024 timestep= 12845 max Sigma= 1129.5254183102693 total t = 97102566683646.33 j = 92 Sigma[Inner] = 51.79726591612613
    dt= 11187385252.668428 timestep= 12855 max Sigma= 1129.831088204407 total t = 97214409536909.66 j = 92 Sigma[Inner] = 67.52435853057891
    dt= 11194270616.458944 timestep= 12865 max Sigma= 1130.1361617502284 total t = 97326321262269.38 j = 92 Sigma[Inner] = 85.31970373291739
    dt= 11201152030.34783 timestep= 12875 max Sigma= 1130.440635965019 total t = 97438301819398.27 j = 92 Sigma[Inner] = 103.38554857330193
    dt= 11208029658.426895 timestep= 12885 max Sigma= 1130.744507886792 total t = 97550351169716.58 j = 92 Sigma[Inner] = 120.35558787027514
    dt= 11214903648.230022 timestep= 12895 max Sigma= 1131.0477745744336 total t = 97662469276188.69 j = 92 Sigma[Inner] = 135.4874027602728
    dt= 11221774136.285147 timestep= 12905 max Sigma= 1131.3504331080235 total t = 97774656103190.08 j = 92 Sigma[Inner] = 148.54977589650892
    dt= 11228641251.528605 timestep= 12915 max Sigma= 1131.652480589242 total t = 97886911616417.69 j = 92 Sigma[Inner] = 159.62130773859778
    dt= 11235505117.470232 timestep= 12925 max Sigma= 1131.953914141826 total t = 97999235782826.48 j = 92 Sigma[Inner] = 168.92364123233438
    dt= 11242365853.603632 timestep= 12935 max Sigma= 1132.2547309120384 total t = 98111628570583.61 j = 92 Sigma[Inner] = 176.71891382613137
    dt= 11249223576.339472 timestep= 12945 max Sigma= 1132.5549280691396 total t = 98224089949033.42 j = 92 Sigma[Inner] = 183.25850098728174
    dt= 11256078399.62035 timestep= 12955 max Sigma= 1132.854502805851 total t = 98336619888670.48 j = 92 Sigma[Inner] = 188.76270847810488
    dt= 11262930435.310411 timestep= 12965 max Sigma= 1133.153452338802 total t = 98449218361117.1 j = 92 Sigma[Inner] = 193.41621282908213
    dt= 11269779793.417639 timestep= 12975 max Sigma= 1133.4517739089522 total t = 98561885339103.89 j = 92 Sigma[Inner] = 197.37037677230413
    dt= 11276626582.188993 timestep= 12985 max Sigma= 1133.7494647819979 total t = 98674620796451.81 j = 92 Sigma[Inner] = 200.74793802877142
    dt= 11283470908.10965 timestep= 12995 max Sigma= 1134.046522248744 total t = 98787424708054.69 j = 92 Sigma[Inner] = 203.64802793339493
    dt= 11290312875.834238 timestep= 13005 max Sigma= 1134.342943625454 total t = 98900297049861.12 j = 92 Sigma[Inner] = 206.15071026862427
    


    
![png](output_5_105.png)
    



    
![png](output_5_106.png)
    



    
![png](output_5_107.png)
    


    dt= 11297152588.075447 timestep= 13015 max Sigma= 1134.6387262541723 total t = 99013237798855.73 j = 92 Sigma[Inner] = 208.3208007693267
    dt= 401437442.7091137 timestep= 13025 max Sigma= 1134.7478122817324 total t = 99044072770829.14 j = 93 Sigma[Inner] = 65.27127767452318
    dt= 772996129.7104596 timestep= 13035 max Sigma= 1134.7827558976091 total t = 99057822174964.52 j = 92 Sigma[Inner] = 64.24189905118035
    dt= 1311425243.98418 timestep= 13045 max Sigma= 1134.7984388906182 total t = 99064366257407.62 j = 94 Sigma[Inner] = 81.2127782633072
    dt= 4508588614.967523 timestep= 13055 max Sigma= 1134.8816205155606 total t = 99099430791006.8 j = 93 Sigma[Inner] = 67.29169767638855
    dt= 5234318144.231438 timestep= 13065 max Sigma= 1135.0086415653445 total t = 99148866593713.56 j = 93 Sigma[Inner] = 58.45392958926835
    dt= 6423892747.92428 timestep= 13075 max Sigma= 1135.1524784311268 total t = 99205285199751.77 j = 93 Sigma[Inner] = 52.31797469553347
    dt= 11315097650.69217 timestep= 13085 max Sigma= 1135.412539865317 total t = 99310218377372.05 j = 92 Sigma[Inner] = 24.955962806737983
    dt= 11321933111.296616 timestep= 13095 max Sigma= 1135.7059838194323 total t = 99423406951639.84 j = 92 Sigma[Inner] = 27.67374508950452
    dt= 11328765605.892939 timestep= 13105 max Sigma= 1135.9987771394922 total t = 99536663863655.52 j = 92 Sigma[Inner] = 32.47400813608181
    dt= 11335595687.218243 timestep= 13115 max Sigma= 1136.290917292811 total t = 99649989086982.6 j = 92 Sigma[Inner] = 40.9012122571475
    dt= 11342423713.437372 timestep= 13125 max Sigma= 1136.5824017837406 total t = 99763382599581.1 j = 92 Sigma[Inner] = 53.57969104303537
    dt= 11349249928.506886 timestep= 13135 max Sigma= 1136.8732281497037 total t = 99876844382311.56 j = 92 Sigma[Inner] = 69.7323008832816
    dt= 11356074510.753815 timestep= 13145 max Sigma= 1137.1633939590045 total t = 99990374418084.02 j = 92 Sigma[Inner] = 87.64856471306312
    dt= 11362897599.795029 timestep= 13155 max Sigma= 1137.4528968096156 total t = 100103972691362.5 j = 92 Sigma[Inner] = 105.5448390094335
    dt= 11369719311.78725 timestep= 13165 max Sigma= 1137.7417343285115 total t = 100217639187868.84 j = 92 Sigma[Inner] = 122.14654724671475
    dt= 11376539748.27989 timestep= 13175 max Sigma= 1138.0299041712992 total t = 100331373894400.53 j = 92 Sigma[Inner] = 136.81362745372263
    dt= 11383359001.493448 timestep= 13185 max Sigma= 1138.317404022019 total t = 100445176798715.94 j = 92 Sigma[Inner] = 149.3914922954969
    dt= 11390177157.556593 timestep= 13195 max Sigma= 1138.6042315930454 total t = 100559047889460.03 j = 92 Sigma[Inner] = 160.00340105683733
    dt= 11396994298.5459 timestep= 13205 max Sigma= 1138.8903846250382 total t = 100672987156115.39 j = 92 Sigma[Inner] = 168.8913149725375
    dt= 11403810503.798033 timestep= 13215 max Sigma= 1139.175860886925 total t = 100786994588969.53 j = 92 Sigma[Inner] = 176.32300209128934
    dt= 9914380202.811317 timestep= 13225 max Sigma= 1139.442028801473 total t = 100892102927837.48 j = 92 Sigma[Inner] = 182.16938204385536
    dt= 9914380202.811317 timestep= 13235 max Sigma= 1139.6889153839988 total t = 100991246729865.61 j = 92 Sigma[Inner] = 186.81203127861406
    dt= 9914380202.811317 timestep= 13245 max Sigma= 1139.935159049453 total t = 101090390531893.73 j = 92 Sigma[Inner] = 190.81746533615507
    dt= 9914380202.811317 timestep= 13255 max Sigma= 1140.1807594179984 total t = 101189534333921.86 j = 92 Sigma[Inner] = 194.2866505666029
    dt= 9914380202.811317 timestep= 13265 max Sigma= 1140.425716127204 total t = 101288678135949.98 j = 92 Sigma[Inner] = 197.30352118074532
    dt= 9987594126.908459 timestep= 13275 max Sigma= 1140.6702746000724 total t = 101387995004065.9 j = 92 Sigma[Inner] = 199.94012466477855
    dt= 10159355639.143536 timestep= 13285 max Sigma= 1140.9177384181594 total t = 101488856991057.56 j = 92 Sigma[Inner] = 202.28274382278752
    dt= 10288655467.754913 timestep= 13295 max Sigma= 1141.168244488295 total t = 101591190835210.98 j = 92 Sigma[Inner] = 204.3691291844617
    dt= 10388179623.780752 timestep= 13305 max Sigma= 1141.4208734139095 total t = 101694645190849.84 j = 92 Sigma[Inner] = 206.2241472477349
    dt= 10466766755.722721 timestep= 13315 max Sigma= 1141.674980201702 total t = 101798973647260.64 j = 92 Sigma[Inner] = 207.8730736916381
    dt= 653783712.4360919 timestep= 13325 max Sigma= 1141.8409776149838 total t = 101857421919533.39 j = 94 Sigma[Inner] = 157.58972114683397
    dt= 1910027507.537892 timestep= 13335 max Sigma= 1141.8580168330766 total t = 101865692987846.84 j = 93 Sigma[Inner] = 73.4436863749396
    dt= 713955825.5676662 timestep= 13345 max Sigma= 1141.8890472812616 total t = 101877274865662.78 j = 94 Sigma[Inner] = 129.17392408017852
    dt= 4117831336.175865 timestep= 13355 max Sigma= 1141.9412840984576 total t = 101902199123915.67 j = 93 Sigma[Inner] = 70.47441089998367
    dt= 5082036265.696254 timestep= 13365 max Sigma= 1142.0535332872216 total t = 101949448693872.5 j = 93 Sigma[Inner] = 60.65344480363079
    dt= 5770065903.216002 timestep= 13375 max Sigma= 1142.18353590921 total t = 102003814713488.39 j = 93 Sigma[Inner] = 53.69231154368501
    dt= 10619697505.676521 timestep= 13385 max Sigma= 1142.3920016705563 total t = 102094898857930.94 j = 92 Sigma[Inner] = 25.44442482932223
    dt= 10659923516.09888 timestep= 13395 max Sigma= 1142.6485068821714 total t = 102201321586234.05 j = 92 Sigma[Inner] = 27.560377294322326
    dt= 10695339214.335442 timestep= 13405 max Sigma= 1142.9051812519208 total t = 102308119079447.25 j = 92 Sigma[Inner] = 31.142328870115747
    dt= 10727020318.89319 timestep= 13415 max Sigma= 1143.1619145653299 total t = 102415249439891.17 j = 92 Sigma[Inner] = 37.52003397038837
    dt= 10755747563.779547 timestep= 13425 max Sigma= 1143.4186203495985 total t = 102522679816684.12 j = 92 Sigma[Inner] = 47.58061573029676
    dt= 10782096693.03955 timestep= 13435 max Sigma= 1143.675229591557 total t = 102630383978228.84 j = 92 Sigma[Inner] = 61.17025958236065
    dt= 10806499355.054249 timestep= 13445 max Sigma= 1143.9316863627878 total t = 102738340616882.42 j = 92 Sigma[Inner] = 77.15503740779755
    dt= 10829284469.984749 timestep= 13455 max Sigma= 1144.1879447306846 total t = 102846532148476.64 j = 92 Sigma[Inner] = 93.99540393816818
    dt= 10850706670.228064 timestep= 13465 max Sigma= 1144.4439665452326 total t = 102954943850079.84 j = 92 Sigma[Inner] = 110.33665008801265
    dt= 10870966097.949743 timestep= 13475 max Sigma= 1144.699719829071 total t = 103063563231803.8 j = 92 Sigma[Inner] = 125.30106982181482
    dt= 10890222361.004473 timestep= 13485 max Sigma= 1144.9551775880511 total t = 103172379572601.7 j = 92 Sigma[Inner] = 138.4889648157165
    dt= 10908604495.700176 timestep= 13495 max Sigma= 1145.2103169183033 total t = 103281383572432.28 j = 92 Sigma[Inner] = 149.84043854794024
    dt= 10926218168.428213 timestep= 13505 max Sigma= 1145.4651183247984 total t = 103390567088040.3 j = 92 Sigma[Inner] = 159.48457611018281
    


    
![png](output_5_109.png)
    



    
![png](output_5_110.png)
    



    
![png](output_5_111.png)
    


    dt= 10943150946.012129 timestep= 13515 max Sigma= 1145.7195651924653 total t = 103499922929572.33 j = 92 Sigma[Inner] = 167.62974611229427
    dt= 10959476199.75937 timestep= 13525 max Sigma= 1145.9736433685978 total t = 103609444701998.73 j = 92 Sigma[Inner] = 174.49975638007143
    dt= 10975256032.015383 timestep= 13535 max Sigma= 1146.227340827341 total t = 103719126679934.83 j = 92 Sigma[Inner] = 180.30295872440686
    dt= 10990543495.62862 timestep= 13545 max Sigma= 1146.4806473954172 total t = 103828963707653.27 j = 92 Sigma[Inner] = 185.22060059542596
    dt= 11005384296.366186 timestep= 13555 max Sigma= 1146.7335545240644 total t = 103938951118316.55 j = 92 Sigma[Inner] = 189.4049074964356
    dt= 11019818113.210669 timestep= 13565 max Sigma= 1146.986055096288 total t = 104049084668038.27 j = 92 Sigma[Inner] = 192.98141300333907
    dt= 11033879633.303928 timestep= 13575 max Sigma= 1147.2381432614418 total t = 104159360481510.75 j = 92 Sigma[Inner] = 196.05272879916663
    dt= 11047599371.617266 timestep= 13585 max Sigma= 1147.4898142912602 total t = 104269775006748.81 j = 92 Sigma[Inner] = 198.70245646281177
    dt= 11061004326.589485 timestep= 13595 max Sigma= 1147.7410644529837 total t = 104380324977092.08 j = 92 Sigma[Inner] = 200.99871569151009
    dt= 11074118509.554592 timestep= 13605 max Sigma= 1147.9918908963173 total t = 104491007379042.81 j = 92 Sigma[Inner] = 202.99713082167943
    dt= 11086963376.13394 timestep= 13615 max Sigma= 1148.2422915517848 total t = 104601819424838.8 j = 92 Sigma[Inner] = 204.7432779739445
    dt= 11099558180.771933 timestep= 13625 max Sigma= 1148.4922650386272 total t = 104712758528902.25 j = 92 Sigma[Inner] = 206.27465606245164
    dt= 11111920270.477985 timestep= 13635 max Sigma= 1148.7418105808408 total t = 104823822287488.25 j = 92 Sigma[Inner] = 207.62226049734394
    dt= 1328787523.298225 timestep= 13645 max Sigma= 1148.9709934539653 total t = 104916305734387.45 j = 93 Sigma[Inner] = 108.68207498638232
    dt= 675776635.4230471 timestep= 13655 max Sigma= 1148.987832465395 total t = 104923178053295.9 j = 94 Sigma[Inner] = 132.41650563412688
    dt= 481927052.0036151 timestep= 13665 max Sigma= 1149.028987695196 total t = 104941381885647.4 j = 94 Sigma[Inner] = 63.041697663673595
    dt= 3049815207.9848146 timestep= 13675 max Sigma= 1149.0540865416156 total t = 104955174164636.8 j = 93 Sigma[Inner] = 72.04125925538972
    dt= 4863682651.3927765 timestep= 13685 max Sigma= 1149.1466393834853 total t = 104998404335283.45 j = 93 Sigma[Inner] = 63.61584123453232
    dt= 5555849290.023259 timestep= 13695 max Sigma= 1149.2622548368067 total t = 105050892883204.72 j = 93 Sigma[Inner] = 55.565494351797014
    dt= 11145184325.20849 timestep= 13705 max Sigma= 1149.433692749507 total t = 105133407899285.61 j = 92 Sigma[Inner] = 24.524516800918214
    dt= 11156799123.837498 timestep= 13715 max Sigma= 1149.681621030277 total t = 105244923754688.02 j = 92 Sigma[Inner] = 26.75968265107027
    dt= 11168246085.315224 timestep= 13725 max Sigma= 1149.9291238913916 total t = 105356554841093.42 j = 92 Sigma[Inner] = 30.40544377042568
    dt= 11179532346.215834 timestep= 13735 max Sigma= 1150.1762026946785 total t = 105468299504782.73 j = 92 Sigma[Inner] = 36.875772630609546
    dt= 11190667659.570496 timestep= 13745 max Sigma= 1150.42285899363 total t = 105580156193230.86 j = 92 Sigma[Inner] = 47.18590058238726
    dt= 11201660856.538033 timestep= 13755 max Sigma= 1150.6690944946095 total t = 105692123446223.36 j = 92 Sigma[Inner] = 61.17497526445599
    dt= 11212519998.528738 timestep= 13765 max Sigma= 1150.9149110129122 total t = 105804199887519.08 j = 92 Sigma[Inner] = 77.58255015218576
    dt= 11223252486.619322 timestep= 13775 max Sigma= 1151.1603104335975 total t = 105916384217781.58 j = 92 Sigma[Inner] = 94.73192210372082
    dt= 11233865144.485954 timestep= 13785 max Sigma= 1151.4052946763206 total t = 106028675208449.19 j = 92 Sigma[Inner] = 111.1997265905498
    dt= 11244364284.096842 timestep= 13795 max Sigma= 1151.6498656636568 total t = 106141071696333.98 j = 92 Sigma[Inner] = 126.11354417436868
    dt= 11254755759.589783 timestep= 13805 max Sigma= 1151.8940252925788 total t = 106253572578809.05 j = 92 Sigma[Inner] = 139.1188882265995
    dt= 11265045012.636002 timestep= 13815 max Sigma= 1152.137775408818 total t = 106366176809485.77 j = 92 Sigma[Inner] = 150.20821022291798
    dt= 11275237111.38184 timestep= 13825 max Sigma= 1152.3811177839088 total t = 106478883394307.97 j = 92 Sigma[Inner] = 159.5532557837654
    dt= 11285336784.352596 timestep= 13835 max Sigma= 1152.6240540947297 total t = 106591691388006.89 j = 92 Sigma[Inner] = 167.3913869129546
    dt= 11295348450.276392 timestep= 13845 max Sigma= 1152.8665859053876 total t = 106704599890871.95 j = 92 Sigma[Inner] = 173.9637451646794
    dt= 11305276244.519913 timestep= 13855 max Sigma= 1153.108714651294 total t = 106817608045800.84 j = 92 Sigma[Inner] = 179.48781754867724
    dt= 11315124042.655903 timestep= 13865 max Sigma= 1153.350441625291 total t = 106930715035597.75 j = 92 Sigma[Inner] = 184.14880948879866
    dt= 11324895481.566568 timestep= 13875 max Sigma= 1153.591767965703 total t = 107043920080493.22 j = 92 Sigma[Inner] = 188.0998722274938
    dt= 11334593978.406633 timestep= 13885 max Sigma= 1153.8326946461773 total t = 107157222435863.25 j = 92 Sigma[Inner] = 191.4658106827204
    dt= 11345181907.700504 timestep= 13896 max Sigma= 1154.097253337506 total t = 107281966572035.31 j = 92 Sigma[Inner] = 194.61254464932637
    dt= 11354737466.438362 timestep= 13906 max Sigma= 1154.3373431225682 total t = 107395471000275.88 j = 92 Sigma[Inner] = 197.05571131041842
    dt= 11364229455.39799 timestep= 13916 max Sigma= 1154.5770351364602 total t = 107509070632196.66 j = 92 Sigma[Inner] = 199.16859372100987
    dt= 11373660552.708021 timestep= 13926 max Sigma= 1154.8163296050643 total t = 107622764846956.16 j = 92 Sigma[Inner] = 201.00385023465373
    dt= 11383033288.559942 timestep= 13936 max Sigma= 1155.0552265613387 total t = 107736553049671.6 j = 92 Sigma[Inner] = 202.60445754551108
    dt= 11392350055.604372 timestep= 13946 max Sigma= 1155.2937258436077 total t = 107850434669997.38 j = 92 Sigma[Inner] = 204.0056614815629
    dt= 11401613118.440672 timestep= 13956 max Sigma= 1155.531827094878 total t = 107964409160802.62 j = 92 Sigma[Inner] = 205.2365167348418
    dt= 11410824622.297508 timestep= 13966 max Sigma= 1155.7695297630783 total t = 108078475996939.19 j = 92 Sigma[Inner] = 206.32110059153385
    dt= 11419986600.990984 timestep= 13976 max Sigma= 1156.0068331021457 total t = 108192634674092.0 j = 92 Sigma[Inner] = 207.27947074332792
    dt= 11429100984.236792 timestep= 13986 max Sigma= 1156.2437361738741 total t = 108306884707705.9 j = 92 Sigma[Inner] = 208.12842311655643
    dt= 557938100.0517305 timestep= 13996 max Sigma= 1156.4193069518649 total t = 108380862853559.8 j = 93 Sigma[Inner] = 130.8659286393301
    dt= 1226812652.3757162 timestep= 14006 max Sigma= 1156.432759357038 total t = 108388040166589.8 j = 94 Sigma[Inner] = 85.80286542279997
    


    
![png](output_5_113.png)
    



    
![png](output_5_114.png)
    



    
![png](output_5_115.png)
    


    dt= 485314012.5848441 timestep= 14016 max Sigma= 1156.4805362939576 total t = 108410420412690.33 j = 94 Sigma[Inner] = 61.66925561554002
    dt= 3576160212.9809484 timestep= 14026 max Sigma= 1156.5112612777707 total t = 108428387137798.52 j = 93 Sigma[Inner] = 72.33269668927875
    dt= 4999607780.4137125 timestep= 14036 max Sigma= 1156.602092129172 total t = 108473814569176.2 j = 93 Sigma[Inner] = 62.18102068285469
    dt= 5785357158.103635 timestep= 14046 max Sigma= 1156.7119099210938 total t = 108527858719590.38 j = 93 Sigma[Inner] = 54.338152643544674
    dt= 11454433358.484444 timestep= 14056 max Sigma= 1156.9066225202628 total t = 108628105471513.73 j = 92 Sigma[Inner] = 24.8619263611868
    dt= 11463386429.758137 timestep= 14066 max Sigma= 1157.1419894786732 total t = 108742699081478.03 j = 92 Sigma[Inner] = 27.676100572346087
    dt= 11472298589.064957 timestep= 14076 max Sigma= 1157.3769491333221 total t = 108857381995578.25 j = 92 Sigma[Inner] = 32.385510990129866
    dt= 11481171764.163795 timestep= 14086 max Sigma= 1157.611499474615 total t = 108972153815360.06 j = 92 Sigma[Inner] = 40.48065325741174
    dt= 11490007650.466583 timestep= 14096 max Sigma= 1157.8456383369628 total t = 109087014160495.08 j = 92 Sigma[Inner] = 52.58006093905198
    dt= 11498807753.49978 timestep= 14106 max Sigma= 1158.0793634007741 total t = 109201962666504.48 j = 92 Sigma[Inner] = 67.96384228218895
    dt= 11507573445.405266 timestep= 14116 max Sigma= 1158.3126721960539 total t = 109316998983196.05 j = 92 Sigma[Inner] = 85.01631249074137
    dt= 11516305998.485378 timestep= 14126 max Sigma= 1158.5455621070628 total t = 109432122773528.61 j = 92 Sigma[Inner] = 102.04397949641212
    dt= 11525006605.864693 timestep= 14136 max Sigma= 1158.7780303776165 total t = 109547333712735.1 j = 92 Sigma[Inner] = 117.83219927554325
    dt= 11533676394.781542 timestep= 14146 max Sigma= 1159.010074116789 total t = 109662631487608.42 j = 92 Sigma[Inner] = 131.7687662093659
    dt= 11542316435.569546 timestep= 14156 max Sigma= 1159.241690304861 total t = 109778015795894.84 j = 92 Sigma[Inner] = 143.7050600191004
    dt= 11550927748.05338 timestep= 14166 max Sigma= 1159.4728757994296 total t = 109893486345761.64 j = 92 Sigma[Inner] = 153.75913719665425
    dt= 11559511306.346079 timestep= 14176 max Sigma= 1159.703627341592 total t = 110009042855319.38 j = 92 Sigma[Inner] = 162.16353495627249
    dt= 11568068042.622557 timestep= 14186 max Sigma= 1159.933941562171 total t = 110124685052186.36 j = 92 Sigma[Inner] = 169.17586093961233
    dt= 11576598850.209846 timestep= 14196 max Sigma= 1160.1638149879404 total t = 110240412673086.61 j = 92 Sigma[Inner] = 175.0361711281256
    dt= 11585104586.199715 timestep= 14206 max Sigma= 1160.39324404782 total t = 110356225463476.2 j = 92 Sigma[Inner] = 179.95144975529496
    dt= 11593586073.710386 timestep= 14216 max Sigma= 1160.6222250790147 total t = 110472123177193.44 j = 92 Sigma[Inner] = 184.09338202765883
    dt= 11602044103.877844 timestep= 14226 max Sigma= 1160.8507543330852 total t = 110588105576130.44 j = 92 Sigma[Inner] = 187.60164261028987
    dt= 11610479437.629242 timestep= 14236 max Sigma= 1161.0788279819253 total t = 110704172429923.1 j = 92 Sigma[Inner] = 190.5888716019485
    dt= 11618892807.274908 timestep= 14246 max Sigma= 1161.3064421236343 total t = 110820323515657.95 j = 92 Sigma[Inner] = 193.14566275456605
    dt= 11627284917.945704 timestep= 14256 max Sigma= 1161.5335927882752 total t = 110936558617594.03 j = 92 Sigma[Inner] = 195.34494256363706
    dt= 11635656448.897066 timestep= 14266 max Sigma= 1161.7602759435072 total t = 111052877526898.31 j = 92 Sigma[Inner] = 197.24559198489638
    dt= 11644008054.698124 timestep= 14276 max Sigma= 1161.986487500079 total t = 111169280041393.7 j = 92 Sigma[Inner] = 198.8953538297444
    dt= 11652340366.321991 timestep= 14286 max Sigma= 1162.2122233171826 total t = 111285765965318.02 j = 92 Sigma[Inner] = 200.33313192072953
    dt= 11660653992.15259 timestep= 14296 max Sigma= 1162.4374792076583 total t = 111402335109093.84 j = 92 Sigma[Inner] = 201.59079672538786
    dt= 11668949518.921892 timestep= 14306 max Sigma= 1162.662250943045 total t = 111518987289107.77 j = 92 Sigma[Inner] = 202.69460032689202
    dt= 11677227512.590252 timestep= 14316 max Sigma= 1162.8865342584759 total t = 111635722327498.78 j = 92 Sigma[Inner] = 203.6662860400521
    dt= 11685488519.181368 timestep= 14326 max Sigma= 1163.110324857413 total t = 111752540051955.12 j = 92 Sigma[Inner] = 204.52396074342136
    dt= 11693733065.581472 timestep= 14336 max Sigma= 1163.3336184162256 total t = 111869440295519.31 j = 92 Sigma[Inner] = 205.28278314997038
    dt= 11701961660.31084 timestep= 14346 max Sigma= 1163.5564105886015 total t = 111986422896401.14 j = 92 Sigma[Inner] = 205.95550919255496
    dt= 11710174794.27377 timestep= 14356 max Sigma= 1163.778697009804 total t = 112103487697798.16 j = 92 Sigma[Inner] = 206.5529262261304
    dt= 11718372941.49155 timestep= 14366 max Sigma= 1164.0004733007615 total t = 112220634547723.39 j = 92 Sigma[Inner] = 207.08420042085694
    dt= 11726556559.821089 timestep= 14376 max Sigma= 1164.2217350720036 total t = 112337863298840.67 j = 92 Sigma[Inner] = 207.55715610207125
    dt= 11734726091.660612 timestep= 14386 max Sigma= 1164.442477927435 total t = 112455173808306.78 j = 92 Sigma[Inner] = 207.9785015028714
    dt= 11742881964.642305 timestep= 14396 max Sigma= 1164.6626974679605 total t = 112572565937620.67 j = 92 Sigma[Inner] = 208.35401212218508
    dt= 558245975.2542006 timestep= 14406 max Sigma= 1164.8249565071023 total t = 112648103798458.16 j = 93 Sigma[Inner] = 129.11289012372504
    dt= 1251701317.8909478 timestep= 14416 max Sigma= 1164.8372409691249 total t = 112655371940572.8 j = 94 Sigma[Inner] = 85.09584298569257
    dt= 484042169.31041604 timestep= 14426 max Sigma= 1164.8812949819162 total t = 112678190584167.25 j = 94 Sigma[Inner] = 61.13004696280494
    dt= 3576794358.5312824 timestep= 14436 max Sigma= 1164.9090534914185 total t = 112696153057420.58 j = 93 Sigma[Inner] = 72.29894949624894
    dt= 5030930426.278087 timestep= 14446 max Sigma= 1164.9914331831792 total t = 112741769684417.34 j = 93 Sigma[Inner] = 62.04472918435062
    dt= 6304235977.486656 timestep= 14456 max Sigma= 1165.091747435519 total t = 112796889496989.61 j = 93 Sigma[Inner] = 53.998562494402485
    dt= 11765820039.008385 timestep= 14466 max Sigma= 1165.2813855930892 total t = 112904349898690.84 j = 92 Sigma[Inner] = 24.79140931790759
    dt= 11773930725.757408 timestep= 14476 max Sigma= 1165.4995639252643 total t = 113022052718924.08 j = 92 Sigma[Inner] = 27.949950182442695
    dt= 11782028443.455198 timestep= 14486 max Sigma= 1165.7171978443914 total t = 113139836573962.58 j = 92 Sigma[Inner] = 33.27379141499012
    dt= 11790113998.955002 timestep= 14496 max Sigma= 1165.9342829798768 total t = 113257701338707.73 j = 92 Sigma[Inner] = 42.23973601520565
    dt= 11798188011.931639 timestep= 14506 max Sigma= 1166.150814985883 total t = 113375646895071.16 j = 92 Sigma[Inner] = 55.231938008537135
    


    
![png](output_5_117.png)
    



    
![png](output_5_118.png)
    



    
![png](output_5_119.png)
    


    dt= 11806250982.388662 timestep= 14516 max Sigma= 1166.3667895410035 total t = 113493673130453.45 j = 92 Sigma[Inner] = 71.23624503166123
    dt= 11814303336.7748 timestep= 14526 max Sigma= 1166.5822023489836 total t = 113611779936822.55 j = 92 Sigma[Inner] = 88.47680328639889
    dt= 11822345454.43125 timestep= 14536 max Sigma= 1166.7970491399942 total t = 113729967210135.34 j = 92 Sigma[Inner] = 105.28261476340764
    dt= 11830377683.066269 timestep= 14546 max Sigma= 1167.0113256721738 total t = 113848234849958.52 j = 92 Sigma[Inner] = 120.56764132504088
    dt= 11838400348.012201 timestep= 14556 max Sigma= 1167.2250277332748 total t = 113966582759206.9 j = 92 Sigma[Inner] = 133.8615893563568
    dt= 11846413757.900225 timestep= 14566 max Sigma= 1167.438151142336 total t = 114085010843953.16 j = 92 Sigma[Inner] = 145.12199601324917
    dt= 11854418208.23407 timestep= 14576 max Sigma= 1167.6506917513134 total t = 114203519013281.7 j = 92 Sigma[Inner] = 154.52935908757735
    dt= 11862413983.707031 timestep= 14586 max Sigma= 1167.8626454466603 total t = 114322107179170.95 j = 92 Sigma[Inner] = 162.3454323160871
    dt= 11870401359.750883 timestep= 14596 max Sigma= 1168.0740081508268 total t = 114440775256394.38 j = 92 Sigma[Inner] = 168.83685803437766
    dt= 11878380603.602945 timestep= 14606 max Sigma= 1168.2847758236817 total t = 114559523162433.89 j = 92 Sigma[Inner] = 174.24226003897442
    dt= 11886351975.061693 timestep= 14616 max Sigma= 1168.494944463843 total t = 114678350817402.33 j = 92 Sigma[Inner] = 178.7625994273494
    dt= 11894315727.033564 timestep= 14626 max Sigma= 1168.704510109927 total t = 114797258143972.25 j = 92 Sigma[Inner] = 182.56208148666983
    dt= 11902272105.933882 timestep= 14636 max Sigma= 1168.913468841707 total t = 114916245067309.33 j = 92 Sigma[Inner] = 185.77295102799644
    dt= 11910221351.981558 timestep= 14646 max Sigma= 1169.121816781188 total t = 115035311515009.02 j = 92 Sigma[Inner] = 188.5011032503972
    dt= 11918163699.41385 timestep= 14656 max Sigma= 1169.329550093598 total t = 115154457417035.97 j = 92 Sigma[Inner] = 190.8312723958198
    dt= 11926099376.639597 timestep= 14666 max Sigma= 1169.5366649882978 total t = 115273682705665.06 j = 92 Sigma[Inner] = 192.83141018885777
    dt= 11934028606.345428 timestep= 14676 max Sigma= 1169.7431577196132 total t = 115392987315423.67 j = 92 Sigma[Inner] = 194.55622390297987
    dt= 11941951605.567348 timestep= 14686 max Sigma= 1169.9490245875913 total t = 115512371183034.94 j = 92 Sigma[Inner] = 196.04997418445072
    dt= 11949868585.738579 timestep= 14696 max Sigma= 1170.1542619386826 total t = 115631834247361.48 j = 92 Sigma[Inner] = 197.34866471774347
    dt= 11957779752.724247 timestep= 14706 max Sigma= 1170.3588661663537 total t = 115751376449349.69 j = 92 Sigma[Inner] = 198.48174907997867
    dt= 11965685306.852257 timestep= 14716 max Sigma= 1170.5628337116336 total t = 115870997731974.06 j = 92 Sigma[Inner] = 199.4734609178414
    dt= 11973585442.949059 timestep= 14726 max Sigma= 1170.7661610635957 total t = 115990698040182.08 j = 92 Sigma[Inner] = 200.34385269701934
    dt= 11981480350.387552 timestep= 14736 max Sigma= 1170.9688447597778 total t = 116110477320839.4 j = 92 Sigma[Inner] = 201.10960968065655
    dt= 11989370213.15299 timestep= 14746 max Sigma= 1171.1708813865464 total t = 116230335522675.84 j = 92 Sigma[Inner] = 201.78469053421293
    dt= 11997255209.931034 timestep= 14756 max Sigma= 1171.3722675794043 total t = 116350272596231.75 j = 92 Sigma[Inner] = 202.3808339251155
    dt= 12005135514.22056 timestep= 14766 max Sigma= 1171.573000023248 total t = 116470288493805.72 j = 92 Sigma[Inner] = 202.9079612019744
    dt= 12013011294.472002 timestep= 14776 max Sigma= 1171.7730754525776 total t = 116590383169403.44 j = 92 Sigma[Inner] = 203.37449815246714
    dt= 12020882714.250542 timestep= 14786 max Sigma= 1171.9724906516587 total t = 116710556578688.11 j = 92 Sigma[Inner] = 203.78763345917977
    dt= 12028749932.422308 timestep= 14796 max Sigma= 1172.171242454643 total t = 116830808678932.56 j = 92 Sigma[Inner] = 204.15352739473533
    dt= 12036613103.360348 timestep= 14806 max Sigma= 1172.3693277456493 total t = 116951139428973.36 j = 92 Sigma[Inner] = 204.47748120438723
    dt= 12044472377.166836 timestep= 14816 max Sigma= 1172.5667434588047 total t = 117071548789167.19 j = 92 Sigma[Inner] = 204.76407527302476
    dt= 12052327899.907022 timestep= 14826 max Sigma= 1172.763486578253 total t = 117192036721349.3 j = 92 Sigma[Inner] = 205.01728238067068
    dt= 12060179813.850534 timestep= 14836 max Sigma= 1172.9595541381293 total t = 117312603188794.56 j = 92 Sigma[Inner] = 205.24056097826437
    dt= 12068028257.7154 timestep= 14846 max Sigma= 1173.1549432225022 total t = 117433248156180.78 j = 92 Sigma[Inner] = 205.43693236074492
    dt= 12075873366.910612 timestep= 14856 max Sigma= 1173.3496509652891 total t = 117553971589554.56 j = 92 Sigma[Inner] = 205.6090448000943
    dt= 12083715273.773113 timestep= 14866 max Sigma= 1173.5436745501404 total t = 117674773456299.48 j = 92 Sigma[Inner] = 205.7592270693646
    dt= 12091554107.795794 timestep= 14876 max Sigma= 1173.7370112103017 total t = 117795653725106.73 j = 92 Sigma[Inner] = 205.88953329651645
    dt= 12099389995.84352 timestep= 14886 max Sigma= 1173.9296582284496 total t = 117916612365947.7 j = 92 Sigma[Inner] = 206.00178070154197
    dt= 12107223062.354692 timestep= 14896 max Sigma= 1174.1216129365034 total t = 118037649350048.98 j = 92 Sigma[Inner] = 206.09758146725105
    dt= 12115053429.526749 timestep= 14906 max Sigma= 1174.3128727154176 total t = 118158764649869.1 j = 92 Sigma[Inner] = 206.17836975459218
    dt= 12122881217.484053 timestep= 14916 max Sigma= 1174.5034349949515 total t = 118279958239076.92 j = 92 Sigma[Inner] = 206.24542468327434
    dt= 12130706544.427708 timestep= 14926 max Sigma= 1174.6932972534191 total t = 118401230092532.0 j = 92 Sigma[Inner] = 206.29988994688637
    dt= 12138529526.766996 timestep= 14936 max Sigma= 1174.882457017422 total t = 118522580186265.86 j = 92 Sigma[Inner] = 206.34279061034553
    dt= 12146350279.232439 timestep= 14946 max Sigma= 1175.0709118615653 total t = 118644008497464.84 j = 92 Sigma[Inner] = 206.37504753991504
    dt= 12154168914.971348 timestep= 14956 max Sigma= 1175.2586594081529 total t = 118765515004453.75 j = 92 Sigma[Inner] = 206.39748983723197
    dt= 12161985545.62643 timestep= 14966 max Sigma= 1175.4456973268718 total t = 118887099686680.6 j = 92 Sigma[Inner] = 206.41086558491278
    dt= 12169800281.39867 timestep= 14976 max Sigma= 1175.63202333446 total t = 119008762524701.83 j = 92 Sigma[Inner] = 206.415851159317
    dt= 12177613231.095581 timestep= 14986 max Sigma= 1175.8176351943573 total t = 119130503500168.14 j = 92 Sigma[Inner] = 206.41305932358867
    dt= 12185424502.166431 timestep= 14996 max Sigma= 1176.0025307163492 total t = 119252322595810.89 j = 92 Sigma[Inner] = 206.40304627928185
    dt= 12193234200.725616 timestep= 15006 max Sigma= 1176.1867077561935 total t = 119374219795428.52 j = 92 Sigma[Inner] = 206.38631782622937
    


    
![png](output_5_121.png)
    



    
![png](output_5_122.png)
    



    
![png](output_5_123.png)
    


    dt= 12201042431.565762 timestep= 15016 max Sigma= 1176.3701642152364 total t = 119496195083873.36 j = 92 Sigma[Inner] = 206.36333475666984
    dt= 12208849298.162092 timestep= 15026 max Sigma= 1176.55289804002 total t = 119618248447038.45 j = 92 Sigma[Inner] = 206.33451759005362
    dt= 12216654902.669252 timestep= 15036 max Sigma= 1176.7349072218772 total t = 119740379871844.28 j = 92 Sigma[Inner] = 206.30025073867202
    dt= 12224459345.912127 timestep= 15046 max Sigma= 1176.9161897965198 total t = 119862589346225.6 j = 92 Sigma[Inner] = 206.26088618068033
    dt= 10885104986.57264 timestep= 15056 max Sigma= 1177.0947600139298 total t = 119982183323929.62 j = 92 Sigma[Inner] = 206.21725527037705
    dt= 10885104986.57264 timestep= 15066 max Sigma= 1177.2547791546963 total t = 120091034373795.4 j = 92 Sigma[Inner] = 206.17423162314023
    dt= 10885104986.57264 timestep= 15076 max Sigma= 1177.414128683446 total t = 120199885423661.19 j = 92 Sigma[Inner] = 206.12788406980508
    dt= 10885104986.57264 timestep= 15086 max Sigma= 1177.5728084874365 total t = 120308736473526.97 j = 92 Sigma[Inner] = 206.0783969234019
    dt= 10885104986.57264 timestep= 15096 max Sigma= 1177.7308184996718 total t = 120417587523392.75 j = 92 Sigma[Inner] = 206.02594152069935
    dt= 10885104986.57264 timestep= 15106 max Sigma= 1177.8881587324236 total t = 120526438573258.53 j = 92 Sigma[Inner] = 205.97067730813743
    dt= 11039622879.85752 timestep= 15116 max Sigma= 1178.0457032834763 total t = 120636052519768.36 j = 92 Sigma[Inner] = 205.91242262894355
    dt= 11180750982.382656 timestep= 15126 max Sigma= 1178.2048647341787 total t = 120747256731175.48 j = 92 Sigma[Inner] = 205.8507114525157
    dt= 11289260353.740168 timestep= 15136 max Sigma= 1178.3651174095119 total t = 120859683515127.4 j = 92 Sigma[Inner] = 205.7857748169805
    dt= 11374664831.898558 timestep= 15146 max Sigma= 1178.5260408239978 total t = 120973061809828.64 j = 92 Sigma[Inner] = 205.71783267088483
    dt= 11443576375.846483 timestep= 15156 max Sigma= 1178.6873367354108 total t = 121087198939614.7 j = 92 Sigma[Inner] = 205.64707530995398
    dt= 11500563737.83287 timestep= 15166 max Sigma= 1178.8487916674883 total t = 121201956492130.19 j = 92 Sigma[Inner] = 205.57366924916101
    dt= 11548793231.003435 timestep= 15176 max Sigma= 1179.0102509629535 total t = 121317233586716.9 j = 92 Sigma[Inner] = 205.49776202917482
    dt= 11590477495.80002 timestep= 15186 max Sigma= 1179.1716011232058 total t = 121432955456984.36 j = 92 Sigma[Inner] = 205.41948593381278
    dt= 11627180674.074417 timestep= 15196 max Sigma= 1179.3327578375195 total t = 121549065691505.89 j = 92 Sigma[Inner] = 205.3389608009942
    dt= 11660023479.854105 timestep= 15206 max Sigma= 1179.4936578377615 total t = 121665520944202.27 j = 92 Sigma[Inner] = 205.2562961385112
    dt= 11689820718.230766 timestep= 15216 max Sigma= 1179.6542533060233 total t = 121782287301503.86 j = 92 Sigma[Inner] = 205.1715927260452
    dt= 11717173845.489824 timestep= 15226 max Sigma= 1179.8145079861276 total t = 121899337762249.73 j = 92 Sigma[Inner] = 205.08494384331527
    dt= 11742533714.449202 timestep= 15236 max Sigma= 1179.9743944377715 total t = 122016650469142.17 j = 92 Sigma[Inner] = 204.99643622707723
    dt= 11766243501.39732 timestep= 15246 max Sigma= 1180.1338920628123 total t = 122134207452036.48 j = 92 Sigma[Inner] = 204.90615083068113
    dt= 11788568381.690296 timestep= 15256 max Sigma= 1180.2929856584312 total t = 122251993723271.28 j = 92 Sigma[Inner] = 204.81416343858936
    dt= 11809716273.80957 timestep= 15266 max Sigma= 1180.4516643339023 total t = 122369996617765.78 j = 92 Sigma[Inner] = 204.7205451730538
    dt= 11829852507.46238 timestep= 15276 max Sigma= 1180.6099206815536 total t = 122488205305239.1 j = 92 Sigma[Inner] = 204.62536291946293
    dt= 11849110316.858263 timestep= 15286 max Sigma= 1180.7677501280389 total t = 122606610424877.58 j = 92 Sigma[Inner] = 204.528679689396
    dt= 11867598435.555742 timestep= 15296 max Sigma= 1180.925150415625 total t = 122725203808136.06 j = 92 Sigma[Inner] = 204.43055493520305
    dt= 11885406657.728065 timestep= 15306 max Sigma= 1181.0821211789967 total t = 122843978265718.62 j = 92 Sigma[Inner] = 204.33104482627087
    dt= 11902609957.483805 timestep= 15316 max Sigma= 1181.2386635936834 total t = 122962927421839.4 j = 92 Sigma[Inner] = 204.2302024945624
    dt= 11919271574.943699 timestep= 15326 max Sigma= 1181.3947800794183 total t = 123082045583715.14 j = 92 Sigma[Inner] = 204.12807825518536
    dt= 11935445354.182 timestep= 15336 max Sigma= 1181.550474046648 total t = 123201327637607.88 j = 92 Sigma[Inner] = 204.02471980643105
    dt= 11951177533.868914 timestep= 15346 max Sigma= 1181.7057496777627 total t = 123320768965098.27 j = 92 Sigma[Inner] = 203.92017241276125
    dt= 11966508133.456553 timestep= 15356 max Sigma= 1181.8606117369488 total t = 123440365374942.06 j = 92 Sigma[Inner] = 203.8144790734999
    dt= 11981472037.47236 timestep= 15366 max Sigma= 1182.0150654041515 total t = 123560113047056.22 j = 92 Sigma[Inner] = 203.7076806794326
    dt= 11996099852.253422 timestep= 15376 max Sigma= 1182.16911612976 total t = 123680008486044.64 j = 92 Sigma[Inner] = 203.59981615907503
    dt= 12010418589.492647 timestep= 15386 max Sigma= 1182.3227695073965 total t = 123800048482300.22 j = 92 Sigma[Inner] = 203.49092261601612
    dt= 12024452216.726177 timestep= 15396 max Sigma= 1182.4760311627401 total t = 123920230079182.1 j = 92 Sigma[Inner] = 203.38103545843308
    dt= 12038222104.644583 timestep= 15406 max Sigma= 1182.6289066566947 total t = 124040550545109.48 j = 92 Sigma[Inner] = 203.2701885216263
    dt= 12051747393.67472 timestep= 15416 max Sigma= 1182.7814014014891 total t = 124161007349668.02 j = 92 Sigma[Inner] = 203.15841418419453
    dt= 12065045296.839184 timestep= 15426 max Sigma= 1182.9335205884972 total t = 124281598143019.7 j = 92 Sigma[Inner] = 203.04574347828597
    dt= 12078131351.887428 timestep= 15436 max Sigma= 1183.0852691266996 total t = 124402320738052.9 j = 92 Sigma[Inner] = 202.93220619420003
    dt= 12091019632.70941 timestep= 15446 max Sigma= 1183.236651590831 total t = 124523173094822.42 j = 92 Sigma[Inner] = 202.8178309794843
    dt= 12103722927.806961 timestep= 15456 max Sigma= 1183.3876721783372 total t = 124644153306917.0 j = 92 Sigma[Inner] = 202.702645432571
    dt= 12116252891.909729 timestep= 15466 max Sigma= 1183.538334674334 total t = 124765259589459.27 j = 92 Sigma[Inner] = 202.58667619092006
    dt= 12128620175.538408 timestep= 15476 max Sigma= 1183.6886424238178 total t = 124886490268497.23 j = 92 Sigma[Inner] = 202.4699490135884
    dt= 12140834536.332764 timestep= 15486 max Sigma= 1183.8385983104326 total t = 125007843771588.0 j = 92 Sigma[Inner] = 202.3524888581198
    dt= 12152904935.201859 timestep= 15496 max Sigma= 1183.988204741137 total t = 125129318619409.39 j = 92 Sigma[Inner] = 202.23431995164367
    dt= 12164839619.762205 timestep= 15506 max Sigma= 1184.137463636168 total t = 125250913418261.02 j = 92 Sigma[Inner] = 202.11546585608826
    


    
![png](output_5_125.png)
    



    
![png](output_5_126.png)
    



    
![png](output_5_127.png)
    


    dt= 12176646197.066236 timestep= 15516 max Sigma= 1184.2863764237202 total t = 125372626853339.48 j = 92 Sigma[Inner] = 201.99594952744215
    dt= 12188331697.257957 timestep= 15526 max Sigma= 1184.4349440388205 total t = 125494457682689.84 j = 92 Sigma[Inner] = 201.87579336903656
    dt= 12199902629.502325 timestep= 15536 max Sigma= 1184.5831669258916 total t = 125616404731749.92 j = 92 Sigma[Inner] = 201.75501927887296
    dt= 12211365031.302738 timestep= 15546 max Sigma= 1184.731045044554 total t = 125738466888416.83 j = 92 Sigma[Inner] = 201.63364869107252
    dt= 12222724512.134422 timestep= 15556 max Sigma= 1184.8785778782367 total t = 125860643098574.69 j = 92 Sigma[Inner] = 201.5117026115795
    dt= 12233986292.170073 timestep= 15566 max Sigma= 1185.0257644452015 total t = 125982932362031.17 j = 92 Sigma[Inner] = 201.3892016483053
    dt= 12245155236.751104 timestep= 15576 max Sigma= 1185.172603311628 total t = 126105333728817.16 j = 92 Sigma[Inner] = 201.26616603595164
    dt= 12256235887.156765 timestep= 15586 max Sigma= 1185.3190926064221 total t = 126227846295810.69 j = 92 Sigma[Inner] = 201.14261565579775
    dt= 12267232488.140478 timestep= 15596 max Sigma= 1185.4652300374503 total t = 126350469203650.22 j = 92 Sigma[Inner] = 201.018570050776
    dt= 12278149012.633484 timestep= 15606 max Sigma= 1185.6110129089273 total t = 126473201633907.38 j = 92 Sigma[Inner] = 200.89404843619567
    dt= 12288989183.959013 timestep= 15616 max Sigma= 1185.7564381397099 total t = 126596042806492.86 j = 92 Sigma[Inner] = 200.76906970649733
    dt= 12299756495.85166 timestep= 15626 max Sigma= 1185.901502282275 total t = 126718991977272.23 j = 92 Sigma[Inner] = 200.6436524384413
    dt= 12310454230.536833 timestep= 15636 max Sigma= 1186.0462015421765 total t = 126842048435870.89 j = 92 Sigma[Inner] = 200.51781489114208
    dt= 12321085475.090658 timestep= 15646 max Sigma= 1186.1905317978171 total t = 126965211503650.38 j = 92 Sigma[Inner] = 200.3915750033635
    dt= 12331653136.2722 timestep= 15656 max Sigma= 1186.3344886203645 total t = 127088480531839.53 j = 92 Sigma[Inner] = 200.26495038848722
    dt= 12342159953.994892 timestep= 15666 max Sigma= 1186.478067293682 total t = 127211854899806.62 j = 92 Sigma[Inner] = 200.13795832755721
    dt= 12352608513.583492 timestep= 15676 max Sigma= 1186.6212628341464 total t = 127335334013459.08 j = 92 Sigma[Inner] = 200.01061576078635
    dt= 12363001256.944502 timestep= 15686 max Sigma= 1186.7640700102536 total t = 127458917303760.0 j = 92 Sigma[Inner] = 199.88293927789582
    dt= 12373340492.76271 timestep= 15696 max Sigma= 1186.9064833619154 total t = 127582604225350.89 j = 92 Sigma[Inner] = 199.7549451076316
    dt= 12383628405.823076 timestep= 15706 max Sigma= 1187.0484972193756 total t = 127706394255271.11 j = 92 Sigma[Inner] = 199.62664910677947
    dt= 12393867065.545622 timestep= 15716 max Sigma= 1187.1901057216721 total t = 127830286891766.81 j = 92 Sigma[Inner] = 199.49806674897172
    dt= 12404058433.810799 timestep= 15726 max Sigma= 1187.331302834601 total t = 127954281653180.64 j = 92 Sigma[Inner] = 199.36921311355053
    dt= 12414204372.144321 timestep= 15736 max Sigma= 1187.4720823681282 total t = 128078378076916.34 j = 92 Sigma[Inner] = 199.24010287472402
    dt= 12424306648.322613 timestep= 15746 max Sigma= 1187.6124379932187 total t = 128202575718472.19 j = 92 Sigma[Inner] = 199.11075029122088
    dt= 12434366942.45329 timestep= 15756 max Sigma= 1187.7523632580535 total t = 128326874150537.1 j = 92 Sigma[Inner] = 198.98116919662255
    dt= 12444386852.579424 timestep= 15766 max Sigma= 1187.8918516036147 total t = 128451272962145.39 j = 92 Sigma[Inner] = 198.85137299052192
    dt= 12454367899.851084 timestep= 15776 max Sigma= 1188.0308963786222 total t = 128575771757884.73 j = 92 Sigma[Inner] = 198.7213746306334
    dt= 12464311533.303041 timestep= 15786 max Sigma= 1188.1694908538163 total t = 128700370157153.9 j = 92 Sigma[Inner] = 198.5911866259517
    dt= 12474219134.273657 timestep= 15796 max Sigma= 1188.3076282355805 total t = 128825067793466.19 j = 92 Sigma[Inner] = 198.4608210310341
    dt= 12484092020.496304 timestep= 15806 max Sigma= 1188.4453016789012 total t = 128949864313795.14 j = 92 Sigma[Inner] = 198.33028944146082
    dt= 12493931449.891659 timestep= 15816 max Sigma= 1188.5825042996803 total t = 129074759377959.73 j = 92 Sigma[Inner] = 198.19960299050493
    dt= 12503738624.086262 timestep= 15826 max Sigma= 1188.7192291863926 total t = 129199752658045.69 j = 92 Sigma[Inner] = 198.0687723470294
    dt= 12513514691.680418 timestep= 15836 max Sigma= 1188.8554694111047 total t = 129324843837860.77 j = 92 Sigma[Inner] = 197.93780771460877
    dt= 12523260751.286144 timestep= 15846 max Sigma= 1188.9912180398778 total t = 129450032612421.33 j = 92 Sigma[Inner] = 197.8067188318627
    dt= 12532977854.354134 timestep= 15856 max Sigma= 1189.126468142547 total t = 129575318687468.31 j = 92 Sigma[Inner] = 197.67551497397397
    dt= 12542667007.80663 timestep= 15866 max Sigma= 1189.2612128019143 total t = 129700701779010.08 j = 92 Sigma[Inner] = 197.54420495535365
    dt= 12552329176.491884 timestep= 15876 max Sigma= 1189.395445122357 total t = 129826181612890.89 j = 92 Sigma[Inner] = 197.41279713340793
    dt= 12561965285.474007 timestep= 15886 max Sigma= 1189.5291582378818 total t = 129951757924382.88 j = 92 Sigma[Inner] = 197.28129941335337
    dt= 12571576222.171402 timestep= 15896 max Sigma= 1189.6623453196319 total t = 130077430457799.97 j = 92 Sigma[Inner] = 197.14971925402097
    dt= 12581162838.355074 timestep= 15906 max Sigma= 1189.7949995828762 total t = 130203198966132.73 j = 92 Sigma[Inner] = 197.01806367458693
    dt= 12590725952.017744 timestep= 15916 max Sigma= 1189.927114293498 total t = 130329063210702.08 j = 92 Sigma[Inner] = 196.88633926216164
    dt= 12600266349.123312 timestep= 15926 max Sigma= 1190.0586827739978 total t = 130455022960831.33 j = 92 Sigma[Inner] = 196.7545521801696
    dt= 12609784785.245678 timestep= 15936 max Sigma= 1190.1896984090367 total t = 130581077993535.02 j = 92 Sigma[Inner] = 196.62270817744863
    dt= 12619281987.10497 timestep= 15946 max Sigma= 1190.3201546505397 total t = 130707228093223.47 j = 92 Sigma[Inner] = 196.49081259799902
    dt= 12628758654.008757 timestep= 15956 max Sigma= 1190.4500450223786 total t = 130833473051422.28 j = 92 Sigma[Inner] = 196.3588703913114
    dt= 12638215459.204865 timestep= 15966 max Sigma= 1190.5793631246493 total t = 130959812666505.72 j = 92 Sigma[Inner] = 196.2268861232052
    dt= 12647653051.152351 timestep= 15976 max Sigma= 1190.7081026375715 total t = 131086246743443.14 j = 92 Sigma[Inner] = 196.09486398710837
    dt= 12657072054.716208 timestep= 15986 max Sigma= 1190.8362573250233 total t = 131212775093557.88 j = 92 Sigma[Inner] = 195.96280781571463
    dt= 12666473072.291185 timestep= 15996 max Sigma= 1190.963821037732 total t = 131339397534297.33 j = 92 Sigma[Inner] = 195.83072109295318
    dt= 12675856684.859604 timestep= 16006 max Sigma= 1191.0907877161396 total t = 131466113889014.25 j = 92 Sigma[Inner] = 195.69860696621097
    


    
![png](output_5_129.png)
    



    
![png](output_5_130.png)
    



    
![png](output_5_131.png)
    


    dt= 12685223452.987658 timestep= 16016 max Sigma= 1191.2171513929586 total t = 131592923986758.14 j = 92 Sigma[Inner] = 195.56646825875043
    dt= 12694573917.764387 timestep= 16026 max Sigma= 1191.342906195437 total t = 131719827662076.3 j = 92 Sigma[Inner] = 195.43430748226635
    dt= 12703908601.687143 timestep= 16036 max Sigma= 1191.4680463473478 total t = 131846824754823.86 j = 92 Sigma[Inner] = 195.3021268495323
    dt= 12713228009.497093 timestep= 16046 max Sigma= 1191.5925661707192 total t = 131973915109982.75 j = 92 Sigma[Inner] = 195.1699282870868
    dt= 12722532628.968042 timestep= 16056 max Sigma= 1191.7164600873232 total t = 132101098577488.28 j = 92 Sigma[Inner] = 195.03771344791568
    dt= 12731822931.651611 timestep= 16066 max Sigma= 1191.8397226199327 total t = 132228375012063.64 j = 92 Sigma[Inner] = 194.90548372408853
    dt= 12741099373.581642 timestep= 16076 max Sigma= 1191.9623483933644 total t = 132355744273061.72 j = 92 Sigma[Inner] = 194.7732402593108
    dt= 12750362395.940264 timestep= 16086 max Sigma= 1192.084332135322 total t = 132483206224313.42 j = 92 Sigma[Inner] = 194.64098396135722
    dt= 12759612425.688293 timestep= 16096 max Sigma= 1192.2056686770463 total t = 132610760733982.83 j = 92 Sigma[Inner] = 194.50871551435418
    dt= 12768849876.162012 timestep= 16106 max Sigma= 1192.3263529537971 total t = 132738407674428.14 j = 92 Sigma[Inner] = 194.37643539088262
    dt= 12778075147.638468 timestep= 16116 max Sigma= 1192.446380005162 total t = 132866146922068.77 j = 92 Sigma[Inner] = 194.24414386387565
    dt= 12787288627.871277 timestep= 16126 max Sigma= 1192.5657449752186 total t = 132993978357257.45 j = 92 Sigma[Inner] = 194.11184101828763
    dt= 12796490692.598753 timestep= 16136 max Sigma= 1192.6844431125526 total t = 133121901864157.98 j = 92 Sigma[Inner] = 193.97952676251512
    dt= 12805681706.025908 timestep= 16146 max Sigma= 1192.802469770141 total t = 133249917330627.86 j = 92 Sigma[Inner] = 193.84720083955108
    dt= 12814862021.28204 timestep= 16156 max Sigma= 1192.9198204051165 total t = 133378024648105.19 j = 92 Sigma[Inner] = 193.71486283785808
    dt= 12824031980.855291 timestep= 16166 max Sigma= 1193.0364905784172 total t = 133506223711500.53 j = 92 Sigma[Inner] = 193.58251220194705
    dt= 12833191917.005587 timestep= 16176 max Sigma= 1193.1524759543304 total t = 133634514419092.77 j = 92 Sigma[Inner] = 193.4501482426505
    dt= 12842342152.157175 timestep= 16186 max Sigma= 1193.2677722999422 total t = 133762896672428.97 j = 92 Sigma[Inner] = 193.3177701470824
    dt= 12851482999.27198 timestep= 16196 max Sigma= 1193.3823754844957 total t = 133891370376228.08 j = 92 Sigma[Inner] = 193.18537698827683
    dt= 12860614762.204855 timestep= 16206 max Sigma= 1193.496281478669 total t = 134019935438288.34 j = 92 Sigma[Inner] = 193.0529677345
    dt= 12869737736.041851 timestep= 16216 max Sigma= 1193.6094863537799 total t = 134148591769398.03 j = 92 Sigma[Inner] = 192.92054125823412
    dt= 12878852207.42236 timestep= 16226 max Sigma= 1193.7219862809209 total t = 134277339283249.47 j = 92 Sigma[Inner] = 192.78809634482752
    dt= 12887958454.846128 timestep= 16236 max Sigma= 1193.8337775300317 total t = 134406177896356.45 j = 92 Sigma[Inner] = 192.655631700814
    dt= 12897056748.9659 timestep= 16246 max Sigma= 1193.944856468918 total t = 134535107527974.38 j = 92 Sigma[Inner] = 192.5231459618989
    dt= 12906147352.866632 timestep= 16256 max Sigma= 1194.0552195622186 total t = 134664128100023.33 j = 92 Sigma[Inner] = 192.39063770061475
    dt= 12915230522.331875 timestep= 16266 max Sigma= 1194.1648633703255 total t = 134793239537013.83 j = 92 Sigma[Inner] = 192.25810543364977
    dt= 12924306506.098068 timestep= 16276 max Sigma= 1194.273784548265 total t = 134922441765975.23 j = 92 Sigma[Inner] = 192.1255476288503
    dt= 12933375546.097517 timestep= 16286 max Sigma= 1194.381979844544 total t = 135051734716386.75 j = 92 Sigma[Inner] = 191.99296271190434
    dt= 12942437877.69043 timestep= 16296 max Sigma= 1194.4894460999597 total t = 135181118320110.52 j = 92 Sigma[Inner] = 191.86034907270889
    dt= 12951493729.886894 timestep= 16306 max Sigma= 1194.596180246385 total t = 135310592511327.17 j = 92 Sigma[Inner] = 191.72770507142792
    dt= 12960543325.559042 timestep= 16316 max Sigma= 1194.7021793055287 total t = 135440157226473.52 j = 92 Sigma[Inner] = 191.59502904424662
    dt= 12969586881.6441 timestep= 16326 max Sigma= 1194.8074403876744 total t = 135569812404182.27 j = 92 Sigma[Inner] = 191.46231930882874
    dt= 12978624609.33886 timestep= 16336 max Sigma= 1194.9119606904007 total t = 135699557985223.8 j = 92 Sigma[Inner] = 191.3295741694839
    dt= 12987656714.28571 timestep= 16346 max Sigma= 1195.0157374972891 total t = 135829393912449.73 j = 92 Sigma[Inner] = 191.19679192205197
    dt= 12996683396.751036 timestep= 16356 max Sigma= 1195.1187681766196 total t = 135959320130738.47 j = 92 Sigma[Inner] = 191.06397085851341
    dt= 13005704851.79614 timestep= 16366 max Sigma= 1195.2210501800523 total t = 136089336586942.22 j = 92 Sigma[Inner] = 190.93110927133168
    dt= 13014721269.441175 timestep= 16376 max Sigma= 1195.3225810413085 total t = 136219443229836.0 j = 92 Sigma[Inner] = 190.79820545753668
    dt= 13023732834.82234 timestep= 16386 max Sigma= 1195.4233583748448 total t = 136349640010068.02 j = 92 Sigma[Inner] = 190.66525772255756
    dt= 13032739728.342772 timestep= 16396 max Sigma= 1195.523379874522 total t = 136479926880111.67 j = 92 Sigma[Inner] = 190.53226438381273
    dt= 13041742125.817463 timestep= 16406 max Sigma= 1195.6226433122781 total t = 136610303794219.05 j = 92 Sigma[Inner] = 190.3992237740659
    dt= 13050740198.612373 timestep= 16416 max Sigma= 1195.721146536797 total t = 136740770708375.92 j = 92 Sigma[Inner] = 190.26613424455508
    dt= 13059734113.77812 timestep= 16426 max Sigma= 1195.8188874721852 total t = 136871327580257.7 j = 92 Sigma[Inner] = 190.13299416790537
    dt= 13068724034.178528 timestep= 16436 max Sigma= 1195.9158641166498 total t = 137001974369187.27 j = 92 Sigma[Inner] = 189.9998019408312
    dt= 13077710118.61419 timestep= 16446 max Sigma= 1196.0120745411812 total t = 137132711036093.66 j = 92 Sigma[Inner] = 189.86655598663765
    dt= 13086692521.941387 timestep= 16456 max Sigma= 1196.1075168882464 total t = 137263537543472.16 j = 92 Sigma[Inner] = 189.73325475752878
    dt= 13095671395.186497 timestep= 16466 max Sigma= 1196.2021893704866 total t = 137394453855345.4 j = 92 Sigma[Inner] = 189.59989673673087
    dt= 13104646885.656153 timestep= 16476 max Sigma= 1196.2960902694263 total t = 137525459937225.84 j = 92 Sigma[Inner] = 189.46648044043818
    dt= 13113619137.043365 timestep= 16486 max Sigma= 1196.3892179341913 total t = 137656555756078.94 j = 92 Sigma[Inner] = 189.33300441958903
    dt= 11916100641.153442 timestep= 16496 max Sigma= 1196.4756653414177 total t = 137778114490608.67 j = 92 Sigma[Inner] = 189.20803973887465
    dt= 11916100641.153442 timestep= 16506 max Sigma= 1196.558877296648 total t = 137897275497020.23 j = 92 Sigma[Inner] = 189.08676000273434
    


    
![png](output_5_133.png)
    



    
![png](output_5_134.png)
    



    
![png](output_5_135.png)
    


    dt= 11916100641.153442 timestep= 16516 max Sigma= 1196.6413975087057 total t = 138016436503431.8 j = 92 Sigma[Inner] = 188.96550242105184
    dt= 11916100641.153442 timestep= 16526 max Sigma= 1196.7232267128204 total t = 138135597509843.36 j = 92 Sigma[Inner] = 188.8442660993368
    dt= 11916100641.153442 timestep= 16536 max Sigma= 1196.8043665074356 total t = 138254758516254.92 j = 92 Sigma[Inner] = 188.72305016613544
    dt= 11974787000.950792 timestep= 16546 max Sigma= 1196.884867587516 total t = 138374049256303.0 j = 92 Sigma[Inner] = 188.60178148248224
    dt= 12130497843.073288 timestep= 16556 max Sigma= 1196.9655201708333 total t = 138494687989705.42 j = 92 Sigma[Inner] = 188.47925917714426
    dt= 12250422748.840982 timestep= 16566 max Sigma= 1197.0463990483086 total t = 138616677401820.72 j = 92 Sigma[Inner] = 188.35534515342673
    dt= 12344706983.02428 timestep= 16576 max Sigma= 1197.1272867449043 total t = 138739717979536.64 j = 92 Sigma[Inner] = 188.23035405053642
    dt= 12420578035.776312 timestep= 16586 max Sigma= 1197.2080328188733 total t = 138863595170138.17 j = 92 Sigma[Inner] = 188.10451093841746
    dt= 12483097318.548618 timestep= 16596 max Sigma= 1197.2885346550563 total t = 138988154171389.4 j = 92 Sigma[Inner] = 187.9779777968881
    dt= 12535806315.104097 timestep= 16606 max Sigma= 1197.3687236882388 total t = 139113281979630.94 j = 92 Sigma[Inner] = 187.85087248592257
    dt= 12581195745.431559 timestep= 16616 max Sigma= 1197.4485557194255 total t = 139238894904390.98 j = 92 Sigma[Inner] = 187.72328195693143
    dt= 12621033015.975363 timestep= 16626 max Sigma= 1197.5280041174908 total t = 139364929961985.2 j = 92 Sigma[Inner] = 187.59527136151053
    dt= 12656586547.511658 timestep= 16636 max Sigma= 1197.6070550042555 total t = 139491338946966.0 j = 92 Sigma[Inner] = 187.46689032341493
    dt= 12688778399.42863 timestep= 16646 max Sigma= 1197.6857037923721 total t = 139618084330190.47 j = 92 Sigma[Inner] = 187.3381772746767
    dt= 12718288112.831991 timestep= 16656 max Sigma= 1197.763952649129 total t = 139745136399115.16 j = 92 Sigma[Inner] = 187.2091624755208
    dt= 12748253335.657253 timestep= 16667 max Sigma= 1197.8495729658196 total t = 139885219498257.06 j = 92 Sigma[Inner] = 187.06692642945592
    dt= 12773638606.444244 timestep= 16677 max Sigma= 1197.927008882009 total t = 140012842968861.61 j = 92 Sigma[Inner] = 186.93735137938017
    dt= 12797557057.095072 timestep= 16687 max Sigma= 1198.0040764445798 total t = 140140712015245.86 j = 92 Sigma[Inner] = 186.80753614390326
    dt= 12820234437.146717 timestep= 16697 max Sigma= 1198.0807899195013 total t = 140268813255423.39 j = 92 Sigma[Inner] = 186.67749404324525
    dt= 12841849560.048594 timestep= 16707 max Sigma= 1198.1571642396518 total t = 140397135295511.5 j = 92 Sigma[Inner] = 186.54723636314333
    dt= 12862545934.21745 timestep= 16717 max Sigma= 1198.2332144650513 total t = 140525668327485.58 j = 92 Sigma[Inner] = 186.4167727764428
    dt= 12882440183.294703 timestep= 16727 max Sigma= 1198.3089553537184 total t = 140654403824641.55 j = 92 Sigma[Inner] = 186.28611166191172
    dt= 12901628208.713398 timestep= 16737 max Sigma= 1198.3844010224802 total t = 140783334308198.12 j = 92 Sigma[Inner] = 186.15526034834363
    dt= 12920189749.301708 timestep= 16747 max Sigma= 1198.459564682016 total t = 140912453166263.16 j = 92 Sigma[Inner] = 186.02422530378385
    dt= 12938191791.60432 timestep= 16757 max Sigma= 1198.5344584338488 total t = 141041754511764.88 j = 92 Sigma[Inner] = 185.8930122840422
    dt= 12955691148.100264 timestep= 16767 max Sigma= 1198.6090931193733 total t = 141171233069691.62 j = 92 Sigma[Inner] = 185.76162645071372
    dt= 12972736427.062069 timestep= 16777 max Sigma= 1198.6834782127564 total t = 141300884086613.88 j = 92 Sigma[Inner] = 185.6300724661682
    dt= 12989369553.315363 timestep= 16787 max Sigma= 1198.7576217508445 total t = 141430703257323.8 j = 92 Sigma[Inner] = 185.49835457100832
    dt= 13005626954.274532 timestep= 16797 max Sigma= 1198.8315302942012 total t = 141560686664762.25 j = 92 Sigma[Inner] = 185.36647664809948
    dt= 13021540494.128 timestep= 16807 max Sigma= 1198.9052089142194 total t = 141690830730362.3 j = 92 Sigma[Inner] = 185.23444227625825
    dt= 13037138216.749622 timestep= 16817 max Sigma= 1198.9786612018988 total t = 141821132172640.62 j = 92 Sigma[Inner] = 185.10225477594437
    dt= 13052444942.000465 timestep= 16827 max Sigma= 1199.0518892944588 total t = 141951587972379.7 j = 92 Sigma[Inner] = 184.9699172487482
    dt= 13067482748.635983 timestep= 16837 max Sigma= 1199.1248939164302 total t = 142082195343125.8 j = 92 Sigma[Inner] = 184.83743261204566
    dt= 13082271368.72918 timestep= 16847 max Sigma= 1199.1976744323078 total t = 142212951706011.97 j = 92 Sigma[Inner] = 184.70480362987507
    dt= 13096828512.448406 timestep= 16857 max Sigma= 1199.2702289082217 total t = 142343854668128.12 j = 92 Sigma[Inner] = 184.57203294083536
    dt= 13111170137.554337 timestep= 16867 max Sigma= 1199.342554180416 total t = 142474902003825.16 j = 92 Sigma[Inner] = 184.439123083609
    dt= 13125310674.658226 timestep= 16877 max Sigma= 1199.4146459286453 total t = 142606091638461.97 j = 92 Sigma[Inner] = 184.30607652055306
    dt= 13139263216.797394 timestep= 16887 max Sigma= 1199.4864987528488 total t = 142737421634203.62 j = 92 Sigma[Inner] = 184.17289565967184
    dt= 13153039680.00915 timestep= 16897 max Sigma= 1199.558106251723 total t = 142868890177550.0 j = 92 Sigma[Inner] = 184.03958287518327
    dt= 13166650940.161312 timestep= 16907 max Sigma= 1199.6294611020114 total t = 143000495568336.25 j = 92 Sigma[Inner] = 183.9061405268056
    dt= 13180106950.20757 timestep= 16917 max Sigma= 1199.7005551375382 total t = 143132236209990.47 j = 92 Sigma[Inner] = 183.7725709778275
    dt= 13193416841.196918 timestep= 16927 max Sigma= 1199.7713794271738 total t = 143264110600872.5 j = 92 Sigma[Inner] = 183.6388766119735
    dt= 13206589009.714636 timestep= 16937 max Sigma= 1199.8419243510673 total t = 143396117326545.97 j = 92 Sigma[Inner] = 183.50505984903936
    dt= 13219631193.923113 timestep= 16947 max Sigma= 1199.9121796746174 total t = 143528255052860.6 j = 92 Sigma[Inner] = 183.3711231592493
    dt= 13232550539.970074 timestep= 16957 max Sigma= 1199.9821346197693 total t = 143660522519740.22 j = 92 Sigma[Inner] = 183.23706907626868
    dt= 13245353660.214252 timestep= 16967 max Sigma= 1200.0517779333115 total t = 143792918535589.53 j = 92 Sigma[Inner] = 183.1029002088051
    dt= 13258046684.4654 timestep= 16977 max Sigma= 1200.1210979519497 total t = 143925441972243.06 j = 92 Sigma[Inner] = 182.96861925072636
    dt= 13270635305.232256 timestep= 16987 max Sigma= 1200.1900826639921 total t = 144058091760393.16 j = 92 Sigma[Inner] = 182.834228989637
    dt= 13283124817.807997 timestep= 16997 max Sigma= 1200.258719767553 total t = 144190866885441.5 j = 92 Sigma[Inner] = 182.69973231386408
    dt= 13295520155.88946 timestep= 17007 max Sigma= 1200.3269967252295 total t = 144323766383726.22 j = 92 Sigma[Inner] = 182.5651322178206
    


    
![png](output_5_137.png)
    



    
![png](output_5_138.png)
    



    
![png](output_5_139.png)
    


    dt= 13307825923.31726 timestep= 17017 max Sigma= 1200.3949008152492 total t = 144456789339083.44 j = 92 Sigma[Inner] = 182.43043180573244
    dt= 13320046422.434607 timestep= 17027 max Sigma= 1200.4624191791227 total t = 144589934879707.28 j = 92 Sigma[Inner] = 182.29563429373474
    dt= 13332185679.488436 timestep= 17037 max Sigma= 1200.5295388658703 total t = 144723202175277.06 j = 92 Sigma[Inner] = 182.16074301036284
    dt= 13344247467.435215 timestep= 17047 max Sigma= 1200.5962468729062 total t = 144856590434323.6 j = 92 Sigma[Inner] = 182.02576139548395
    dt= 13356235326.462263 timestep= 17057 max Sigma= 1200.66253018369 total t = 144990098901810.9 j = 92 Sigma[Inner] = 181.89069299773135
    dt= 13368152582.492718 timestep= 17067 max Sigma= 1200.7283758022631 total t = 145123726856912.0 j = 92 Sigma[Inner] = 181.7555414705233
    dt= 13380002363.905682 timestep= 17077 max Sigma= 1200.793770784805 total t = 145257473610959.5 j = 92 Sigma[Inner] = 181.62031056675949
    dt= 13391787616.672733 timestep= 17087 max Sigma= 1200.85870226835 total t = 145391338505554.75 j = 92 Sigma[Inner] = 181.4850041323041
    dt= 13403511118.085915 timestep= 17097 max Sigma= 1200.9231574967969 total t = 145525320910820.12 j = 92 Sigma[Inner] = 181.3496260983715
    dt= 13415175489.229826 timestep= 17107 max Sigma= 1200.9871238443704 total t = 145659420223782.12 j = 92 Sigma[Inner] = 181.2141804729396
    dt= 13426783206.331749 timestep= 17117 max Sigma= 1201.0505888366758 total t = 145793635866872.5 j = 92 Sigma[Inner] = 181.0786713313218
    dt= 13438336611.1075 timestep= 17127 max Sigma= 1201.1135401694892 total t = 145927967286537.12 j = 92 Sigma[Inner] = 180.94310280602824
    dt= 13449837920.206017 timestep= 17137 max Sigma= 1201.1759657254277 total t = 146062413951943.88 j = 92 Sigma[Inner] = 180.8074790760507
    dt= 13461289233.844477 timestep= 17147 max Sigma= 1201.2378535886432 total t = 146196975353779.75 j = 92 Sigma[Inner] = 180.6718043557028
    dt= 13472692543.714203 timestep= 17157 max Sigma= 1201.2991920576687 total t = 146331651003130.97 j = 92 Sigma[Inner] = 180.53608288314254
    dt= 13484049740.229338 timestep= 17167 max Sigma= 1201.3599696565525 total t = 146466440430437.88 j = 92 Sigma[Inner] = 180.4003189087017
    dt= 13495362619.181494 timestep= 17177 max Sigma= 1201.4201751444066 total t = 146601343184519.16 j = 92 Sigma[Inner] = 180.26451668313888
    dt= 13506632887.857368 timestep= 17187 max Sigma= 1201.4797975234824 total t = 146736358831659.94 j = 92 Sigma[Inner] = 180.12868044592614
    dt= 13517862170.66942 timestep= 17197 max Sigma= 1201.5388260458997 total t = 146871486954757.9 j = 92 Sigma[Inner] = 179.99281441367145
    dt= 13529052014.344944 timestep= 17207 max Sigma= 1201.5972502191244 total t = 147006727152523.44 j = 92 Sigma[Inner] = 179.8569227687695
    dt= 13540203892.713835 timestep= 17217 max Sigma= 1201.6550598103088 total t = 147142079038729.53 j = 92 Sigma[Inner] = 179.72100964836602
    dt= 13551319211.131182 timestep= 17227 max Sigma= 1201.7122448495832 total t = 147277542241507.3 j = 92 Sigma[Inner] = 179.58507913370957
    dt= 13562399310.567265 timestep= 17237 max Sigma= 1201.7687956323975 total t = 147413116402684.12 j = 92 Sigma[Inner] = 179.4491352399567
    dt= 13573445471.39413 timestep= 17247 max Sigma= 1201.8247027209902 total t = 147548801177160.72 j = 92 Sigma[Inner] = 179.3131819064868
    dt= 13584458916.895107 timestep= 17257 max Sigma= 1201.879956945072 total t = 147684596232324.78 j = 92 Sigma[Inner] = 179.17722298777386
    dt= 13595440816.52092 timestep= 17267 max Sigma= 1201.9345494017953 total t = 147820501247498.1 j = 92 Sigma[Inner] = 179.0412622448531
    dt= 13606392288.913994 timestep= 17277 max Sigma= 1201.988471455078 total t = 147956515913415.22 j = 92 Sigma[Inner] = 178.90530333741296
    dt= 13617314404.720135 timestep= 17287 max Sigma= 1202.041714734351 total t = 148092639931730.9 j = 92 Sigma[Inner] = 178.76934981653537
    dt= 13628208189.205473 timestep= 17297 max Sigma= 1202.0942711327855 total t = 148228873014554.6 j = 92 Sigma[Inner] = 178.63340511809804
    dt= 13639074624.694174 timestep= 17307 max Sigma= 1202.146132805058 total t = 148365214884010.34 j = 92 Sigma[Inner] = 178.49747255684812
    dt= 13649914652.841816 timestep= 17317 max Sigma= 1202.197292164705 total t = 148501665271819.94 j = 92 Sigma[Inner] = 178.36155532114856
    dt= 13660729176.757174 timestep= 17327 max Sigma= 1202.2477418811118 total t = 148638223918908.06 j = 92 Sigma[Inner] = 178.22565646839394
    dt= 13671519062.984732 timestep= 17337 max Sigma= 1202.2974748761842 total t = 148774890575027.8 j = 92 Sigma[Inner] = 178.0897789210876
    dt= 13682285143.35848 timestep= 17347 max Sigma= 1202.34648432074 total t = 148911664998405.12 j = 92 Sigma[Inner] = 177.95392546356572
    dt= 13693028216.737188 timestep= 17357 max Sigma= 1202.3947636306618 total t = 149048546955401.8 j = 92 Sigma[Inner] = 177.81809873935353
    dt= 13703749050.63005 timestep= 17367 max Sigma= 1202.442306462836 total t = 149185536220194.5 j = 92 Sigma[Inner] = 177.68230124913148
    dt= 13714448382.721045 timestep= 17377 max Sigma= 1202.4891067109227 total t = 149322632574469.47 j = 92 Sigma[Inner] = 177.54653534929
    dt= 13725126922.299604 timestep= 17387 max Sigma= 1202.5351585009714 total t = 149459835807132.72 j = 92 Sigma[Inner] = 177.41080325104627
    dt= 13735785351.604416 timestep= 17397 max Sigma= 1202.580456186919 total t = 149597145714033.25 j = 92 Sigma[Inner] = 177.27510702009644
    dt= 13746424327.086857 timestep= 17407 max Sigma= 1202.6249943459904 total t = 149734562097699.6 j = 92 Sigma[Inner] = 177.13944857677458
    dt= 13757044480.599733 timestep= 17417 max Sigma= 1202.6687677740167 total t = 149872084767088.47 j = 92 Sigma[Inner] = 177.00382969668868
    dt= 13767646420.516768 timestep= 17427 max Sigma= 1202.7117714807055 total t = 150009713537345.25 j = 92 Sigma[Inner] = 176.86825201180307
    dt= 13778230732.787657 timestep= 17437 max Sigma= 1202.754000684864 total t = 150147448229574.7 j = 92 Sigma[Inner] = 176.73271701193596
    dt= 13788797981.933289 timestep= 17447 max Sigma= 1202.795450809604 total t = 150285288670622.5 j = 92 Sigma[Inner] = 176.59722604664222
    dt= 13799348711.985237 timestep= 17457 max Sigma= 1202.8361174775346 total t = 150423234692866.16 j = 92 Sigma[Inner] = 176.46178032744973
    dt= 13809883447.37322 timestep= 17467 max Sigma= 1202.8759965059605 total t = 150561286134014.88 j = 92 Sigma[Inner] = 176.32638093041874
    dt= 13820402693.764381 timestep= 17477 max Sigma= 1202.9150839020986 total t = 150699442836918.34 j = 92 Sigma[Inner] = 176.1910287989944
    dt= 13830906938.857151 timestep= 17487 max Sigma= 1202.953375858314 total t = 150837704649383.47 j = 92 Sigma[Inner] = 176.05572474712352
    dt= 13841396653.133154 timestep= 17497 max Sigma= 1202.9908687473994 total t = 150976071423998.75 j = 92 Sigma[Inner] = 175.92046946260646
    dt= 13851872290.5696 timestep= 17507 max Sigma= 1203.0275591178906 total t = 151114543017966.28 j = 92 Sigma[Inner] = 175.78526351065747
    


    
![png](output_5_141.png)
    



    
![png](output_5_142.png)
    



    
![png](output_5_143.png)
    


    dt= 13862334289.31484 timestep= 17517 max Sigma= 1203.0634436894366 total t = 151253119292940.28 j = 92 Sigma[Inner] = 175.65010733764615
    dt= 13872783072.329472 timestep= 17527 max Sigma= 1203.0985193482254 total t = 151391800114872.4 j = 92 Sigma[Inner] = 175.5150012749955
    dt= 13883219047.99507 timestep= 17537 max Sigma= 1203.1327831424708 total t = 151530585353863.2 j = 92 Sigma[Inner] = 175.3799455432118
    dt= 13893642610.692673 timestep= 17547 max Sigma= 1203.1662322779694 total t = 151669474884019.56 j = 92 Sigma[Inner] = 175.24494025602434
    dt= 13904054141.352919 timestep= 17557 max Sigma= 1203.1988641137252 total t = 151808468583317.8 j = 92 Sigma[Inner] = 175.1099854246124
    dt= 13914454007.979322 timestep= 17567 max Sigma= 1203.2306761576501 total t = 151947566333472.06 j = 92 Sigma[Inner] = 174.9750809618993
    dt= 13924842566.146702 timestep= 17577 max Sigma= 1203.2616660623432 total t = 152086768019807.75 j = 92 Sigma[Inner] = 174.84022668689528
    dt= 13935220159.475908 timestep= 17587 max Sigma= 1203.291831620947 total t = 152226073531139.84 j = 92 Sigma[Inner] = 174.7054223290702
    dt= 13945587120.086405 timestep= 17597 max Sigma= 1203.321170763088 total t = 152365482759655.88 j = 92 Sigma[Inner] = 174.57066753274006
    dt= 13955943769.028015 timestep= 17607 max Sigma= 1203.3496815508997 total t = 152504995600803.4 j = 92 Sigma[Inner] = 174.43596186145243
    dt= 13966290416.692928 timestep= 17617 max Sigma= 1203.3773621751293 total t = 152644611953181.56 j = 92 Sigma[Inner] = 174.30130480235533
    dt= 13976627363.209156 timestep= 17627 max Sigma= 1203.40421095133 total t = 152784331718436.56 j = 92 Sigma[Inner] = 174.16669577053818
    dt= 13986954898.816526 timestep= 17637 max Sigma= 1203.4302263161412 total t = 152924154801161.38 j = 92 Sigma[Inner] = 174.03213411333147
    dt= 13011757564.199371 timestep= 17647 max Sigma= 1203.4550580195262 total t = 153061127655979.53 j = 92 Sigma[Inner] = 173.8995102646139
    dt= 13011757564.199371 timestep= 17657 max Sigma= 1203.4777173131863 total t = 153191245231621.4 j = 92 Sigma[Inner] = 173.77454536133467
    dt= 13011757564.199371 timestep= 17667 max Sigma= 1203.4996404590843 total t = 153321362807263.28 j = 92 Sigma[Inner] = 173.6497045302725
    dt= 13011757564.199371 timestep= 17677 max Sigma= 1203.5208356334813 total t = 153451480382905.16 j = 92 Sigma[Inner] = 173.52498676402524
    dt= 13011757564.199371 timestep= 17687 max Sigma= 1203.5413199352645 total t = 153581597958547.03 j = 92 Sigma[Inner] = 173.40039108335546
    dt= 13011757564.199371 timestep= 17697 max Sigma= 1203.5611201442837 total t = 153711715534188.9 j = 92 Sigma[Inner] = 173.27591646491754
    dt= 13132483946.906752 timestep= 17707 max Sigma= 1203.5803312287594 total t = 153842361567986.4 j = 92 Sigma[Inner] = 173.15117230343975
    dt= 13262513760.906872 timestep= 17717 max Sigma= 1203.5991322212108 total t = 153974428501475.84 j = 92 Sigma[Inner] = 173.02520041637194
    dt= 13364656924.978476 timestep= 17727 max Sigma= 1203.6175221121057 total t = 154107634840732.5 j = 92 Sigma[Inner] = 172.89823652562453
    dt= 13446660510.1163 timestep= 17737 max Sigma= 1203.6355112412898 total t = 154241746491079.6 j = 92 Sigma[Inner] = 172.77051305409802
    dt= 13514018036.82424 timestep= 17747 max Sigma= 1203.6531267059256 total t = 154376593842157.94 j = 92 Sigma[Inner] = 172.64219794723417
    dt= 13563096993.130703 timestep= 17757 max Sigma= 1203.6704046512052 total t = 154512035035474.06 j = 92 Sigma[Inner] = 172.5134233851137
    dt= 13595057442.059547 timestep= 17767 max Sigma= 1203.6873715389675 total t = 154647842984205.0 j = 92 Sigma[Inner] = 172.3844065122337
    dt= 13625493196.21474 timestep= 17777 max Sigma= 1203.704073067958 total t = 154783962252749.0 j = 92 Sigma[Inner] = 172.25521440908759
    dt= 13654361610.215593 timestep= 17787 max Sigma= 1203.72055850461 total t = 154920377238559.0 j = 92 Sigma[Inner] = 172.12586127911416
    dt= 13681726060.420933 timestep= 17797 max Sigma= 1203.736872580159 total t = 155057072558097.44 j = 92 Sigma[Inner] = 171.99636114575145
    dt= 13707698531.852016 timestep= 17807 max Sigma= 1203.7530549152584 total t = 155194033763397.2 j = 92 Sigma[Inner] = 171.86672712335064
    dt= 13732408206.262342 timestep= 17817 max Sigma= 1203.7691396359926 total t = 155331247639476.25 j = 92 Sigma[Inner] = 171.73697111278167
    dt= 13755984862.447315 timestep= 17827 max Sigma= 1203.7851551733386 total t = 155468702275971.25 j = 92 Sigma[Inner] = 171.6071037261057
    dt= 13778550655.514273 timestep= 17837 max Sigma= 1203.8011242240093 total t = 155606387022815.88 j = 92 Sigma[Inner] = 171.47713432903802
    dt= 13800216567.057703 timestep= 17847 max Sigma= 1203.8170638453275 total t = 155744292391684.1 j = 92 Sigma[Inner] = 171.34707113867546
    dt= 13821081392.334442 timestep= 17857 max Sigma= 1203.8329856562805 total t = 155882409937197.9 j = 92 Sigma[Inner] = 171.21692134204943
    dt= 13841232043.686049 timestep= 17867 max Sigma= 1203.8488961188245 total t = 156020732135991.2 j = 92 Sigma[Inner] = 171.08669121717872
    dt= 13860744479.943453 timestep= 17877 max Sigma= 1203.8647968765658 total t = 156159252272668.28 j = 92 Sigma[Inner] = 170.95638624745746
    dt= 13879684879.293848 timestep= 17887 max Sigma= 1203.8806851313295 total t = 156297964336623.44 j = 92 Sigma[Inner] = 170.82601122534118
    dt= 13898110850.757011 timestep= 17897 max Sigma= 1203.8965540414922 total t = 156436862930919.0 j = 92 Sigma[Inner] = 170.6955703440945
    dt= 13916072581.032103 timestep= 17907 max Sigma= 1203.9123931290576 total t = 156575943192980.44 j = 92 Sigma[Inner] = 170.5650672778235
    dt= 13933613870.671211 timestep= 17917 max Sigma= 1203.928188685195 total t = 156715200726172.62 j = 92 Sigma[Inner] = 170.43450525071756
    dt= 13950773044.889946 timestep= 17927 max Sigma= 1203.9439241663529 total t = 156854631541052.1 j = 92 Sigma[Inner] = 170.30388709670763
    dt= 13967583740.756855 timestep= 17937 max Sigma= 1203.959580575062 total t = 156994232005042.22 j = 92 Sigma[Inner] = 170.17321531079963
    dt= 13984075580.431185 timestep= 17947 max Sigma= 1203.9751368212167 total t = 157133998799343.34 j = 92 Sigma[Inner] = 170.04249209328495
    dt= 14000274743.297016 timestep= 17957 max Sigma= 1203.9905700609972 total t = 157273928882008.38 j = 92 Sigma[Inner] = 169.91171938791405
    dt= 14016204450.460876 timestep= 17967 max Sigma= 1204.0058560116977 total t = 157414019456246.53 j = 92 Sigma[Inner] = 169.7808989149917
    dt= 14031885374.398518 timestep= 17977 max Sigma= 1204.0209692415992 total t = 157554267943150.38 j = 92 Sigma[Inner] = 169.65003220022257
    dt= 14047335985.268698 timestep= 17987 max Sigma= 1204.0358834347248 total t = 157694671958162.12 j = 92 Sigma[Inner] = 169.51912060001303
    dt= 14062572843.951166 timestep= 17997 max Sigma= 1204.050571630826 total t = 157835229290702.5 j = 92 Sigma[Inner] = 169.38816532382984
    dt= 14077610850.41895 timestep= 18007 max Sigma= 1204.0650064413476 total t = 157975937886477.44 j = 92 Sigma[Inner] = 169.25716745411663
    


    
![png](output_5_145.png)
    



    
![png](output_5_146.png)
    



    
![png](output_5_147.png)
    


    dt= 14092463454.723734 timestep= 18017 max Sigma= 1204.079160242412 total t = 158116795832057.9 j = 92 Sigma[Inner] = 169.12612796418816
    dt= 14107142836.69584 timestep= 18027 max Sigma= 1204.0930053460363 total t = 158257801341392.4 j = 92 Sigma[Inner] = 168.9950477344474
    dt= 14121660059.4473 timestep= 18037 max Sigma= 1204.1065141509507 total t = 158398952743968.6 j = 92 Sigma[Inner] = 168.86392756720858
    dt= 14136025200.908335 timestep= 18047 max Sigma= 1204.1196592744282 total t = 158540248474386.56 j = 92 Sigma[Inner] = 168.73276820035275
    dt= 14150247466.908705 timestep= 18057 max Sigma= 1204.1324136665814 total t = 158681687063142.47 j = 92 Sigma[Inner] = 168.60157031999609
    dt= 14164335288.717402 timestep= 18067 max Sigma= 1204.1447507085636 total t = 158823267128456.16 j = 92 Sigma[Inner] = 168.47033457230907
    dt= 14178296407.459501 timestep= 18077 max Sigma= 1204.1566442960852 total t = 158964987368999.4 j = 92 Sigma[Inner] = 168.33906157459046
    dt= 14192137947.42061 timestep= 18087 max Sigma= 1204.1680689096006 total t = 159106846557406.34 j = 92 Sigma[Inner] = 168.2077519256677
    dt= 14205866479.912682 timestep= 18097 max Sigma= 1204.1789996724565 total t = 159248843534463.5 j = 92 Sigma[Inner] = 168.07640621567293
    dt= 14219488079.097525 timestep= 18107 max Sigma= 1204.1894123982268 total t = 159390977203892.97 j = 92 Sigma[Inner] = 167.94502503522205
    dt= 14233008370.936174 timestep= 18117 max Sigma= 1204.1992836283705 total t = 159533246527655.72 j = 92 Sigma[Inner] = 167.81360898400703
    dt= 14246432576.24345 timestep= 18127 max Sigma= 1204.2085906612792 total t = 159675650521710.62 j = 92 Sigma[Inner] = 167.68215867880056
    dt= 14259765548.671371 timestep= 18137 max Sigma= 1204.2173115736941 total t = 159818188252175.9 j = 92 Sigma[Inner] = 167.55067476086197
    dt= 14273011808.316153 timestep= 18147 max Sigma= 1204.2254252353935 total t = 159960858831845.56 j = 92 Sigma[Inner] = 167.4191579027284
    dt= 14286175571.536568 timestep= 18157 max Sigma= 1204.2329113179808 total t = 160103661417020.94 j = 92 Sigma[Inner] = 167.28760881437174
    dt= 14299260777.482485 timestep= 18167 max Sigma= 1204.2397502985166 total t = 160246595204621.7 j = 92 Sigma[Inner] = 167.15602824870038
    dt= 14312271111.758165 timestep= 18177 max Sigma= 1204.245923458686 total t = 160389659429545.97 j = 92 Sigma[Inner] = 167.0244170063883
    dt= 14325210027.582928 timestep= 18187 max Sigma= 1204.2514128801097 total t = 160532853362253.4 j = 92 Sigma[Inner] = 166.89277594001473
    dt= 14338080764.759808 timestep= 18197 max Sigma= 1204.2562014363564 total t = 160676176306547.16 j = 92 Sigma[Inner] = 166.76110595750453
    dt= 14350886366.719059 timestep= 18207 max Sigma= 1204.260272782156 total t = 160819627597535.0 j = 92 Sigma[Inner] = 166.62940802486193
    dt= 14363629695.866385 timestep= 18217 max Sigma= 1204.2636113402523 total t = 160963206599751.5 j = 92 Sigma[Inner] = 166.4976831682002
    dt= 14376313447.434727 timestep= 18227 max Sigma= 1204.2662022862987 total t = 161106912705424.75 j = 92 Sigma[Inner] = 166.3659324750715
    dt= 14388940162.011763 timestep= 18237 max Sigma= 1204.2680315321443 total t = 161250745332874.97 j = 92 Sigma[Inner] = 166.2341570951123
    dt= 14401512236.892807 timestep= 18247 max Sigma= 1204.2690857078244 total t = 161394703925031.28 j = 92 Sigma[Inner] = 166.10235824002282
    dt= 14414031936.389593 timestep= 18257 max Sigma= 1204.2693521425267 total t = 161538787948056.03 j = 92 Sigma[Inner] = 165.97053718290647
    dt= 14426501401.208746 timestep= 18267 max Sigma= 1204.2688188447794 total t = 161682996890067.66 j = 92 Sigma[Inner] = 165.83869525700007
    dt= 14438922656.999866 timestep= 18277 max Sigma= 1204.2674744820697 total t = 161827330259952.03 j = 92 Sigma[Inner] = 165.70683385383163
    dt= 14451297622.160658 timestep= 18287 max Sigma= 1204.2653083600787 total t = 161971787586255.5 j = 92 Sigma[Inner] = 165.57495442084553
    dt= 14463628114.97627 timestep= 18297 max Sigma= 1204.2623104016923 total t = 162116368416152.12 j = 92 Sigma[Inner] = 165.4430584585396
    dt= 14475915860.160656 timestep= 18307 max Sigma= 1204.2584711259256 total t = 162261072314479.06 j = 92 Sigma[Inner] = 165.31114751715967
    dt= 14488162494.86018 timestep= 18317 max Sigma= 1204.2537816268828 total t = 162405898862834.22 j = 92 Sigma[Inner] = 165.1792231930013
    dt= 14500369574.172316 timestep= 18327 max Sigma= 1204.2482335528475 total t = 162550847658731.25 j = 92 Sigma[Inner] = 165.04728712436813
    dt= 14512538576.226954 timestep= 18337 max Sigma= 1204.2418190856035 total t = 162695918314807.4 j = 92 Sigma[Inner] = 164.91534098723776
    dt= 14524670906.871782 timestep= 18347 max Sigma= 1204.2345309200396 total t = 162841110458080.3 j = 92 Sigma[Inner] = 164.78338649068436
    dt= 14536767903.999493 timestep= 18357 max Sigma= 1204.2263622441121 total t = 162986423729248.5 j = 92 Sigma[Inner] = 164.65142537210932
    dt= 14548830841.549603 timestep= 18367 max Sigma= 1204.2173067192182 total t = 163131857782034.7 j = 92 Sigma[Inner] = 164.51945939232704
    dt= 14560860933.215096 timestep= 18377 max Sigma= 1204.2073584610048 total t = 163277412282566.62 j = 92 Sigma[Inner] = 164.3874903305527
    dt= 14572859335.880043 timestep= 18387 max Sigma= 1204.1965120206596 total t = 163423086908793.88 j = 92 Sigma[Inner] = 164.25551997933704
    dt= 14584827152.812252 timestep= 18397 max Sigma= 1204.1847623667036 total t = 163568881349937.38 j = 92 Sigma[Inner] = 164.12355013948968
    dt= 14596765436.632334 timestep= 18407 max Sigma= 1204.1721048673023 total t = 163714795305969.8 j = 92 Sigma[Inner] = 163.9915826150304
    dt= 14608675192.078348 timestep= 18417 max Sigma= 1204.158535273116 total t = 163860828487124.6 j = 92 Sigma[Inner] = 163.85961920820503
    dt= 14620557378.248478 timestep= 18427 max Sigma= 1204.1678730936208 total t = 164006980613430.6 j = 92 Sigma[Inner] = 163.72766171459844
    dt= 14632412883.361862 timestep= 18437 max Sigma= 1204.176182633208 total t = 164153251414175.84 j = 92 Sigma[Inner] = 163.59571191843824
    dt= 14644242407.729595 timestep= 18447 max Sigma= 1204.1810397357574 total t = 164299640626607.12 j = 92 Sigma[Inner] = 163.46377158867875
    dt= 14656046391.467625 timestep= 18457 max Sigma= 1204.1827703759427 total t = 164446147993548.44 j = 92 Sigma[Inner] = 163.33184247629293
    dt= 14667825050.29698 timestep= 18467 max Sigma= 1204.180884952478 total t = 164592773260935.28 j = 92 Sigma[Inner] = 163.19992631194447
    dt= 14679578454.84122 timestep= 18477 max Sigma= 1204.1748529074894 total t = 164739516175990.53 j = 92 Sigma[Inner] = 163.06802480334542
    dt= 14691306618.06321 timestep= 18487 max Sigma= 1204.1652663262244 total t = 164886376486254.8 j = 92 Sigma[Inner] = 162.93613963209194
    dt= 14703009575.698515 timestep= 18497 max Sigma= 1204.152603173845 total t = 165033353939464.03 j = 92 Sigma[Inner] = 162.8042724499868
    dt= 14714687447.178087 timestep= 18507 max Sigma= 1204.1372444694362 total t = 165180448284131.2 j = 92 Sigma[Inner] = 162.67242487497853
    


    
![png](output_5_149.png)
    



    
![png](output_5_150.png)
    



    
![png](output_5_151.png)
    


    dt= 14726340472.848944 timestep= 18517 max Sigma= 1204.1194949758178 total t = 165327659270607.47 j = 92 Sigma[Inner] = 162.54059848692495
    dt= 14737969029.941292 timestep= 18527 max Sigma= 1204.0995993360516 total t = 165474986652394.7 j = 92 Sigma[Inner] = 162.40879482339287
    dt= 14749573632.143671 timestep= 18537 max Sigma= 1204.0777546885602 total t = 165622430187523.3 j = 92 Sigma[Inner] = 162.27701537566855
    dt= 14171725847.449713 timestep= 18547 max Sigma= 1204.054893240721 total t = 165764726452948.7 j = 92 Sigma[Inner] = 162.14943359976877
    dt= 14171725847.449713 timestep= 18557 max Sigma= 1204.0307465919107 total t = 165906443711423.06 j = 92 Sigma[Inner] = 162.0230056173664
    dt= 14171725847.449713 timestep= 18567 max Sigma= 1204.0054060355712 total t = 166048160969897.44 j = 92 Sigma[Inner] = 161.89669838164264
    dt= 14171725847.449713 timestep= 18577 max Sigma= 1203.9791959945678 total t = 166189878228371.8 j = 92 Sigma[Inner] = 161.77051260777944
    dt= 14171725847.449713 timestep= 18587 max Sigma= 1203.9524463857804 total t = 166331595486846.2 j = 92 Sigma[Inner] = 161.64444896065285
    dt= 14223630791.329245 timestep= 18597 max Sigma= 1203.9254501868702 total t = 166473432745205.7 j = 92 Sigma[Inner] = 161.51844756043798
    dt= 14353531512.940575 timestep= 18607 max Sigma= 1203.898285352125 total t = 166616409828368.78 j = 92 Sigma[Inner] = 161.39162721333557
    dt= 14430462930.037676 timestep= 18617 max Sigma= 1203.871164402745 total t = 166760466609139.75 j = 92 Sigma[Inner] = 161.26392842825683
    dt= 14458375507.211664 timestep= 18627 max Sigma= 1203.8443936927279 total t = 166904922337591.75 j = 92 Sigma[Inner] = 161.13596116814267
    dt= 14488111303.556599 timestep= 18637 max Sigma= 1203.818125580518 total t = 167049668916424.2 j = 92 Sigma[Inner] = 161.00786770584767
    dt= 14518122448.452488 timestep= 18647 max Sigma= 1203.7924316198037 total t = 167194715282175.66 j = 92 Sigma[Inner] = 160.87963992655716
    dt= 14547592401.17323 timestep= 18657 max Sigma= 1203.7673430461853 total t = 167340059248670.2 j = 92 Sigma[Inner] = 160.75128017610405
    dt= 14576123043.497587 timestep= 18667 max Sigma= 1203.742856478122 total t = 167485692958782.56 j = 92 Sigma[Inner] = 160.62279612222903
    dt= 14603552949.013538 timestep= 18677 max Sigma= 1203.7189400602083 total t = 167631605988556.9 j = 92 Sigma[Inner] = 160.49419783038056
    dt= 14629851415.331577 timestep= 18687 max Sigma= 1203.695539430666 total t = 167777787084376.97 j = 92 Sigma[Inner] = 160.3654961276457
    dt= 14655056799.22221 timestep= 18697 max Sigma= 1203.6725831757574 total t = 167924225103482.47 j = 92 Sigma[Inner] = 160.23670171663258
    dt= 14674666584.143377 timestep= 18707 max Sigma= 1203.6499888671524 total t = 168070896725583.5 j = 92 Sigma[Inner] = 160.1078319200156
    dt= 14688654226.264696 timestep= 18717 max Sigma= 1203.6276754751848 total t = 168217720038265.16 j = 92 Sigma[Inner] = 159.97895928643766
    dt= 14702928519.714834 timestep= 18727 max Sigma= 1203.605547098443 total t = 168364684897659.78 j = 92 Sigma[Inner] = 159.85009830850825
    dt= 14717391105.807434 timestep= 18737 max Sigma= 1203.5835034135298 total t = 168511793605107.94 j = 92 Sigma[Inner] = 159.72124721506893
    dt= 14731969132.321487 timestep= 18747 max Sigma= 1203.5614447403639 total t = 168659047624735.62 j = 92 Sigma[Inner] = 159.59240495444405
    dt= 14746608820.876944 timestep= 18757 max Sigma= 1203.5392735837327 total t = 168806447801632.44 j = 92 Sigma[Inner] = 159.4635709951306
    dt= 14761270638.812952 timestep= 18767 max Sigma= 1203.5168957792635 total t = 168953994524884.62 j = 92 Sigma[Inner] = 159.33474517730792
    dt= 14775925689.418802 timestep= 18777 max Sigma= 1203.4942213138304 total t = 169101687849309.75 j = 92 Sigma[Inner] = 159.205927602479
    dt= 14790553018.529608 timestep= 18787 max Sigma= 1203.4711648817213 total t = 169249527586351.28 j = 92 Sigma[Inner] = 159.07711855157717
    dt= 14805137606.211777 timestep= 18797 max Sigma= 1203.4476462292785 total t = 169397513371973.06 j = 92 Sigma[Inner] = 158.94831842428846
    dt= 14819668868.805187 timestep= 18807 max Sigma= 1203.4235903326494 total t = 169545644717401.2 j = 92 Sigma[Inner] = 158.81952769418234
    dt= 14834139540.457684 timestep= 18817 max Sigma= 1203.3989274459243 total t = 169693921047067.06 j = 92 Sigma[Inner] = 158.69074687563463
    dt= 14848544836.684994 timestep= 18827 max Sigma= 1203.3735930504654 total t = 169842341726985.12 j = 92 Sigma[Inner] = 158.56197649955993
    dt= 14862881827.604593 timestep= 18837 max Sigma= 1203.347527730587 total t = 169990906085971.2 j = 92 Sigma[Inner] = 158.43321709574099
    dt= 14877148967.228945 timestep= 18847 max Sigma= 1203.3206769959388 total t = 170139613431488.8 j = 92 Sigma[Inner] = 158.30446918011114
    dt= 14891345739.110233 timestep= 18857 max Sigma= 1203.2929910668802 total t = 170288463061456.8 j = 92 Sigma[Inner] = 158.17573324577089
    dt= 14905472388.918814 timestep= 18867 max Sigma= 1203.2644246357538 total t = 170437454273012.12 j = 92 Sigma[Inner] = 158.04700975682894
    dt= 14919529722.138924 timestep= 18877 max Sigma= 1203.2349366141518 total t = 170586586368972.0 j = 92 Sigma[Inner] = 157.91829914439208
    dt= 14933518950.677307 timestep= 18887 max Sigma= 1203.2044898739885 total t = 170735858662553.72 j = 92 Sigma[Inner] = 157.78960180419898
    dt= 14947441576.32505 timestep= 18897 max Sigma= 1203.173050988315 total t = 170885270480771.22 j = 92 Sigma[Inner] = 157.6609180955198
    dt= 14961299302.07621 timestep= 18907 max Sigma= 1203.1405899763067 total t = 171034821166826.75 j = 92 Sigma[Inner] = 157.5322483410396
    dt= 14975093964.574398 timestep= 18917 max Sigma= 1203.1070800556618 total t = 171184510081736.03 j = 92 Sigma[Inner] = 157.40359282751186
    dt= 14988827482.6399 timestep= 18927 max Sigma= 1203.0724974046777 total t = 171334336605370.1 j = 92 Sigma[Inner] = 157.2749518070195
    dt= 15002501818.078794 timestep= 18937 max Sigma= 1203.0368209355252 total t = 171484300137052.47 j = 92 Sigma[Inner] = 157.14632549872212
    dt= 15016118945.906513 timestep= 18947 max Sigma= 1203.00003207966 total t = 171634400095818.9 j = 92 Sigma[Inner] = 157.01771409099177
    dt= 15029680831.812864 timestep= 18957 max Sigma= 1202.9621145858537 total t = 171784635920419.94 j = 92 Sigma[Inner] = 156.88911774386114
    dt= 15043189415.215836 timestep= 18967 max Sigma= 1202.9230543309736 total t = 171935007069130.72 j = 92 Sigma[Inner] = 156.7605365917254
    dt= 15056646596.642483 timestep= 18977 max Sigma= 1202.882839143415 total t = 172085513019415.38 j = 92 Sigma[Inner] = 156.63197074624298
    dt= 15070054228.469631 timestep= 18987 max Sigma= 1202.8414586388565 total t = 172236153267485.06 j = 92 Sigma[Inner] = 156.50342029939318
    dt= 15083414108.279839 timestep= 18997 max Sigma= 1202.79890406792 total t = 172386927327777.84 j = 92 Sigma[Inner] = 156.3748853266504
    dt= 15096727974.25699 timestep= 19007 max Sigma= 1202.7551681751966 total t = 172537834732384.03 j = 92 Sigma[Inner] = 156.24636589023848
    


    
![png](output_5_153.png)
    



    
![png](output_5_154.png)
    



    
![png](output_5_155.png)
    


    dt= 15109997502.174597 timestep= 19017 max Sigma= 1202.7102450690545 total t = 172688875030434.44 j = 92 Sigma[Inner] = 156.1178620424303
    dt= 15123224303.628263 timestep= 19027 max Sigma= 1202.6641301016107 total t = 172840047787465.7 j = 92 Sigma[Inner] = 155.98937382886183
    dt= 15136409925.239424 timestep= 19037 max Sigma= 1202.616819758236 total t = 172991352584773.22 j = 92 Sigma[Inner] = 155.86090129182682
    dt= 15149555848.615616 timestep= 19047 max Sigma= 1202.5683115559618 total t = 173142789018761.0 j = 92 Sigma[Inner] = 155.73244447352374
    dt= 15162663490.897657 timestep= 19057 max Sigma= 1202.5186039501746 total t = 173294356700293.72 j = 92 Sigma[Inner] = 155.60400341922448
    dt= 15175734205.75944 timestep= 19067 max Sigma= 1202.4676962490018 total t = 173446055254057.47 j = 92 Sigma[Inner] = 155.47557818033823
    dt= 15188769284.753178 timestep= 19077 max Sigma= 1202.415588534812 total t = 173597884317932.3 j = 92 Sigma[Inner] = 155.34716881734334
    dt= 15201769958.914831 timestep= 19087 max Sigma= 1202.3622815922915 total t = 173749843542379.4 j = 92 Sigma[Inner] = 155.21877540256338
    dt= 15214737400.56135 timestep= 19097 max Sigma= 1202.3077768425837 total t = 173901932589845.84 j = 92 Sigma[Inner] = 155.0903980227661
    dt= 15227672725.224812 timestep= 19107 max Sigma= 1202.2520762830095 total t = 174054151134187.94 j = 92 Sigma[Inner] = 154.9620367815642
    dt= 15240576993.679596 timestep= 19117 max Sigma= 1202.1951824319176 total t = 174206498860114.38 j = 92 Sigma[Inner] = 154.8336918016021
    dt= 15253451214.026941 timestep= 19127 max Sigma= 1202.137098278249 total t = 174358975462650.28 j = 92 Sigma[Inner] = 154.70536322651392
    dt= 15266296343.808823 timestep= 19137 max Sigma= 1202.0778272354273 total t = 174511580646621.94 j = 92 Sigma[Inner] = 154.57705122264164
    dt= 15279113292.12803 timestep= 19147 max Sigma= 1202.017373099211 total t = 174664314126162.72 j = 92 Sigma[Inner] = 154.44875598050479
    dt= 15291679087.408266 timestep= 19157 max Sigma= 1201.955929239928 total t = 174817175067947.34 j = 92 Sigma[Inner] = 154.32047799492142
    dt= 15302435401.201067 timestep= 19167 max Sigma= 1201.8949205061735 total t = 174970153081177.25 j = 92 Sigma[Inner] = 154.1922246841975
    dt= 15310585334.08917 timestep= 19177 max Sigma= 1201.8357012323554 total t = 175123224386060.9 j = 92 Sigma[Inner] = 154.06401532103507
    dt= 15316416716.460571 timestep= 19187 max Sigma= 1201.778982665732 total t = 175276363975948.88 j = 92 Sigma[Inner] = 153.93587107177655
    dt= 15320584735.13775 timestep= 19197 max Sigma= 1201.7250399731745 total t = 175429552147128.84 j = 92 Sigma[Inner] = 153.8078089304648
    dt= 15323773882.478395 timestep= 19207 max Sigma= 1201.6738624218478 total t = 175582776082244.3 j = 92 Sigma[Inner] = 153.6798401344279
    dt= 15326581253.017956 timestep= 19217 max Sigma= 1201.6252444092474 total t = 175736029352462.28 j = 92 Sigma[Inner] = 153.551970509448
    dt= 15329508583.307396 timestep= 19227 max Sigma= 1201.5788286022087 total t = 175889310992028.88 j = 92 Sigma[Inner] = 153.42420125966876
    dt= 15332924879.432137 timestep= 19237 max Sigma= 1201.5341775460206 total t = 176042624349579.5 j = 92 Sigma[Inner] = 153.29652991411308
    dt= 15337060255.076796 timestep= 19247 max Sigma= 1201.4908330716144 total t = 176195975686920.12 j = 92 Sigma[Inner] = 153.16895150356456
    dt= 15342030964.108585 timestep= 19257 max Sigma= 1201.4483510525415 total t = 176349372915057.97 j = 92 Sigma[Inner] = 153.04145964240632
    dt= 15347870996.751354 timestep= 19267 max Sigma= 1201.4063202603074 total t = 176502824629660.47 j = 92 Sigma[Inner] = 152.91404736099966
    dt= 15354560191.961601 timestep= 19277 max Sigma= 1201.364371274515 total t = 176656339447772.9 j = 92 Sigma[Inner] = 152.78670768044282
    dt= 15362045990.434101 timestep= 19287 max Sigma= 1201.322179317267 total t = 176809925591403.97 j = 92 Sigma[Inner] = 152.65943397412988
    dt= 15370258938.29032 timestep= 19297 max Sigma= 1201.2794635072833 total t = 176963590653569.88 j = 92 Sigma[Inner] = 152.5322201709532
    dt= 15379123070.220644 timestep= 19307 max Sigma= 1201.2359841270008 total t = 177117341490136.8 j = 92 Sigma[Inner] = 152.4050608490076
    dt= 15388562463.136753 timestep= 19317 max Sigma= 1201.1915389056091 total t = 177271184193410.22 j = 92 Sigma[Inner] = 152.27795125799693
    dt= 15398505088.688057 timestep= 19327 max Sigma= 1201.1459589364754 total t = 177425124115572.78 j = 92 Sigma[Inner] = 152.15088729808954
    dt= 15408884844.551998 timestep= 19337 max Sigma= 1201.0991045984315 total t = 177579165920008.16 j = 92 Sigma[Inner] = 152.02386547435933
    dt= 15419642407.619852 timestep= 19347 max Sigma= 1201.0508616905317 total t = 177733313646014.53 j = 92 Sigma[Inner] = 151.8968828394584
    dt= 15430725358.973696 timestep= 19357 max Sigma= 1201.0011378884888 total t = 177887570777741.0 j = 92 Sigma[Inner] = 151.76993693252567
    dt= 15442087884.780931 timestep= 19367 max Sigma= 1200.949859567821 total t = 178041940311838.56 j = 92 Sigma[Inner] = 151.6430257191473
    dt= 15453690252.447786 timestep= 19377 max Sigma= 1200.8969690004976 total t = 178196424820754.34 j = 92 Sigma[Inner] = 151.51614753506007
    dt= 15465498188.562717 timestep= 19387 max Sigma= 1200.8424219096912 total t = 178351026510165.62 j = 92 Sigma[Inner] = 151.38930103491987
    dt= 15477482235.907892 timestep= 19397 max Sigma= 1200.7861853554068 total t = 178505747270030.1 j = 92 Sigma[Inner] = 151.26248514660534
    dt= 15489617134.284819 timestep= 19407 max Sigma= 1200.7282359183546 total t = 178660588719310.6 j = 92 Sigma[Inner] = 151.135699031019
    dt= 15501881248.922888 timestep= 19417 max Sigma= 1200.6685581480529 total t = 178815552244756.16 j = 92 Sigma[Inner] = 151.00894204706253
    dt= 15514256057.076326 timestep= 19427 max Sigma= 1200.6071432420877 total t = 178970639034282.72 j = 92 Sigma[Inner] = 150.88221372132728
    dt= 15526725695.436417 timestep= 19437 max Sigma= 1200.5439879256917 total t = 179125850105554.28 j = 92 Sigma[Inner] = 150.75551372198612
    dt= 15540535656.879797 timestep= 19448 max Sigma= 1200.472508577506 total t = 179296726866021.4 j = 92 Sigma[Inner] = 150.616176188956
    dt= 15553161472.565302 timestep= 19458 max Sigma= 1200.4103962854385 total t = 179452201616090.0 j = 92 Sigma[Inner] = 150.48953510223222
    dt= 15565806865.26255 timestep= 19468 max Sigma= 1200.3605702772252 total t = 179607802807178.4 j = 92 Sigma[Inner] = 150.36292208436853
    dt= 15578355463.109655 timestep= 19478 max Sigma= 1200.3191960198988 total t = 179763530024638.62 j = 92 Sigma[Inner] = 150.23633786813323
    dt= 15590695011.022367 timestep= 19488 max Sigma= 1200.2829486515818 total t = 179919381654939.03 j = 92 Sigma[Inner] = 150.10978415148105
    dt= 15602757116.627144 timestep= 19498 max Sigma= 1200.2493930066998 total t = 180075355191050.34 j = 92 Sigma[Inner] = 149.98326337598343
    dt= 15614519507.86237 timestep= 19508 max Sigma= 1200.2166949659106 total t = 180231447701563.5 j = 92 Sigma[Inner] = 149.8567783436946
    


    
![png](output_5_157.png)
    



    
![png](output_5_158.png)
    



    
![png](output_5_159.png)
    


    dt= 15625995936.932049 timestep= 19518 max Sigma= 1200.183126307481 total t = 180387656239803.53 j = 92 Sigma[Inner] = 149.73033187506056
    dt= 15637225825.128447 timestep= 19528 max Sigma= 1200.147386666354 total t = 180543978145871.5 j = 92 Sigma[Inner] = 149.6039265554284
    dt= 15648263947.69233 timestep= 19538 max Sigma= 1200.108671599051 total t = 180700411247338.28 j = 92 Sigma[Inner] = 149.47756456514787
    dt= 15659170517.872652 timestep= 19548 max Sigma= 1200.066510173231 total t = 180856953956718.4 j = 92 Sigma[Inner] = 149.35124759446592
    dt= 15670003856.499973 timestep= 19558 max Sigma= 1200.0206535558436 total t = 181013605283048.56 j = 92 Sigma[Inner] = 149.22497683039666
    dt= 15680815965.604225 timestep= 19568 max Sigma= 1199.9709995954372 total t = 181170364786396.84 j = 92 Sigma[Inner] = 149.09875299208986
    dt= 15691650437.215439 timestep= 19578 max Sigma= 1199.9175417588153 total t = 181327232501554.4 j = 92 Sigma[Inner] = 148.97257639276975
    dt= 15702541961.6042 timestep= 19588 max Sigma= 1199.860334578234 total t = 181484208850197.0 j = 92 Sigma[Inner] = 148.84644701195643
    dt= 15713516800.4953 timestep= 19598 max Sigma= 1199.799470314267 total t = 181641294553892.12 j = 92 Sigma[Inner] = 148.72036456744817
    dt= 15724593759.737442 timestep= 19608 max Sigma= 1199.7350632516157 total t = 181798490554899.47 j = 92 Sigma[Inner] = 148.59432858111506
    dt= 15735785353.422924 timestep= 19618 max Sigma= 1199.6672391971315 total t = 181955797947949.4 j = 92 Sigma[Inner] = 148.4683384357464
    dt= 15747098973.4837 timestep= 19628 max Sigma= 1199.5961285262874 total t = 182113217923805.4 j = 92 Sigma[Inner] = 148.34239342221505
    dt= 15758537964.150776 timestep= 19638 max Sigma= 1199.5218616498898 total t = 182270751724055.56 j = 92 Sigma[Inner] = 148.21649277738587
    dt= 15770102556.03311 timestep= 19648 max Sigma= 1199.444566129452 total t = 182428400605897.66 j = 92 Sigma[Inner] = 148.0906357137783
    dt= 15781790648.026812 timestep= 19658 max Sigma= 1199.364364912384 total t = 182586165815427.8 j = 92 Sigma[Inner] = 147.96482144221372
    dt= 15793598443.834057 timestep= 19668 max Sigma= 1199.2813753239473 total t = 182744048567938.8 j = 92 Sigma[Inner] = 147.83904918868592
    dt= 15805520958.909214 timestep= 19678 max Sigma= 1199.1957085664983 total t = 182902050033855.2 j = 92 Sigma[Inner] = 147.7133182065929
    dt= 15817552416.926384 timestep= 19688 max Sigma= 1199.1074695545324 total t = 183060171329114.88 j = 92 Sigma[Inner] = 147.58762778531653
    dt= 15829686554.888147 timestep= 19698 max Sigma= 1199.0167569678172 total t = 183218413508996.84 j = 92 Sigma[Inner] = 147.46197725597634
    dt= 15841916854.36247 timestep= 19708 max Sigma= 1198.9236634419594 total t = 183376777564580.9 j = 92 Sigma[Inner] = 147.3363659950283
    dt= 15854236714.006168 timestep= 19718 max Sigma= 1198.8282758414143 total t = 183535264421184.97 j = 92 Sigma[Inner] = 147.21079342624623
    dt= 15866639576.059986 timestep= 19728 max Sigma= 1198.7306755776901 total t = 183693874938264.88 j = 92 Sigma[Inner] = 147.08525902150453
    dt= 15879119017.168993 timestep= 19738 max Sigma= 1198.6309389477801 total t = 183852609910375.6 j = 92 Sigma[Inner] = 146.9597623006888
    dt= 15891668811.827238 timestep= 19748 max Sigma= 1198.5291374763576 total t = 184011470068883.9 j = 92 Sigma[Inner] = 146.83430283098295
    dt= 15904282975.006489 timestep= 19758 max Sigma= 1198.4253382511347 total t = 184170456084196.12 j = 92 Sigma[Inner] = 146.7088802257202
    dt= 15916955789.098104 timestep= 19768 max Sigma= 1198.319604244806 total t = 184329568568322.12 j = 92 Sigma[Inner] = 146.58349414293951
    dt= 15929681819.14384 timestep= 19778 max Sigma= 1198.21199461977 total t = 184488808077641.78 j = 92 Sigma[Inner] = 146.45814428375124
    dt= 15942455919.415615 timestep= 19788 max Sigma= 1198.1025650136717 total t = 184648175115773.56 j = 92 Sigma[Inner] = 146.33283039058816
    dt= 15955273233.684488 timestep= 19798 max Sigma= 1197.991367805067 total t = 184807670136472.88 j = 92 Sigma[Inner] = 146.2075522453968
    dt= 15968129190.959843 timestep= 19808 max Sigma= 1197.8784523593183 total t = 184967293546506.62 j = 92 Sigma[Inner] = 146.08230966780812
    dt= 15981019498.046274 timestep= 19818 max Sigma= 1197.7638652553544 total t = 185127045708466.75 j = 92 Sigma[Inner] = 145.957102513315
    dt= 15993940129.933577 timestep= 19828 max Sigma= 1197.6476504942382 total t = 185286926943496.94 j = 92 Sigma[Inner] = 145.83193067147474
    dt= 16006887318.77979 timestep= 19838 max Sigma= 1197.5298496906505 total t = 185446937533914.66 j = 92 Sigma[Inner] = 145.7067940641495
    dt= 16019857542.053534 timestep= 19848 max Sigma= 1197.4105022485005 total t = 185607077725718.3 j = 92 Sigma[Inner] = 145.58169264379137
    dt= 16032847510.25345 timestep= 19858 max Sigma= 1197.2896455218488 total t = 185767347730972.78 j = 92 Sigma[Inner] = 145.45662639177792
    dt= 16045854154.511078 timestep= 19868 max Sigma= 1197.1673149623464 total t = 185927747730071.56 j = 92 Sigma[Inner] = 145.3315953168001
    dt= 16058874614.298082 timestep= 19878 max Sigma= 1197.0435442543076 total t = 186088277873875.5 j = 92 Sigma[Inner] = 145.20659945330323
    dt= 16071906225.395098 timestep= 19888 max Sigma= 1196.918365438492 total t = 186248938285729.56 j = 92 Sigma[Inner] = 145.08163885998158
    dt= 16084946508.231094 timestep= 19898 max Sigma= 1196.791809025584 total t = 186409729063361.62 j = 92 Sigma[Inner] = 144.95671361832606
    dt= 16097876389.895655 timestep= 19908 max Sigma= 1196.6639041289975 total t = 186570650127933.2 j = 92 Sigma[Inner] = 144.8318238591259
    dt= 16110129152.135237 timestep= 19918 max Sigma= 1196.534682292288 total t = 186731696264531.2 j = 92 Sigma[Inner] = 144.70697334915272
    dt= 16122402235.611254 timestep= 19928 max Sigma= 1196.4041727012404 total t = 186892865041965.2 j = 92 Sigma[Inner] = 144.58216466562797
    dt= 16134693927.01345 timestep= 19938 max Sigma= 1196.2724007234674 total t = 187054156653959.0 j = 92 Sigma[Inner] = 144.45739780715013
    dt= 16147002589.757687 timestep= 19948 max Sigma= 1196.1393906255582 total t = 187215571277528.34 j = 92 Sigma[Inner] = 144.3326727884223
    dt= 16159326661.93274 timestep= 19958 max Sigma= 1196.0051656427029 total t = 187377109073735.22 j = 92 Sigma[Inner] = 144.20798963951842
    dt= 16171664654.434664 timestep= 19968 max Sigma= 1195.8697480428905 total t = 187538770188424.7 j = 92 Sigma[Inner] = 144.08334840513407
    dt= 16184015148.899357 timestep= 19978 max Sigma= 1195.7331591861582 total t = 187700554752941.53 j = 92 Sigma[Inner] = 143.9587491438354
    dt= 16196376795.48872 timestep= 19988 max Sigma= 1195.595419579328 total t = 187862462884825.62 j = 92 Sigma[Inner] = 143.83419192731114
    dt= 16208748310.576126 timestep= 19998 max Sigma= 1195.4565489266274 total t = 188024494688484.66 j = 92 Sigma[Inner] = 143.70967683963136
    dt= 16221128474.3687 timestep= 20008 max Sigma= 1195.3165661765524 total t = 188186650255843.03 j = 92 Sigma[Inner] = 143.58520397651432
    


    
![png](output_5_161.png)
    



    
![png](output_5_162.png)
    



    
![png](output_5_163.png)
    


    dt= 16233516128.496902 timestep= 20018 max Sigma= 1195.175489565295 total t = 188348929666967.12 j = 92 Sigma[Inner] = 143.46077344460596
    dt= 16245910173.596346 timestep= 20028 max Sigma= 1195.0333366570303 total t = 188511332990666.16 j = 92 Sigma[Inner] = 143.33638536077143
    dt= 16258309566.901936 timestep= 20038 max Sigma= 1194.8901243813245 total t = 188673860285068.88 j = 92 Sigma[Inner] = 143.21203985140272
    dt= 16270713319.870375 timestep= 20048 max Sigma= 1194.7458690679111 total t = 188836511598176.56 j = 92 Sigma[Inner] = 143.08773705174193
    dt= 16283120495.84383 timestep= 20058 max Sigma= 1194.6005864790454 total t = 188999286968392.4 j = 92 Sigma[Inner] = 142.9634771052229
    dt= 16295530207.76459 timestep= 20068 max Sigma= 1194.4542918396444 total t = 189162186425027.6 j = 92 Sigma[Inner] = 142.8392601628301
    dt= 16307941615.94862 timestep= 20078 max Sigma= 1194.3069998653814 total t = 189325209988785.22 j = 92 Sigma[Inner] = 142.71508638247673
    dt= 16320353925.923489 timestep= 20088 max Sigma= 1194.158724788909 total t = 189488357672221.9 j = 92 Sigma[Inner] = 142.59095592840143
    dt= 16332766386.334991 timestep= 20098 max Sigma= 1194.00948038435 total t = 189651629480188.34 j = 92 Sigma[Inner] = 142.46686897058402
    dt= 16345178286.925114 timestep= 20108 max Sigma= 1193.859279990199 total t = 189815025410249.4 j = 92 Sigma[Inner] = 142.34282568418038
    dt= 16357588956.583315 timestep= 20118 max Sigma= 1193.7081365307508 total t = 189978545453084.25 j = 92 Sigma[Inner] = 142.21882624897603
    dt= 16369997761.471727 timestep= 20128 max Sigma= 1193.5560625361773 total t = 190142189592867.44 j = 92 Sigma[Inner] = 142.0948708488577
    dt= 16382404103.22457 timestep= 20138 max Sigma= 1193.403070161343 total t = 190305957807631.78 j = 92 Sigma[Inner] = 141.97095967130394
    dt= 16394807417.221254 timestep= 20148 max Sigma= 1193.249171203467 total t = 190469850069613.38 j = 92 Sigma[Inner] = 141.8470929068923
    dt= 16407207170.932186 timestep= 20158 max Sigma= 1193.094377118704 total t = 190633866345579.8 j = 92 Sigma[Inner] = 141.7232707488254
    dt= 16419602862.336077 timestep= 20168 max Sigma= 1192.938699037732 total t = 190798006597142.7 j = 92 Sigma[Inner] = 141.59949339247268
    dt= 16431994018.407043 timestep= 20178 max Sigma= 1192.7821477804164 total t = 190962270781053.9 j = 92 Sigma[Inner] = 141.4757610349293
    dt= 16444380193.669678 timestep= 20188 max Sigma= 1192.6247338696123 total t = 191126658849487.88 j = 92 Sigma[Inner] = 141.35207387459175
    dt= 16456760968.820211 timestep= 20198 max Sigma= 1192.466467544172 total t = 191291170750309.38 j = 92 Sigma[Inner] = 141.22843211074843
    dt= 16469135949.411406 timestep= 20208 max Sigma= 1192.307358771208 total t = 191455806427327.78 j = 92 Sigma[Inner] = 141.10483594318708
    dt= 16481504764.599232 timestep= 20218 max Sigma= 1192.1474172576673 total t = 191620565820538.94 j = 92 Sigma[Inner] = 140.98128557181664
    dt= 16493867065.948883 timestep= 20228 max Sigma= 1191.9866524612573 total t = 191785448866354.38 j = 92 Sigma[Inner] = 140.8577811963052
    dt= 16506222526.297886 timestep= 20238 max Sigma= 1191.8250736007756 total t = 191950455497819.25 j = 92 Sigma[Inner] = 140.7343230157323
    dt= 16518570838.674088 timestep= 20248 max Sigma= 1191.6626896658759 total t = 192115585644818.9 j = 92 Sigma[Inner] = 140.6109112282568
    dt= 16530911715.26623 timestep= 20258 max Sigma= 1191.4995094263122 total t = 192280839234275.34 j = 92 Sigma[Inner] = 140.48754603079973
    dt= 16543244886.444775 timestep= 20268 max Sigma= 1191.3355414406915 total t = 192446216190333.12 j = 92 Sigma[Inner] = 140.3642276187417
    dt= 16555570099.830883 timestep= 20278 max Sigma= 1191.170794064769 total t = 192611716434536.16 j = 92 Sigma[Inner] = 140.24095618563513
    dt= 16567887119.411404 timestep= 20288 max Sigma= 1191.005275459315 total t = 192777339885995.28 j = 92 Sigma[Inner] = 140.11773192293217
    dt= 16580195724.697695 timestep= 20298 max Sigma= 1190.838993597578 total t = 192943086461547.28 j = 92 Sigma[Inner] = 139.9945550197265
    dt= 16592495709.926373 timestep= 20308 max Sigma= 1190.671956272374 total t = 193108956075905.75 j = 92 Sigma[Inner] = 139.8714256625116
    dt= 16604786883.299902 timestep= 20318 max Sigma= 1190.504171102817 total t = 193274948641804.0 j = 92 Sigma[Inner] = 139.74834403495294
    dt= 16617069066.265223 timestep= 20328 max Sigma= 1190.3356455407222 total t = 193441064070130.72 j = 92 Sigma[Inner] = 139.6253103176758
    dt= 16629342092.828526 timestep= 20338 max Sigma= 1190.166386876694 total t = 193607302270058.88 j = 92 Sigma[Inner] = 139.50232468806854
    dt= 16641605808.904503 timestep= 20348 max Sigma= 1189.9964022459221 total t = 193773663149167.34 j = 92 Sigma[Inner] = 139.37938732010136
    dt= 16653860071.698341 timestep= 20358 max Sigma= 1189.825698633699 total t = 193940146613556.75 j = 92 Sigma[Inner] = 139.2564983841599
    dt= 16666104749.118782 timestep= 20368 max Sigma= 1189.6542828806805 total t = 194106752567959.3 j = 92 Sigma[Inner] = 139.13365804689477
    dt= 16678339719.220898 timestep= 20378 max Sigma= 1189.4821616878974 total t = 194273480915842.34 j = 92 Sigma[Inner] = 139.0108664710866
    dt= 16690564869.676891 timestep= 20388 max Sigma= 1189.3093416215404 total t = 194440331559507.22 j = 92 Sigma[Inner] = 138.8881238155261
    dt= 16702780097.273672 timestep= 20398 max Sigma= 1189.135829117522 total t = 194607304400182.44 j = 92 Sigma[Inner] = 138.7654302349103
    dt= 16714985307.435799 timestep= 20408 max Sigma= 1188.9616304858362 total t = 194774399338112.28 j = 92 Sigma[Inner] = 138.64278587975315
    dt= 16727180413.772457 timestep= 20418 max Sigma= 1188.786751914723 total t = 194941616272640.75 j = 92 Sigma[Inner] = 138.5201908963127
    dt= 16739365337.647451 timestep= 20428 max Sigma= 1188.6111994746464 total t = 195108955102290.84 j = 92 Sigma[Inner] = 138.3976454265322
    dt= 16751540007.770758 timestep= 20438 max Sigma= 1188.4349791221027 total t = 195276415724839.88 j = 92 Sigma[Inner] = 138.27514960799655
    dt= 16763704359.810833 timestep= 20448 max Sigma= 1188.2580967032618 total t = 195443998037391.0 j = 92 Sigma[Inner] = 138.1527035739044
    dt= 16775858336.026402 timestep= 20458 max Sigma= 1188.0805579574524 total t = 195611701936440.5 j = 92 Sigma[Inner] = 138.0303074530535
    dt= 16788001884.916857 timestep= 20468 max Sigma= 1187.9023685205013 total t = 195779527317941.84 j = 92 Sigma[Inner] = 137.90796136984102
    dt= 16800134960.890265 timestep= 20478 max Sigma= 1187.723533927932 total t = 195947474077366.34 j = 92 Sigma[Inner] = 137.7856654442779
    dt= 16812257523.948122 timestep= 20488 max Sigma= 1187.5440596180308 total t = 196115542109760.3 j = 92 Sigma[Inner] = 137.6634197920166
    dt= 16824369539.385963 timestep= 20498 max Sigma= 1187.363950934784 total t = 196283731309799.5 j = 92 Sigma[Inner] = 137.54122452439248
    dt= 16836470977.509016 timestep= 20508 max Sigma= 1187.183213130703 total t = 196452041571840.3 j = 92 Sigma[Inner] = 137.41907974847737
    


    
![png](output_5_165.png)
    



    
![png](output_5_166.png)
    



    
![png](output_5_167.png)
    


    dt= 16848561813.362226 timestep= 20518 max Sigma= 1187.0018513695275 total t = 196620472789968.6 j = 92 Sigma[Inner] = 137.2969855671464
    dt= 16860642026.473734 timestep= 20528 max Sigma= 1186.819870728828 total t = 196789024858045.4 j = 92 Sigma[Inner] = 137.17494207915638
    dt= 16872711600.611296 timestep= 20538 max Sigma= 1186.6372762024994 total t = 196957697669750.7 j = 92 Sigma[Inner] = 137.0529493792357
    dt= 16884770523.550915 timestep= 20548 max Sigma= 1186.4540727031617 total t = 197126491118624.3 j = 92 Sigma[Inner] = 136.93100755818566
    dt= 16896818786.857044 timestep= 20558 max Sigma= 1186.2702650644671 total t = 197295405098104.72 j = 92 Sigma[Inner] = 136.80911670299182
    dt= 16908856385.673779 timestep= 20568 max Sigma= 1186.0858580433164 total t = 197464439501566.1 j = 92 Sigma[Inner] = 136.68727689694558
    dt= 16920883318.52655 timestep= 20578 max Sigma= 1185.9008563219923 total t = 197633594222352.72 j = 92 Sigma[Inner] = 136.56548821977506
    dt= 16932899587.133684 timestep= 20588 max Sigma= 1185.7152645102146 total t = 197802869153811.78 j = 92 Sigma[Inner] = 136.44375074778526
    dt= 16944905196.227316 timestep= 20598 max Sigma= 1185.5290871471157 total t = 197972264189324.53 j = 92 Sigma[Inner] = 136.32206455400595
    dt= 16956900153.383402 timestep= 20608 max Sigma= 1185.3423287031449 total t = 198141779222335.16 j = 92 Sigma[Inner] = 136.20042970834777
    dt= 16968884468.860012 timestep= 20618 max Sigma= 1185.154993581904 total t = 198311414146378.62 j = 92 Sigma[Inner] = 136.0788462777652
    dt= 16980858155.44383 timestep= 20628 max Sigma= 1184.9670861219126 total t = 198481168855106.34 j = 92 Sigma[Inner] = 135.9573143264261
    dt= 16992821228.304192 timestep= 20638 max Sigma= 1184.7786105983157 total t = 198651043242311.03 j = 92 Sigma[Inner] = 135.8358339158874
    dt= 17004773704.854464 timestep= 20648 max Sigma= 1184.589571224525 total t = 198821037201949.3 j = 92 Sigma[Inner] = 135.7144051052764
    dt= 17016715604.620255 timestep= 20658 max Sigma= 1184.399972153805 total t = 198991150628163.8 j = 92 Sigma[Inner] = 135.59302795147607
    dt= 17028646949.114231 timestep= 20668 max Sigma= 1184.209817480805 total t = 199161383415303.47 j = 92 Sigma[Inner] = 135.47170250931566
    dt= 17040567761.717052 timestep= 20678 max Sigma= 1184.0191112430334 total t = 199331735457942.62 j = 92 Sigma[Inner] = 135.35042883176428
    dt= 17052478067.564344 timestep= 20688 max Sigma= 1183.8278574222875 total t = 199502206650899.25 j = 92 Sigma[Inner] = 135.22920697012765
    dt= 17064377893.439236 timestep= 20698 max Sigma= 1183.6360599460293 total t = 199672796889251.7 j = 92 Sigma[Inner] = 135.10803697424817
    dt= 17076267267.670166 timestep= 20708 max Sigma= 1183.4437226887164 total t = 199843506068354.88 j = 92 Sigma[Inner] = 134.9869188927057
    dt= 17088146220.033821 timestep= 20718 max Sigma= 1183.2508494730914 total t = 200014334083854.94 j = 92 Sigma[Inner] = 134.86585277302146
    dt= 17100014781.662905 timestep= 20728 max Sigma= 1183.0574440714252 total t = 200185280831703.38 j = 92 Sigma[Inner] = 134.7448386618619
    dt= 17111872984.958456 timestep= 20738 max Sigma= 1182.8635102067171 total t = 200356346208170.12 j = 92 Sigma[Inner] = 134.62387660524325
    dt= 17123720863.506462 timestep= 20748 max Sigma= 1182.6690515538642 total t = 200527530109855.53 j = 92 Sigma[Inner] = 134.5029666487365
    dt= 17135558451.998749 timestep= 20758 max Sigma= 1182.4740717407826 total t = 200698832433702.03 j = 92 Sigma[Inner] = 134.38210883767172
    dt= 17147385786.157677 timestep= 20768 max Sigma= 1182.2785743495015 total t = 200870253077004.75 j = 92 Sigma[Inner] = 134.26130321734166
    dt= 17159202902.664577 timestep= 20778 max Sigma= 1182.082562917215 total t = 201041791937421.28 j = 92 Sigma[Inner] = 134.14054983320415
    dt= 17171009839.091862 timestep= 20788 max Sigma= 1181.8860409373083 total t = 201213448912981.12 j = 92 Sigma[Inner] = 134.0198487310827
    dt= 17182806633.83836 timestep= 20798 max Sigma= 1181.689011860343 total t = 201385223902093.97 j = 92 Sigma[Inner] = 133.89919995736523
    dt= 17194593326.068016 timestep= 20808 max Sigma= 1181.4914790950206 total t = 201557116803557.8 j = 92 Sigma[Inner] = 133.77860355920046
    dt= 17206369955.65153 timestep= 20818 max Sigma= 1181.2934460091092 total t = 201729127516566.3 j = 92 Sigma[Inner] = 133.65805958469136
    dt= 17218136563.110924 timestep= 20828 max Sigma= 1181.0949159303461 total t = 201901255940715.22 j = 92 Sigma[Inner] = 133.5375680830858
    dt= 17229893189.56693 timestep= 20838 max Sigma= 1180.8958921473125 total t = 202073501976009.03 j = 92 Sigma[Inner] = 133.41712910496352
    dt= 17241639876.689045 timestep= 20848 max Sigma= 1180.6963779102794 total t = 202245865522866.5 j = 92 Sigma[Inner] = 133.29674270242003
    dt= 17253376666.64789 timestep= 20858 max Sigma= 1180.4963764320314 total t = 202418346482125.78 j = 92 Sigma[Inner] = 133.17640892924604
    dt= 17265103602.070232 timestep= 20868 max Sigma= 1180.295890888664 total t = 202590944755049.38 j = 92 Sigma[Inner] = 133.05612784110292
    dt= 17276820725.996037 timestep= 20878 max Sigma= 1180.0949244203573 total t = 202763660243328.3 j = 92 Sigma[Inner] = 132.93589949569395
    dt= 17288528081.83781 timestep= 20888 max Sigma= 1179.8934801321286 total t = 202936492849086.06 j = 92 Sigma[Inner] = 132.81572395293102
    dt= 17300225713.341984 timestep= 20898 max Sigma= 1179.6915610945628 total t = 203109442474882.16 j = 92 Sigma[Inner] = 132.69560127509618
    dt= 17311913664.55218 timestep= 20908 max Sigma= 1179.489170344521 total t = 203282509023715.28 j = 92 Sigma[Inner] = 132.57553152699862
    dt= 17323591979.774464 timestep= 20918 max Sigma= 1179.2863108858287 total t = 203455692399025.88 j = 92 Sigma[Inner] = 132.45551477612668
    dt= 17335260703.544224 timestep= 20928 max Sigma= 1179.0829856899459 total t = 203628992504698.84 j = 92 Sigma[Inner] = 132.33555109279408
    dt= 17346919880.594868 timestep= 20938 max Sigma= 1178.8791976966172 total t = 203802409245065.62 j = 92 Sigma[Inner] = 132.21564055028145
    dt= 17358569555.828068 timestep= 20948 max Sigma= 1178.6749498145043 total t = 203975942524905.78 j = 92 Sigma[Inner] = 132.09578322497185
    dt= 17370209774.28551 timestep= 20958 max Sigma= 1178.4702449218005 total t = 204149592249448.84 j = 92 Sigma[Inner] = 131.97597919648132
    dt= 17381840581.12217 timestep= 20968 max Sigma= 1178.2650858668287 total t = 204323358324375.22 j = 92 Sigma[Inner] = 131.85622854778333
    dt= 17393462021.580894 timestep= 20978 max Sigma= 1178.059475468623 total t = 204497240655817.53 j = 92 Sigma[Inner] = 131.73653136532786
    dt= 17405074140.968407 timestep= 20988 max Sigma= 1177.8534165174942 total t = 204671239150361.12 j = 92 Sigma[Inner] = 131.61688773915466
    dt= 17416676984.632458 timestep= 20998 max Sigma= 1177.6469117755796 total t = 204845353715044.72 j = 92 Sigma[Inner] = 131.49729776300074
    dt= 17428270597.940308 timestep= 21008 max Sigma= 1177.4399639773796 total t = 205019584257360.6 j = 92 Sigma[Inner] = 131.37776153440228
    


    
![png](output_5_169.png)
    



    
![png](output_5_170.png)
    



    
![png](output_5_171.png)
    


    dt= 17439855026.258167 timestep= 21018 max Sigma= 1177.2325758302804 total t = 205193930685254.62 j = 92 Sigma[Inner] = 131.25827915479053
    dt= 17451430314.931942 timestep= 21028 max Sigma= 1177.0247500150583 total t = 205368392907126.4 j = 92 Sigma[Inner] = 131.13885072958217
    dt= 17462996509.26885 timestep= 21038 max Sigma= 1176.8164891863773 total t = 205542970831828.78 j = 92 Sigma[Inner] = 131.01947636826372
    dt= 17474553654.520115 timestep= 21048 max Sigma= 1176.6077959732702 total t = 205717664368667.2 j = 92 Sigma[Inner] = 130.90015618447032
    dt= 17486101795.864506 timestep= 21058 max Sigma= 1176.3986729796056 total t = 205892473427399.44 j = 92 Sigma[Inner] = 130.7808902960589
    dt= 17497640978.39297 timestep= 21068 max Sigma= 1176.1891227845479 total t = 206067397918234.44 j = 92 Sigma[Inner] = 130.66167882517556
    dt= 17509171247.09385 timestep= 21078 max Sigma= 1175.9791479430003 total t = 206242437751831.62 j = 92 Sigma[Inner] = 130.54252189831763
    dt= 17520692646.83912 timestep= 21088 max Sigma= 1175.768750986041 total t = 206417592839299.56 j = 92 Sigma[Inner] = 130.42341964638962
    dt= 17532205222.371338 timestep= 21098 max Sigma= 1175.5579344213445 total t = 206592863092194.97 j = 92 Sigma[Inner] = 130.3043722047544
    dt= 17543709018.291325 timestep= 21108 max Sigma= 1175.3467007335955 total t = 206768248422521.03 j = 92 Sigma[Inner] = 130.18537971327868
    dt= 17555204079.046528 timestep= 21118 max Sigma= 1175.1350523848914 total t = 206943748742726.3 j = 92 Sigma[Inner] = 130.0664423163729
    dt= 17566690448.920105 timestep= 21128 max Sigma= 1174.9229918151332 total t = 207119363965702.84 j = 92 Sigma[Inner] = 129.9475601630265
    dt= 17578168172.020634 timestep= 21138 max Sigma= 1174.7105214424105 total t = 207295094004784.5 j = 92 Sigma[Inner] = 129.82873340683742
    dt= 17589637292.27241 timestep= 21148 max Sigma= 1174.4976436633738 total t = 207470938773745.34 j = 92 Sigma[Inner] = 129.7099622060373
    dt= 17601097853.406334 timestep= 21158 max Sigma= 1174.2843608535993 total t = 207646898186797.6 j = 92 Sigma[Inner] = 129.5912467235108
    dt= 17612549898.951366 timestep= 21168 max Sigma= 1174.070675367944 total t = 207822972158589.56 j = 92 Sigma[Inner] = 129.47258712681116
    dt= 17623993472.226482 timestep= 21178 max Sigma= 1173.856589540893 total t = 207999160604203.84 j = 92 Sigma[Inner] = 129.35398358817005
    dt= 17635428616.333164 timestep= 21188 max Sigma= 1173.6421056869005 total t = 208175463439155.03 j = 92 Sigma[Inner] = 129.2354362845038
    dt= 17646855374.14834 timestep= 21198 max Sigma= 1173.427226100718 total t = 208351880579387.38 j = 92 Sigma[Inner] = 129.1169453974139
    dt= 17658273788.317802 timestep= 21208 max Sigma= 1173.2119530577188 total t = 208528411941272.78 j = 92 Sigma[Inner] = 128.9985111131847
    dt= 17669683901.250053 timestep= 21218 max Sigma= 1172.9962888142143 total t = 208705057441608.25 j = 92 Sigma[Inner] = 128.88013362277513
    dt= 17681085755.11056 timestep= 21228 max Sigma= 1172.7802356077616 total t = 208881816997613.4 j = 92 Sigma[Inner] = 128.761813121808
    dt= 17692479391.816372 timestep= 21238 max Sigma= 1172.5637956574637 total t = 209058690526928.34 j = 92 Sigma[Inner] = 128.64354981055396
    dt= 17703864853.03122 timestep= 21248 max Sigma= 1172.3469711642674 total t = 209235677947610.88 j = 92 Sigma[Inner] = 128.52534389391192
    dt= 17715242180.16082 timestep= 21258 max Sigma= 1172.1297643112468 total t = 209412779178134.0 j = 92 Sigma[Inner] = 128.4071955813865
    dt= 17726611414.348656 timestep= 21268 max Sigma= 1171.9121772638875 total t = 209589994137383.4 j = 92 Sigma[Inner] = 128.28910508706068
    dt= 17737972596.47203 timestep= 21278 max Sigma= 1171.6942121703598 total t = 209767322744654.66 j = 92 Sigma[Inner] = 128.1710726295657
    dt= 17749325767.138367 timestep= 21288 max Sigma= 1171.4758711617885 total t = 209944764919650.78 j = 92 Sigma[Inner] = 128.0530984320473
    dt= 17760670966.681923 timestep= 21298 max Sigma= 1171.2571563525153 total t = 210122320582479.28 j = 92 Sigma[Inner] = 127.93518272212881
    dt= 17772008235.1607 timestep= 21308 max Sigma= 1171.0380698403553 total t = 210299989653649.4 j = 92 Sigma[Inner] = 127.81732573187107
    dt= 17783337612.35364 timestep= 21318 max Sigma= 1170.8186137068499 total t = 210477772054069.56 j = 92 Sigma[Inner] = 127.69952769772921
    dt= 17794659137.7581 timestep= 21328 max Sigma= 1170.59879001751 total t = 210655667705044.3 j = 92 Sigma[Inner] = 127.58178886050693
    dt= 17805972850.587513 timestep= 21338 max Sigma= 1170.3786008220598 total t = 210833676528271.78 j = 92 Sigma[Inner] = 127.46410946530749
    dt= 17817278789.76933 timestep= 21348 max Sigma= 1170.1580481546694 total t = 211011798445840.47 j = 92 Sigma[Inner] = 127.34648976148259
    dt= 17828576993.94318 timestep= 21358 max Sigma= 1169.9371340341863 total t = 211190033380226.66 j = 92 Sigma[Inner] = 127.2289300025783
    dt= 17839867501.459213 timestep= 21368 max Sigma= 1169.7158604643616 total t = 211368381254291.47 j = 92 Sigma[Inner] = 127.11143044627892
    dt= 17851150350.37662 timestep= 21378 max Sigma= 1169.4942294340683 total t = 211546841991277.8 j = 92 Sigma[Inner] = 126.99399135434844
    dt= 17862425578.462433 timestep= 21388 max Sigma= 1169.27224291752 total t = 211725415514807.78 j = 92 Sigma[Inner] = 126.87661299256969
    dt= 17873693223.19038 timestep= 21398 max Sigma= 1169.049902874481 total t = 211904101748879.28 j = 92 Sigma[Inner] = 126.75929563068156
    dt= 17884953321.739983 timestep= 21408 max Sigma= 1168.8272112504724 total t = 212082900617863.53 j = 92 Sigma[Inner] = 126.64203954231456
    dt= 17896205910.995853 timestep= 21418 max Sigma= 1168.604169976977 total t = 212261812046501.84 j = 92 Sigma[Inner] = 126.5248450049241
    dt= 17907451027.547066 timestep= 21428 max Sigma= 1168.380780971636 total t = 212440835959902.84 j = 92 Sigma[Inner] = 126.4077122997221
    dt= 17918688707.686657 timestep= 21438 max Sigma= 1168.157046138445 total t = 212619972283539.22 j = 92 Sigma[Inner] = 126.29064171160756
    dt= 17929918987.411407 timestep= 21448 max Sigma= 1167.932967367943 total t = 212799220943245.2 j = 92 Sigma[Inner] = 126.17363352909486
    dt= 17941045180.586994 timestep= 21458 max Sigma= 1167.7085466435728 total t = 212978581683675.62 j = 92 Sigma[Inner] = 126.05668809951686
    dt= 17951897368.548702 timestep= 21468 max Sigma= 1167.4837888755778 total t = 213158051830390.97 j = 92 Sigma[Inner] = 125.93980730074533
    dt= 17962740037.16278 timestep= 21478 max Sigma= 1167.2586973381658 total t = 213337630446585.06 j = 92 Sigma[Inner] = 125.82299215951883
    dt= 17973573240.143173 timestep= 21488 max Sigma= 1167.0332739094945 total t = 213517317437360.28 j = 92 Sigma[Inner] = 125.70624298414668
    dt= 17984397030.639343 timestep= 21498 max Sigma= 1166.8075204547815 total t = 213697112708352.94 j = 92 Sigma[Inner] = 125.58956008598092
    dt= 17995211461.212677 timestep= 21508 max Sigma= 1166.581438826479 total t = 213877016165727.88 j = 92 Sigma[Inner] = 125.47294377933981
    


    
![png](output_5_173.png)
    



    
![png](output_5_174.png)
    



    
![png](output_5_175.png)
    


    dt= 18006016583.842983 timestep= 21518 max Sigma= 1166.3550308644394 total t = 214057027716172.2 j = 92 Sigma[Inner] = 125.35639438142913
    dt= 18016812449.93491 timestep= 21528 max Sigma= 1166.1282983960832 total t = 214237147266889.72 j = 92 Sigma[Inner] = 125.23991221226349
    dt= 18027599110.324352 timestep= 21538 max Sigma= 1165.9012432365623 total t = 214417374725595.06 j = 92 Sigma[Inner] = 125.12349759458685
    dt= 18038376615.284813 timestep= 21548 max Sigma= 1165.6738671889168 total t = 214597710000508.28 j = 92 Sigma[Inner] = 125.00715085379301
    dt= 18049145014.533638 timestep= 21558 max Sigma= 1165.4461720442334 total t = 214778153000348.88 j = 92 Sigma[Inner] = 124.89087231784518
    dt= 18059904357.238297 timestep= 21568 max Sigma= 1165.218159581798 total t = 214958703634330.6 j = 92 Sigma[Inner] = 124.7746623171956
    dt= 18070654692.02252 timestep= 21578 max Sigma= 1164.9898315692465 total t = 215139361812155.94 j = 92 Sigma[Inner] = 124.65852118470482
    dt= 18081396066.97242 timestep= 21588 max Sigma= 1164.7611897627112 total t = 215320127444010.56 j = 92 Sigma[Inner] = 124.54244925556017
    dt= 18092128529.642548 timestep= 21598 max Sigma= 1164.5322359069676 total t = 215501000440558.12 j = 92 Sigma[Inner] = 124.42644686719471
    dt= 18102852127.0619 timestep= 21608 max Sigma= 1164.302971735575 total t = 215681980712934.94 j = 92 Sigma[Inner] = 124.31051435920534
    dt= 18113566905.73985 timestep= 21618 max Sigma= 1164.0733989710168 total t = 215863068172744.66 j = 92 Sigma[Inner] = 124.19465207327166
    dt= 18124272911.67203 timestep= 21628 max Sigma= 1163.8435193248367 total t = 216044262732053.38 j = 92 Sigma[Inner] = 124.07886035307367
    dt= 18134970190.346172 timestep= 21638 max Sigma= 1163.6133344977745 total t = 216225564303384.16 j = 92 Sigma[Inner] = 123.9631395442106
    dt= 18145658786.74781 timestep= 21648 max Sigma= 1163.382846179896 total t = 216406972799712.25 j = 92 Sigma[Inner] = 123.84748999411916
    dt= 18156338745.366062 timestep= 21658 max Sigma= 1163.1520560507247 total t = 216588488134460.1 j = 92 Sigma[Inner] = 123.73191205199187
    dt= 18167010110.199234 timestep= 21668 max Sigma= 1162.9209657793674 total t = 216770110221492.16 j = 92 Sigma[Inner] = 123.61640606869628
    dt= 18177672924.760414 timestep= 21678 max Sigma= 1162.6895770246404 total t = 216951838975110.4 j = 92 Sigma[Inner] = 123.50097239669356
    dt= 18188327232.082977 timestep= 21688 max Sigma= 1162.4578914351907 total t = 217133674310049.16 j = 92 Sigma[Inner] = 123.38561138995797
    dt= 18198973074.726067 timestep= 21698 max Sigma= 1162.2259106496188 total t = 217315616141470.44 j = 92 Sigma[Inner] = 123.27032340389653
    dt= 18209610494.78003 timestep= 21708 max Sigma= 1161.993636296594 total t = 217497664384959.4 j = 92 Sigma[Inner] = 123.15510879526876
    dt= 18220239533.871693 timestep= 21718 max Sigma= 1161.7610699949735 total t = 217679818956519.5 j = 92 Sigma[Inner] = 123.03996792210725
    dt= 18230860233.169743 timestep= 21728 max Sigma= 1161.5282133539151 total t = 217862079772567.78 j = 92 Sigma[Inner] = 122.92490114363882
    dt= 18241472633.389854 timestep= 21738 max Sigma= 1161.2950679729904 total t = 218044446749930.72 j = 92 Sigma[Inner] = 122.80990882020556
    dt= 18252076774.799957 timestep= 21748 max Sigma= 1161.0616354422934 total t = 218226919805839.44 j = 92 Sigma[Inner] = 122.69499131318696
    dt= 18262672697.22529 timestep= 21758 max Sigma= 1160.8279173425515 total t = 218409498857925.12 j = 92 Sigma[Inner] = 122.58014898492249
    dt= 18273260440.053474 timestep= 21768 max Sigma= 1160.5939152452293 total t = 218592183824215.0 j = 92 Sigma[Inner] = 122.46538219863461
    dt= 18283840042.239525 timestep= 21778 max Sigma= 1160.359630712636 total t = 218774974623127.66 j = 92 Sigma[Inner] = 122.35069131835273
    dt= 18294411542.310757 timestep= 21788 max Sigma= 1160.1250652980261 total t = 218957871173468.94 j = 92 Sigma[Inner] = 122.23607670883762
    dt= 18304974978.3717 timestep= 21798 max Sigma= 1159.8902205457011 total t = 219140873394427.6 j = 92 Sigma[Inner] = 122.12153873550663
    dt= 18315530388.108936 timestep= 21808 max Sigma= 1159.65509799111 total t = 219323981205571.1 j = 92 Sigma[Inner] = 122.00707776435961
    dt= 18326077808.795807 timestep= 21818 max Sigma= 1159.419699160946 total t = 219507194526841.66 j = 92 Sigma[Inner] = 121.89269416190557
    dt= 18336617277.2972 timestep= 21828 max Sigma= 1159.184025573243 total t = 219690513278551.84 j = 92 Sigma[Inner] = 121.77838829509032
    dt= 18347148830.074196 timestep= 21838 max Sigma= 1158.9480787374698 total t = 219873937381380.6 j = 92 Sigma[Inner] = 121.66416053122452
    dt= 18357672503.18863 timestep= 21848 max Sigma= 1158.7118601546242 total t = 220057466756369.44 j = 92 Sigma[Inner] = 121.55001123791308
    dt= 18368188332.307705 timestep= 21858 max Sigma= 1158.4753713173232 total t = 220241101324918.22 j = 92 Sigma[Inner] = 121.43594078298489
    dt= 18378696352.708427 timestep= 21868 max Sigma= 1158.2386137098938 total t = 220424841008781.25 j = 92 Sigma[Inner] = 121.32194953442378
    dt= 18389196599.28211 timestep= 21878 max Sigma= 1158.0015888084602 total t = 220608685730063.6 j = 92 Sigma[Inner] = 121.20803786030015
    dt= 18399689106.538708 timestep= 21888 max Sigma= 1157.7642980810333 total t = 220792635411217.1 j = 92 Sigma[Inner] = 121.09420612870379
    dt= 18410173908.61121 timestep= 21898 max Sigma= 1157.526742987593 total t = 220976689975036.72 j = 92 Sigma[Inner] = 120.98045470767707
    dt= 18420651039.25987 timestep= 21908 max Sigma= 1157.2889249801758 total t = 221160849344656.53 j = 92 Sigma[Inner] = 120.86678396514985
    dt= 18431120531.876476 timestep= 21918 max Sigma= 1157.0508455029544 total t = 221345113443546.22 j = 92 Sigma[Inner] = 120.75319426887458
    dt= 18441582419.488552 timestep= 21928 max Sigma= 1156.812505992323 total t = 221529482195507.44 j = 92 Sigma[Inner] = 120.63968598636271
    dt= 18452036734.763435 timestep= 21938 max Sigma= 1156.5739078769739 total t = 221713955524670.1 j = 92 Sigma[Inner] = 120.52625948482215
    dt= 18462483510.012417 timestep= 21948 max Sigma= 1156.3350525779783 total t = 221898533355488.88 j = 92 Sigma[Inner] = 120.4129151310953
    dt= 18472922777.194725 timestep= 21958 max Sigma= 1156.0959415088644 total t = 222083215612739.5 j = 92 Sigma[Inner] = 120.29965329159867
    dt= 18483354567.92155 timestep= 21968 max Sigma= 1155.856576075693 total t = 222268002221515.53 j = 92 Sigma[Inner] = 120.1864743322629
    dt= 18493778913.459972 timestep= 21978 max Sigma= 1155.616957677134 total t = 222452893107224.7 j = 92 Sigma[Inner] = 120.073378618474
    dt= 18504195844.736824 timestep= 21988 max Sigma= 1155.3770877045386 total t = 222637888195585.44 j = 92 Sigma[Inner] = 119.96036651501572
    dt= 18514605392.342564 timestep= 21998 max Sigma= 1155.1369675420135 total t = 222822987412623.6 j = 92 Sigma[Inner] = 119.84743838601267
    dt= 18525007586.53505 timestep= 22008 max Sigma= 1154.8965985664943 total t = 223008190684669.25 j = 92 Sigma[Inner] = 119.73459459487472
    


    
![png](output_5_177.png)
    



    
![png](output_5_178.png)
    



    
![png](output_5_179.png)
    


    dt= 18535402457.243305 timestep= 22018 max Sigma= 1154.6559821478133 total t = 223193497938353.1 j = 92 Sigma[Inner] = 119.62183550424206
    dt= 18545790034.071194 timestep= 22028 max Sigma= 1154.4151196487715 total t = 223378909100603.4 j = 92 Sigma[Inner] = 119.50916147593165
    dt= 18556170346.30111 timestep= 22038 max Sigma= 1154.1740124252044 total t = 223564424098642.7 j = 92 Sigma[Inner] = 119.3965728708844
    dt= 18566543422.897556 timestep= 22048 max Sigma= 1153.932661826054 total t = 223750042859984.53 j = 92 Sigma[Inner] = 119.2840700491134
    dt= 18576909292.510742 timestep= 22058 max Sigma= 1153.691069193432 total t = 223935765312430.4 j = 92 Sigma[Inner] = 119.17165336965319
    dt= 18587267983.480076 timestep= 22068 max Sigma= 1153.4492358626853 total t = 224121591384066.6 j = 92 Sigma[Inner] = 119.05932319051031
    dt= 18597619523.83769 timestep= 22078 max Sigma= 1153.2071631624626 total t = 224307521003261.12 j = 92 Sigma[Inner] = 118.94707986861431
    dt= 18607963941.311832 timestep= 22088 max Sigma= 1152.9648524147767 total t = 224493554098660.62 j = 92 Sigma[Inner] = 118.83492375977018
    dt= 18618301263.33028 timestep= 22098 max Sigma= 1152.7223049350675 total t = 224679690599187.38 j = 92 Sigma[Inner] = 118.72285521861157
    dt= 18628631517.0237 timestep= 22108 max Sigma= 1152.479522032264 total t = 224865930434036.3 j = 92 Sigma[Inner] = 118.61087459855518
    dt= 18638954729.228966 timestep= 22118 max Sigma= 1152.2365050088433 total t = 225052273532671.94 j = 92 Sigma[Inner] = 118.49898225175606
    dt= 18649270926.4924 timestep= 22128 max Sigma= 1151.993255160894 total t = 225238719824825.62 j = 92 Sigma[Inner] = 118.38717852906375
    dt= 18659580135.07305 timestep= 22138 max Sigma= 1151.7497737781725 total t = 225425269240492.62 j = 92 Sigma[Inner] = 118.2754637799797
    dt= 18669882380.94584 timestep= 22148 max Sigma= 1151.5060621441621 total t = 225611921709929.25 j = 92 Sigma[Inner] = 118.16383835261551
    dt= 18680177689.80475 timestep= 22158 max Sigma= 1151.2621215361296 total t = 225798677163649.97 j = 92 Sigma[Inner] = 118.05230259365199
    dt= 18690466087.065907 timestep= 22168 max Sigma= 1151.017953225184 total t = 225985535532424.62 j = 92 Sigma[Inner] = 117.94085684829946
    dt= 18700747597.870705 timestep= 22178 max Sigma= 1150.7735584763295 total t = 226172496747275.75 j = 92 Sigma[Inner] = 117.82950146025885
    dt= 18711022247.088768 timestep= 22188 max Sigma= 1150.5289385485228 total t = 226359560739475.78 j = 92 Sigma[Inner] = 117.7182367716839
    dt= 18721290059.321045 timestep= 22198 max Sigma= 1150.284094694725 total t = 226546727440544.4 j = 92 Sigma[Inner] = 117.60706312314392
    dt= 18731551058.902702 timestep= 22208 max Sigma= 1150.0390281619566 total t = 226733996782245.75 j = 92 Sigma[Inner] = 117.49598085358811
    dt= 18741805269.90605 timestep= 22218 max Sigma= 1149.7937401913503 total t = 226921368696586.0 j = 92 Sigma[Inner] = 117.38499030031025
    dt= 18753077089.61424 timestep= 22229 max Sigma= 1149.523669137133 total t = 227127596192900.16 j = 92 Sigma[Inner] = 117.263007024498
    dt= 18763317121.8058 timestep= 22239 max Sigma= 1149.2779201610506 total t = 227315183289523.22 j = 92 Sigma[Inner] = 117.15221016573834
    dt= 18773550438.40407 timestep= 22249 max Sigma= 1149.0319535577758 total t = 227502872749513.38 j = 92 Sigma[Inner] = 117.0415060579833
    dt= 18783777062.433643 timestep= 22259 max Sigma= 1148.785770544832 total t = 227690664505841.5 j = 92 Sigma[Inner] = 116.9308950313921
    dt= 18793997016.670654 timestep= 22269 max Sigma= 1148.5393723341124 total t = 227878558491707.38 j = 92 Sigma[Inner] = 116.8203774143038
    dt= 18804210323.645527 timestep= 22279 max Sigma= 1148.292760131928 total t = 228066554640537.22 j = 92 Sigma[Inner] = 116.70995353320752
    dt= 18814417005.645607 timestep= 22289 max Sigma= 1148.045935139052 total t = 228254652885981.16 j = 92 Sigma[Inner] = 116.59962371271392
    dt= 18824617084.71789 timestep= 22299 max Sigma= 1147.7988985507714 total t = 228442853161910.9 j = 92 Sigma[Inner] = 116.48938827552706
    dt= 18834810582.671562 timestep= 22309 max Sigma= 1147.5516515569313 total t = 228631155402417.25 j = 92 Sigma[Inner] = 116.37924754241759
    dt= 18844997521.080627 timestep= 22319 max Sigma= 1147.3041953419784 total t = 228819559541808.0 j = 92 Sigma[Inner] = 116.26920183219647
    dt= 18855177921.286453 timestep= 22329 max Sigma= 1147.056531085011 total t = 229008065514605.2 j = 92 Sigma[Inner] = 116.15925146168944
    dt= 18865351804.400307 timestep= 22339 max Sigma= 1146.8086599598187 total t = 229196673255543.1 j = 92 Sigma[Inner] = 116.04939674571263
    dt= 18875519191.30583 timestep= 22349 max Sigma= 1146.560583134929 total t = 229385382699565.84 j = 92 Sigma[Inner] = 115.93963799704876
    dt= 18885680102.66151 timestep= 22359 max Sigma= 1146.312301773651 total t = 229574193781825.22 j = 92 Sigma[Inner] = 115.82997552642426
    dt= 18895834558.90311 timestep= 22369 max Sigma= 1146.063817034115 total t = 229763106437678.28 j = 92 Sigma[Inner] = 115.72040964248681
    dt= 18905982580.2461 timestep= 22379 max Sigma= 1145.815130069318 total t = 229952120602685.25 j = 92 Sigma[Inner] = 115.61094065178445
    dt= 18916124186.687996 timestep= 22389 max Sigma= 1145.5662420271651 total t = 230141236212607.22 j = 92 Sigma[Inner] = 115.50156885874426
    dt= 18926259398.01072 timestep= 22399 max Sigma= 1145.3171540505073 total t = 230330453203404.22 j = 92 Sigma[Inner] = 115.39229456565313
    dt= 18936388233.78293 timestep= 22409 max Sigma= 1145.0678672771862 total t = 230519771511232.9 j = 92 Sigma[Inner] = 115.28311807263815
    dt= 18946510713.362286 timestep= 22419 max Sigma= 1144.8183828400706 total t = 230709191072444.38 j = 92 Sigma[Inner] = 115.1740396776485
    dt= 18956626855.897743 timestep= 22429 max Sigma= 1144.5687018670992 total t = 230898711823582.2 j = 92 Sigma[Inner] = 115.06505967643763
    dt= 18966736680.33178 timestep= 22439 max Sigma= 1144.3188254813167 total t = 231088333701380.22 j = 92 Sigma[Inner] = 114.9561783625463
    dt= 18976840205.40258 timestep= 22449 max Sigma= 1144.0687548009125 total t = 231278056642760.72 j = 92 Sigma[Inner] = 114.84739602728625
    dt= 18986937449.646244 timestep= 22459 max Sigma= 1143.8184909392621 total t = 231467880584832.16 j = 92 Sigma[Inner] = 114.73871295972477
    dt= 18997028431.39896 timestep= 22469 max Sigma= 1143.5680350049595 total t = 231657805464887.3 j = 92 Sigma[Inner] = 114.63012944666953
    dt= 19007113168.7991 timestep= 22479 max Sigma= 1143.317388101858 total t = 231847831220401.2 j = 92 Sigma[Inner] = 114.52164577265461
    dt= 19017191679.789345 timestep= 22489 max Sigma= 1143.066551329106 total t = 232037957789029.06 j = 92 Sigma[Inner] = 114.41326221992675
    dt= 19027263982.118748 timestep= 22499 max Sigma= 1142.815525781182 total t = 232228185108604.66 j = 92 Sigma[Inner] = 114.3049790684325
    dt= 19037330093.344837 timestep= 22509 max Sigma= 1142.56431254793 total t = 232418513117138.03 j = 92 Sigma[Inner] = 114.19679659580574
    


    
![png](output_5_181.png)
    



    
![png](output_5_182.png)
    



    
![png](output_5_183.png)
    


    dt= 19047390030.835606 timestep= 22519 max Sigma= 1142.312912714597 total t = 232608941752813.9 j = 92 Sigma[Inner] = 114.08871507735614
    dt= 19057443811.771507 timestep= 22529 max Sigma= 1142.0613273618649 total t = 232799470953989.53 j = 92 Sigma[Inner] = 113.98073478605812
    dt= 19067491453.147503 timestep= 22539 max Sigma= 1141.809557565887 total t = 232990100659192.94 j = 92 Sigma[Inner] = 113.87285599254024
    dt= 19077532971.77497 timestep= 22549 max Sigma= 1141.55760439832 total t = 233180830807121.28 j = 92 Sigma[Inner] = 113.76507896507546
    dt= 19087568384.28365 timestep= 22559 max Sigma= 1141.30546892636 total t = 233371661336638.56 j = 92 Sigma[Inner] = 113.65740396957177
    dt= 19097597707.123577 timestep= 22569 max Sigma= 1141.0531522127726 total t = 233562592186774.25 j = 92 Sigma[Inner] = 113.5498312695635
    dt= 19107620956.566986 timestep= 22579 max Sigma= 1140.8006553159269 total t = 233753623296721.3 j = 92 Sigma[Inner] = 113.4423611262031
    dt= 19117638148.71011 timestep= 22589 max Sigma= 1140.5479792898282 total t = 233944754605834.47 j = 92 Sigma[Inner] = 113.33499379825359
    dt= 19127649299.475136 timestep= 22599 max Sigma= 1140.2951251841496 total t = 234135986053628.38 j = 92 Sigma[Inner] = 113.22772954208149
    dt= 19137654424.61196 timestep= 22609 max Sigma= 1140.0420940442627 total t = 234327317579776.1 j = 92 Sigma[Inner] = 113.12056861165019
    dt= 19147653539.700012 timestep= 22619 max Sigma= 1139.7888869112692 total t = 234518749124107.1 j = 92 Sigma[Inner] = 113.01351125851403
    dt= 19157646660.150043 timestep= 22629 max Sigma= 1139.5355048220322 total t = 234710280626605.84 j = 92 Sigma[Inner] = 112.90655773181277
    dt= 19167633801.205902 timestep= 22639 max Sigma= 1139.2819488092048 total t = 234901912027409.94 j = 92 Sigma[Inner] = 112.79970827826654
    dt= 19177614977.946266 timestep= 22649 max Sigma= 1139.0282199012624 total t = 235093643266808.47 j = 92 Sigma[Inner] = 112.69296314217127
    dt= 19187590205.28635 timestep= 22659 max Sigma= 1138.7743191225304 total t = 235285474285240.44 j = 92 Sigma[Inner] = 112.58632256539488
    dt= 19197559497.97967 timestep= 22669 max Sigma= 1138.5202474932141 total t = 235477405023293.12 j = 92 Sigma[Inner] = 112.47978678737346
    dt= 19207522870.619644 timestep= 22679 max Sigma= 1138.266006029428 total t = 235669435421700.5 j = 92 Sigma[Inner] = 112.37335604510817
    dt= 19217480337.641346 timestep= 22689 max Sigma= 1138.0115957432233 total t = 235861565421341.56 j = 92 Sigma[Inner] = 112.26703057316278
    dt= 19227431913.323082 timestep= 22699 max Sigma= 1137.7570176426182 total t = 236053794963238.72 j = 92 Sigma[Inner] = 112.16081060366129
    dt= 19237377611.78808 timestep= 22709 max Sigma= 1137.5022727316225 total t = 236246123988556.47 j = 92 Sigma[Inner] = 112.05469636628624
    dt= 19247317447.006012 timestep= 22719 max Sigma= 1137.2473620102685 total t = 236438552438599.5 j = 92 Sigma[Inner] = 111.94868808827725
    dt= 19257251432.794697 timestep= 22729 max Sigma= 1136.9922864746366 total t = 236631080254811.53 j = 92 Sigma[Inner] = 111.8427859944302
    dt= 19267179582.821594 timestep= 22739 max Sigma= 1136.737047116883 total t = 236823707378773.5 j = 92 Sigma[Inner] = 111.73699030709663
    dt= 19277101910.605385 timestep= 22749 max Sigma= 1136.481644925264 total t = 237016433752202.3 j = 92 Sigma[Inner] = 111.63130124618354
    dt= 19287018429.51753 timestep= 22759 max Sigma= 1136.226080884166 total t = 237209259316949.28 j = 92 Sigma[Inner] = 111.52571902915392
    dt= 19296929152.783714 timestep= 22769 max Sigma= 1135.9703559741297 total t = 237402184014998.4 j = 92 Sigma[Inner] = 111.4202438710269
    dt= 19306834093.485455 timestep= 22779 max Sigma= 1135.7144711718738 total t = 237595207788465.34 j = 92 Sigma[Inner] = 111.31487598437921
    dt= 19316733264.561535 timestep= 22789 max Sigma= 1135.4584274503247 total t = 237788330579595.75 j = 92 Sigma[Inner] = 111.20961557934635
    dt= 19326626678.80947 timestep= 22799 max Sigma= 1135.2022257786377 total t = 237981552330763.94 j = 92 Sigma[Inner] = 111.10446286362432
    dt= 19336514348.886955 timestep= 22809 max Sigma= 1134.9458671222244 total t = 238174872984471.22 j = 92 Sigma[Inner] = 110.99941804247159
    dt= 19346396287.313347 timestep= 22819 max Sigma= 1134.6893524427758 total t = 238368292483344.9 j = 92 Sigma[Inner] = 110.89448131871174
    dt= 19356272506.47103 timestep= 22829 max Sigma= 1134.4326826982874 total t = 238561810770136.75 j = 92 Sigma[Inner] = 110.78965289273594
    dt= 19366143018.606873 timestep= 22839 max Sigma= 1134.1758588430832 total t = 238755427787721.47 j = 92 Sigma[Inner] = 110.68493296250622
    dt= 19376007835.833565 timestep= 22849 max Sigma= 1133.9188818278392 total t = 238949143479095.62 j = 92 Sigma[Inner] = 110.58032172355877
    dt= 19385866970.131046 timestep= 22859 max Sigma= 1133.6617525996062 total t = 239142957787376.1 j = 92 Sigma[Inner] = 110.47581936900748
    dt= 19395720433.347836 timestep= 22869 max Sigma= 1133.4044721018342 total t = 239336870655798.88 j = 92 Sigma[Inner] = 110.37142608954815
    dt= 19405568237.202377 timestep= 22879 max Sigma= 1133.1470412743945 total t = 239530882027717.75 j = 92 Sigma[Inner] = 110.26714207346257
    dt= 19415410393.284367 timestep= 22889 max Sigma= 1132.8894610536038 total t = 239724991846602.9 j = 92 Sigma[Inner] = 110.16296750662322
    dt= 19425246913.056084 timestep= 22899 max Sigma= 1132.6317323722453 total t = 239919200056039.75 j = 92 Sigma[Inner] = 110.05890257249794
    dt= 19435077807.853718 timestep= 22909 max Sigma= 1132.3738561595908 total t = 240113506599727.66 j = 92 Sigma[Inner] = 109.95494745215525
    dt= 19444903088.88856 timestep= 22919 max Sigma= 1132.1158333414235 total t = 240307911421478.66 j = 92 Sigma[Inner] = 109.85110232426949
    dt= 19454722767.248383 timestep= 22929 max Sigma= 1131.85766484006 total t = 240502414465216.22 j = 92 Sigma[Inner] = 109.74736736512659
    dt= 19464536853.898643 timestep= 22939 max Sigma= 1131.5993515743703 total t = 240697015674973.97 j = 92 Sigma[Inner] = 109.64374274862992
    dt= 19474345359.683727 timestep= 22949 max Sigma= 1131.3408944598011 total t = 240891714994894.53 j = 92 Sigma[Inner] = 109.54022864630626
    dt= 19484148295.32822 timestep= 22959 max Sigma= 1131.0822944083952 total t = 241086512369228.4 j = 92 Sigma[Inner] = 109.43682522731237
    dt= 19493945671.43807 timestep= 22969 max Sigma= 1130.8235523288126 total t = 241281407742332.53 j = 92 Sigma[Inner] = 109.33353265844144
    dt= 19503737498.50186 timestep= 22979 max Sigma= 1130.5646691263523 total t = 241476401058669.4 j = 92 Sigma[Inner] = 109.23035110412992
    dt= 19513523786.891922 timestep= 22989 max Sigma= 1130.3056457029702 total t = 241671492262805.72 j = 92 Sigma[Inner] = 109.12728072646455
    dt= 19523304546.865585 timestep= 22999 max Sigma= 1130.046482957303 total t = 241866681299411.22 j = 92 Sigma[Inner] = 109.0243216851895
    dt= 19533079788.566322 timestep= 23009 max Sigma= 1129.7871817846844 total t = 242061968113257.7 j = 92 Sigma[Inner] = 108.92147413771397
    


    
![png](output_5_185.png)
    



    
![png](output_5_186.png)
    



    
![png](output_5_187.png)
    


    dt= 19542849522.02487 timestep= 23019 max Sigma= 1129.527743077167 total t = 242257352649217.6 j = 92 Sigma[Inner] = 108.81873823911974
    dt= 19552613757.16044 timestep= 23029 max Sigma= 1129.2681677235414 total t = 242452834852263.12 j = 92 Sigma[Inner] = 108.71611414216886
    dt= 19562372503.781773 timestep= 23039 max Sigma= 1129.0084566093553 total t = 242648414667465.12 j = 92 Sigma[Inner] = 108.61360199731193
    dt= 19572125771.58832 timestep= 23049 max Sigma= 1128.7486106169326 total t = 242844092039991.94 j = 92 Sigma[Inner] = 108.51120195269611
    dt= 19581873570.17132 timestep= 23059 max Sigma= 1128.4886306253925 total t = 243039866915108.2 j = 92 Sigma[Inner] = 108.40891415417356
    dt= 19591615909.01489 timestep= 23069 max Sigma= 1128.2285175106686 total t = 243235739238173.88 j = 92 Sigma[Inner] = 108.30673874530987
    dt= 19601352797.49711 timestep= 23079 max Sigma= 1127.968272145527 total t = 243431708954643.4 j = 92 Sigma[Inner] = 108.20467586739292
    dt= 19611084244.89113 timestep= 23089 max Sigma= 1127.7078953995847 total t = 243627776010064.2 j = 92 Sigma[Inner] = 108.1027256594417
    dt= 19620810260.366173 timestep= 23099 max Sigma= 1127.4473881393264 total t = 243823940350075.8 j = 92 Sigma[Inner] = 108.00088825821523
    dt= 19630530852.988625 timestep= 23109 max Sigma= 1127.1867512281256 total t = 244020201920409.06 j = 92 Sigma[Inner] = 107.89916379822186
    dt= 19640246031.723095 timestep= 23119 max Sigma= 1126.925985526261 total t = 244216560666884.75 j = 92 Sigma[Inner] = 107.79755241172853
    dt= 19649955805.433357 timestep= 23129 max Sigma= 1126.6650918909313 total t = 244413016535412.9 j = 92 Sigma[Inner] = 107.69605422877014
    dt= 19659660182.883465 timestep= 23139 max Sigma= 1126.4040711762773 total t = 244609569471991.56 j = 92 Sigma[Inner] = 107.59466937715929
    dt= 19669359172.738697 timestep= 23149 max Sigma= 1126.1429242333961 total t = 244806219422705.8 j = 92 Sigma[Inner] = 107.49339798249589
    dt= 19679052783.56657 timestep= 23159 max Sigma= 1125.8816519103589 total t = 245002966333727.0 j = 92 Sigma[Inner] = 107.39224016817703
    dt= 19688741023.837845 timestep= 23169 max Sigma= 1125.6202550522285 total t = 245199810151311.4 j = 92 Sigma[Inner] = 107.29119605540691
    dt= 19698423901.927464 timestep= 23179 max Sigma= 1125.3587345010753 total t = 245396750821799.62 j = 92 Sigma[Inner] = 107.19026576320702
    dt= 19708101426.115555 timestep= 23189 max Sigma= 1125.097091095994 total t = 245593788291615.5 j = 92 Sigma[Inner] = 107.08944940842616
    dt= 19717773604.588364 timestep= 23199 max Sigma= 1124.8353256731202 total t = 245790922507265.12 j = 92 Sigma[Inner] = 106.98874710575099
    dt= 19727440445.43922 timestep= 23209 max Sigma= 1124.5734390656476 total t = 245988153415335.9 j = 92 Sigma[Inner] = 106.88815896771611
    dt= 19737101956.669422 timestep= 23219 max Sigma= 1124.3114321038418 total t = 246185480962495.7 j = 92 Sigma[Inner] = 106.78768510471478
    dt= 19746758146.189247 timestep= 23229 max Sigma= 1124.0493056150578 total t = 246382905095491.94 j = 92 Sigma[Inner] = 106.68732562500945
    dt= 19756409021.818817 timestep= 23239 max Sigma= 1123.7870604237557 total t = 246580425761150.56 j = 92 Sigma[Inner] = 106.58708063474253
    dt= 19766054591.288998 timestep= 23249 max Sigma= 1123.5246973515164 total t = 246778042906375.25 j = 92 Sigma[Inner] = 106.48695023794698
    dt= 19775694862.242313 timestep= 23259 max Sigma= 1123.2622172170566 total t = 246975756478146.5 j = 92 Sigma[Inner] = 106.38693453655733
    dt= 19785329842.233875 timestep= 23269 max Sigma= 1122.9996208362445 total t = 247173566423520.8 j = 92 Sigma[Inner] = 106.28703363042055
    dt= 19794959538.732174 timestep= 23279 max Sigma= 1122.7369090221148 total t = 247371472689629.72 j = 92 Sigma[Inner] = 106.18724761730701
    dt= 19804583959.12007 timestep= 23289 max Sigma= 1122.4740825848835 total t = 247569475223678.94 j = 92 Sigma[Inner] = 106.08757659292166
    dt= 19814203110.69551 timestep= 23299 max Sigma= 1122.2111423319627 total t = 247767573972947.6 j = 92 Sigma[Inner] = 105.98802065091519
    dt= 19823817000.672523 timestep= 23309 max Sigma= 1121.9480890679768 total t = 247965768884787.3 j = 92 Sigma[Inner] = 105.88857988289509
    dt= 19833425636.182 timestep= 23319 max Sigma= 1121.6849235947748 total t = 248164059906621.34 j = 92 Sigma[Inner] = 105.78925437843714
    dt= 19843029024.272503 timestep= 23329 max Sigma= 1121.4216467114468 total t = 248362446985943.94 j = 92 Sigma[Inner] = 105.69004422509657
    dt= 19852627171.91119 timestep= 23339 max Sigma= 1121.158259214336 total t = 248560930070319.22 j = 92 Sigma[Inner] = 105.59094950841956
    dt= 19862220085.984516 timestep= 23349 max Sigma= 1120.8947618970572 total t = 248759509107380.62 j = 92 Sigma[Inner] = 105.4919703119547
    dt= 19871807773.29916 timestep= 23359 max Sigma= 1120.6311555505044 total t = 248958184044830.03 j = 92 Sigma[Inner] = 105.3931067172643
    dt= 19881390240.58272 timestep= 23369 max Sigma= 1120.3674409628713 total t = 249156954830436.88 j = 92 Sigma[Inner] = 105.29435880393628
    dt= 19890967494.484653 timestep= 23379 max Sigma= 1120.1036189196598 total t = 249355821412037.44 j = 92 Sigma[Inner] = 105.19572664959531
    dt= 19900539541.576885 timestep= 23389 max Sigma= 1119.8396902036977 total t = 249554783737534.22 j = 92 Sigma[Inner] = 105.0972103299149
    dt= 19910106388.35474 timestep= 23399 max Sigma= 1119.5756555951498 total t = 249753841754894.84 j = 92 Sigma[Inner] = 104.99880991862868
    dt= 19919668041.23767 timestep= 23409 max Sigma= 1119.3115158715318 total t = 249952995412151.56 j = 92 Sigma[Inner] = 104.90052548754225
    dt= 19929224506.56998 timestep= 23419 max Sigma= 1119.0472718077244 total t = 250152244657400.4 j = 92 Sigma[Inner] = 104.80235710654493
    dt= 19938775790.62164 timestep= 23429 max Sigma= 1118.782924175985 total t = 250351589438800.34 j = 92 Sigma[Inner] = 104.70430484362157
    dt= 19948321899.589016 timestep= 23439 max Sigma= 1118.5184737459622 total t = 250551029704572.75 j = 92 Sigma[Inner] = 104.60636876486426
    dt= 19957862839.59561 timestep= 23449 max Sigma= 1118.2539212847078 total t = 250750565403000.56 j = 92 Sigma[Inner] = 104.50854893448398
    dt= 19966798795.679024 timestep= 23459 max Sigma= 1117.9892708492728 total t = 250950193398943.66 j = 92 Sigma[Inner] = 104.41084662955
    dt= 19975681082.463036 timestep= 23469 max Sigma= 1117.7245284335436 total t = 251149910244351.75 j = 92 Sigma[Inner] = 104.31326383206638
    dt= 19984557466.059628 timestep= 23479 max Sigma= 1117.4596948675344 total t = 251349715880152.53 j = 92 Sigma[Inner] = 104.21580061189545
    dt= 19993427955.685295 timestep= 23489 max Sigma= 1117.1947709152241 total t = 251549610247364.84 j = 92 Sigma[Inner] = 104.11845701365209
    dt= 20002292560.474094 timestep= 23499 max Sigma= 1116.929757337882 total t = 251749593287099.3 j = 92 Sigma[Inner] = 104.02123308011575
    dt= 20011151289.478405 timestep= 23509 max Sigma= 1116.6646548940778 total t = 251949664940557.34 j = 92 Sigma[Inner] = 103.92412885224257
    


    
![png](output_5_189.png)
    



    
![png](output_5_190.png)
    



    
![png](output_5_191.png)
    


    dt= 20020004151.66985 timestep= 23519 max Sigma= 1116.3994643396995 total t = 252149825149030.66 j = 92 Sigma[Inner] = 103.82714436917749
    dt= 20028851155.9401 timestep= 23529 max Sigma= 1116.1341864279616 total t = 252350073853900.0 j = 92 Sigma[Inner] = 103.7302796682666
    dt= 20037692311.101727 timestep= 23539 max Sigma= 1115.8688219094229 total t = 252550410996634.7 j = 92 Sigma[Inner] = 103.63353478506967
    dt= 20046527625.88896 timestep= 23549 max Sigma= 1115.603371531996 total t = 252750836518791.72 j = 92 Sigma[Inner] = 103.5369097533726
    dt= 20055357108.95864 timestep= 23559 max Sigma= 1115.3378360409606 total t = 252951350362015.1 j = 92 Sigma[Inner] = 103.44040460520007
    dt= 20064180768.890903 timestep= 23569 max Sigma= 1115.0722161789784 total t = 253151952468034.88 j = 92 Sigma[Inner] = 103.34401937082802
    dt= 20072998614.190063 timestep= 23579 max Sigma= 1114.8065126861034 total t = 253352642778666.47 j = 92 Sigma[Inner] = 103.24775407879616
    dt= 20081810653.28536 timestep= 23589 max Sigma= 1114.5407262997942 total t = 253553421235810.06 j = 92 Sigma[Inner] = 103.15160875592068
    dt= 20090616894.531708 timestep= 23599 max Sigma= 1114.274857754928 total t = 253754287781449.6 j = 92 Sigma[Inner] = 103.0555834273067
    dt= 20099417346.210617 timestep= 23609 max Sigma= 1114.008907783812 total t = 253955242357652.2 j = 92 Sigma[Inner] = 102.95967811636079
    dt= 20108212016.53076 timestep= 23619 max Sigma= 1113.7428771161942 total t = 254156284906567.38 j = 92 Sigma[Inner] = 102.86389284480357
    dt= 20117000913.628872 timestep= 23629 max Sigma= 1113.4767664792778 total t = 254357415370426.3 j = 92 Sigma[Inner] = 102.76822763268218
    dt= 20125784045.570442 timestep= 23639 max Sigma= 1113.2105765977312 total t = 254558633691541.22 j = 92 Sigma[Inner] = 102.67268249838271
    dt= 20134561420.350468 timestep= 23649 max Sigma= 1112.9443081937009 total t = 254759939812304.6 j = 92 Sigma[Inner] = 102.57725745864276
    dt= 20143333045.894146 timestep= 23659 max Sigma= 1112.6779619868216 total t = 254961333675188.5 j = 92 Sigma[Inner] = 102.48195252856385
    dt= 20152098930.057693 timestep= 23669 max Sigma= 1112.4115386942299 total t = 255162815222743.72 j = 92 Sigma[Inner] = 102.38676772162376
    dt= 20160859080.628952 timestep= 23679 max Sigma= 1112.1450390305752 total t = 255364384397599.5 j = 92 Sigma[Inner] = 102.29170304968908
    dt= 20169613505.328182 timestep= 23689 max Sigma= 1111.8784637080296 total t = 255566041142462.34 j = 92 Sigma[Inner] = 102.1967585230274
    dt= 20178362211.808735 timestep= 23699 max Sigma= 1111.6118134363005 total t = 255767785400115.62 j = 92 Sigma[Inner] = 102.10193415031976
    dt= 20187105207.65769 timestep= 23709 max Sigma= 1111.345088922642 total t = 255969617113419.06 j = 92 Sigma[Inner] = 102.00722993867291
    dt= 20195842500.396656 timestep= 23719 max Sigma= 1111.078290871865 total t = 256171536225307.72 j = 92 Sigma[Inner] = 101.91264589363165
    dt= 20204574097.48238 timestep= 23729 max Sigma= 1110.8114199863492 total t = 256373542678791.5 j = 92 Sigma[Inner] = 101.81818201919091
    dt= 20213300006.3074 timestep= 23739 max Sigma= 1110.5444769660535 total t = 256575636416954.66 j = 92 Sigma[Inner] = 101.7238383178081
    dt= 20222020234.200775 timestep= 23749 max Sigma= 1110.2774625085267 total t = 256777817382954.9 j = 92 Sigma[Inner] = 101.6296147904153
    dt= 20230734788.428707 timestep= 23759 max Sigma= 1110.0103773089204 total t = 256980085520022.97 j = 92 Sigma[Inner] = 101.53551143643125
    dt= 20239443676.19518 timestep= 23769 max Sigma= 1109.7432220599944 total t = 257182440771461.78 j = 92 Sigma[Inner] = 101.44152825377365
    dt= 20248146904.642662 timestep= 23779 max Sigma= 1109.4759974521337 total t = 257384883080646.16 j = 92 Sigma[Inner] = 101.34766523887099
    dt= 20256844480.852673 timestep= 23789 max Sigma= 1109.2087041733548 total t = 257587412391021.9 j = 92 Sigma[Inner] = 101.25392238667469
    dt= 20265536411.84647 timestep= 23799 max Sigma= 1108.9413429093165 total t = 257790028646105.34 j = 92 Sigma[Inner] = 101.16029969067112
    dt= 20274222704.585686 timestep= 23809 max Sigma= 1108.6739143433324 total t = 257992731789482.6 j = 92 Sigma[Inner] = 101.06679714289349
    dt= 20282903365.972866 timestep= 23819 max Sigma= 1108.406419156379 total t = 258195521764809.16 j = 92 Sigma[Inner] = 100.97341473393378
    dt= 20291578402.85219 timestep= 23829 max Sigma= 1108.1388580271066 total t = 258398398515809.16 j = 92 Sigma[Inner] = 100.88015245295446
    dt= 20300247822.00998 timestep= 23839 max Sigma= 1107.8712316318492 total t = 258601361986274.84 j = 92 Sigma[Inner] = 100.78701028770061
    dt= 20308911630.17541 timestep= 23849 max Sigma= 1107.6035406446356 total t = 258804412120066.2 j = 92 Sigma[Inner] = 100.69398822451133
    dt= 20317569834.020992 timestep= 23859 max Sigma= 1107.335785737198 total t = 259007548861109.94 j = 92 Sigma[Inner] = 100.6010862483316
    dt= 20326222440.16327 timestep= 23869 max Sigma= 1107.067967578981 total t = 259210772153399.3 j = 92 Sigma[Inner] = 100.50830434272409
    dt= 20334869455.163277 timestep= 23879 max Sigma= 1106.8000868371553 total t = 259414081940993.5 j = 92 Sigma[Inner] = 100.4156424898806
    dt= 20343510885.527267 timestep= 23889 max Sigma= 1106.5321441766216 total t = 259617478168016.8 j = 92 Sigma[Inner] = 100.32310067063376
    dt= 20352146737.707165 timestep= 23899 max Sigma= 1106.2641402600257 total t = 259820960778658.4 j = 92 Sigma[Inner] = 100.23067886446852
    dt= 20360777018.101192 timestep= 23909 max Sigma= 1105.9960757477652 total t = 260024529717171.75 j = 92 Sigma[Inner] = 100.1383770495336
    dt= 20369401733.054405 timestep= 23919 max Sigma= 1105.7279512979992 total t = 260228184927873.9 j = 92 Sigma[Inner] = 100.04619520265298
    dt= 20378020888.85926 timestep= 23929 max Sigma= 1105.459767566657 total t = 260431926355145.1 j = 92 Sigma[Inner] = 99.95413329933731
    dt= 20386634491.75618 timestep= 23939 max Sigma= 1105.1915252074502 total t = 260635753943428.22 j = 92 Sigma[Inner] = 99.86219131379517
    dt= 20395242547.93406 timestep= 23949 max Sigma= 1104.9232248718795 total t = 260839667637228.34 j = 92 Sigma[Inner] = 99.77036921894442
    dt= 20403845063.53085 timestep= 23959 max Sigma= 1104.6548672092442 total t = 261043667381111.94 j = 92 Sigma[Inner] = 99.67866698642338
    dt= 20412442044.634033 timestep= 23969 max Sigma= 1104.3864528666516 total t = 261247753119706.78 j = 92 Sigma[Inner] = 99.58708458660205
    dt= 20421033497.281235 timestep= 23979 max Sigma= 1104.117982489027 total t = 261451924797701.22 j = 92 Sigma[Inner] = 99.49562198859326
    dt= 20429619427.46068 timestep= 23989 max Sigma= 1103.849456719121 total t = 261656182359843.66 j = 92 Sigma[Inner] = 99.40427916026367
    dt= 20438199841.111717 timestep= 23999 max Sigma= 1103.5808761975193 total t = 261860525750942.0 j = 92 Sigma[Inner] = 99.3130560682449
    dt= 20446774744.125374 timestep= 24009 max Sigma= 1103.3122415626513 total t = 262064954915863.53 j = 92 Sigma[Inner] = 99.22195267794439
    


    
![png](output_5_193.png)
    



    
![png](output_5_194.png)
    



    
![png](output_5_195.png)
    


    dt= 20455344142.344837 timestep= 24019 max Sigma= 1103.0435534507997 total t = 262269469799534.03 j = 92 Sigma[Inner] = 99.13096895355643
    dt= 20463908041.56593 timestep= 24029 max Sigma= 1102.7748124961083 total t = 262474070346937.5 j = 92 Sigma[Inner] = 99.04010485807284
    dt= 20472466447.537678 timestep= 24039 max Sigma= 1102.5060193305903 total t = 262678756503115.5 j = 92 Sigma[Inner] = 98.94936035329403
    dt= 20481019365.96273 timestep= 24049 max Sigma= 1102.2371745841383 total t = 262883528213167.1 j = 92 Sigma[Inner] = 98.85873539983957
    dt= 20489566802.497902 timestep= 24059 max Sigma= 1101.968278884532 total t = 263088385422247.94 j = 92 Sigma[Inner] = 98.7682299571589
    dt= 20498108762.754623 timestep= 24069 max Sigma= 1101.6993328574467 total t = 263293328075569.94 j = 92 Sigma[Inner] = 98.67784398354205
    dt= 20506645252.29945 timestep= 24079 max Sigma= 1101.4303371264607 total t = 263498356118401.1 j = 92 Sigma[Inner] = 98.58757743613018
    dt= 20515176276.65453 timestep= 24089 max Sigma= 1101.161292313066 total t = 263703469496064.56 j = 92 Sigma[Inner] = 98.49743027092603
    dt= 20523701841.298027 timestep= 24099 max Sigma= 1100.8921990366744 total t = 263908668153938.66 j = 92 Sigma[Inner] = 98.40740244280455
    dt= 20532221951.66466 timestep= 24109 max Sigma= 1100.623057914627 total t = 264113952037456.22 j = 92 Sigma[Inner] = 98.31749390552326
    dt= 20540736613.14612 timestep= 24119 max Sigma= 1100.353869562201 total t = 264319321092104.1 j = 92 Sigma[Inner] = 98.22770461173243
    dt= 20549245831.09154 timestep= 24129 max Sigma= 1100.0846345926202 total t = 264524775263422.97 j = 92 Sigma[Inner] = 98.13803451298571
    dt= 20557749610.807934 timestep= 24139 max Sigma= 1099.8153536170607 total t = 264730314497006.7 j = 92 Sigma[Inner] = 98.04848355975015
    dt= 20566247957.560665 timestep= 24149 max Sigma= 1099.5460272446596 total t = 264935938738501.9 j = 92 Sigma[Inner] = 97.95905170141636
    dt= 20574740876.57387 timestep= 24159 max Sigma= 1099.2766560825237 total t = 265141647933607.84 j = 92 Sigma[Inner] = 97.86973888630871
    dt= 20583228373.030903 timestep= 24169 max Sigma= 1099.0072407357354 total t = 265347442028075.56 j = 92 Sigma[Inner] = 97.78054506169549
    dt= 20591710452.07479 timestep= 24179 max Sigma= 1098.7377818073635 total t = 265553320967707.8 j = 92 Sigma[Inner] = 97.69147017379876
    dt= 20600187118.808624 timestep= 24189 max Sigma= 1098.468279898469 total t = 265759284698358.66 j = 92 Sigma[Inner] = 97.60251416780433
    dt= 20608658378.296032 timestep= 24199 max Sigma= 1098.1987356081124 total t = 265965333165932.84 j = 92 Sigma[Inner] = 97.51367698787185
    dt= 20617124235.56158 timestep= 24209 max Sigma= 1097.9291495333623 total t = 266171466316385.5 j = 92 Sigma[Inner] = 97.42495857714434
    dt= 20625584695.59119 timestep= 24219 max Sigma= 1097.6595222693043 total t = 266377684095721.94 j = 92 Sigma[Inner] = 97.33635887775819
    dt= 20634039763.332527 timestep= 24229 max Sigma= 1097.3898544090443 total t = 266583986449997.06 j = 92 Sigma[Inner] = 97.24787783085277
    dt= 20642489443.695457 timestep= 24239 max Sigma= 1097.1201465437214 total t = 266790373325314.94 j = 92 Sigma[Inner] = 97.15951537658022
    dt= 20650933741.552456 timestep= 24249 max Sigma= 1096.8503992625112 total t = 266996844667828.7 j = 92 Sigma[Inner] = 97.07127145411492
    dt= 20659372661.738953 timestep= 24259 max Sigma= 1096.5806131526358 total t = 267203400423739.9 j = 92 Sigma[Inner] = 96.98314600166303
    dt= 20667806209.0538 timestep= 24269 max Sigma= 1096.310788799369 total t = 267410040539298.2 j = 92 Sigma[Inner] = 96.89513895647211
    dt= 20676234388.259605 timestep= 24279 max Sigma= 1096.0409267860452 total t = 267616764960801.1 j = 92 Sigma[Inner] = 96.80725025484044
    dt= 20684657204.083187 timestep= 24289 max Sigma= 1095.7710276940668 total t = 267823573634593.6 j = 92 Sigma[Inner] = 96.71947983212641
    dt= 20693074661.215897 timestep= 24299 max Sigma= 1095.5010921029088 total t = 268030466507067.66 j = 92 Sigma[Inner] = 96.63182762275775
    dt= 20701486764.314087 timestep= 24309 max Sigma= 1095.2311205901296 total t = 268237443524661.97 j = 92 Sigma[Inner] = 96.54429356024103
    dt= 20709893517.999382 timestep= 24319 max Sigma= 1094.9611137313761 total t = 268444504633861.75 j = 92 Sigma[Inner] = 96.45687757717056
    dt= 20718294926.85915 timestep= 24329 max Sigma= 1094.691072100391 total t = 268651649781198.06 j = 92 Sigma[Inner] = 96.36957960523763
    dt= 20726690995.446823 timestep= 24339 max Sigma= 1094.420996269019 total t = 268858878913247.78 j = 92 Sigma[Inner] = 96.2823995752397
    dt= 20735081728.28229 timestep= 24349 max Sigma= 1094.1508868072158 total t = 269066191976632.97 j = 92 Sigma[Inner] = 96.19533741708926
    dt= 20743467129.852272 timestep= 24359 max Sigma= 1093.8807442830525 total t = 269273588918020.88 j = 92 Sigma[Inner] = 96.10839305982283
    dt= 20751847204.610645 timestep= 24369 max Sigma= 1093.6105692627252 total t = 269481069684123.44 j = 92 Sigma[Inner] = 96.02156643160991
    dt= 20760221956.97886 timestep= 24379 max Sigma= 1093.3403623105598 total t = 269688634221696.72 j = 92 Sigma[Inner] = 95.93485745976176
    dt= 20768591391.346245 timestep= 24389 max Sigma= 1093.0701239890193 total t = 269896282477541.06 j = 92 Sigma[Inner] = 95.84826607074017
    dt= 20776955512.070347 timestep= 24399 max Sigma= 1092.7998548587102 total t = 270104014398500.5 j = 92 Sigma[Inner] = 95.76179219016637
    dt= 20785314323.477367 timestep= 24409 max Sigma= 1092.5295554783906 total t = 270311829931462.34 j = 92 Sigma[Inner] = 95.67543574282938
    dt= 20793667829.862415 timestep= 24419 max Sigma= 1092.2592264049756 total t = 270519729023357.12 j = 92 Sigma[Inner] = 95.58919665269487
    dt= 20802016035.489918 timestep= 24429 max Sigma= 1091.9888681935433 total t = 270727711621158.06 j = 92 Sigma[Inner] = 95.50307484291362
    dt= 20810358944.59391 timestep= 24439 max Sigma= 1091.7184813973436 total t = 270935777671880.9 j = 92 Sigma[Inner] = 95.41707023583002
    dt= 20818696561.37838 timestep= 24449 max Sigma= 1091.448066567802 total t = 271143927122583.56 j = 92 Sigma[Inner] = 95.33118275299051
    dt= 20827028890.01763 timestep= 24459 max Sigma= 1091.1776242545288 total t = 271352159920365.9 j = 92 Sigma[Inner] = 95.24541231515191
    dt= 20835355934.656612 timestep= 24469 max Sigma= 1090.9071550053231 total t = 271560476012369.2 j = 92 Sigma[Inner] = 95.15975884228985
    dt= 20843677699.411213 timestep= 24479 max Sigma= 1090.636659366181 total t = 271768875345776.1 j = 92 Sigma[Inner] = 95.07422225360692
    dt= 20851994188.368587 timestep= 24489 max Sigma= 1090.3661378813006 total t = 271977357867810.38 j = 92 Sigma[Inner] = 94.98880246754086
    dt= 20860305405.58751 timestep= 24499 max Sigma= 1090.0955910930902 total t = 272185923525736.28 j = 92 Sigma[Inner] = 94.90349940177296
    dt= 20868611355.098648 timestep= 24509 max Sigma= 1089.8250195421717 total t = 272394572266858.7 j = 92 Sigma[Inner] = 94.8183129732357
    


    
![png](output_5_197.png)
    



    
![png](output_5_198.png)
    



    
![png](output_5_199.png)
    


    dt= 20876912040.90496 timestep= 24519 max Sigma= 1089.55442376739 total t = 272603304038522.5 j = 92 Sigma[Inner] = 94.73324309812129
    dt= 20885207466.981895 timestep= 24529 max Sigma= 1089.2838043058164 total t = 272812118788112.56 j = 92 Sigma[Inner] = 94.64828969188929
    dt= 20893497637.277786 timestep= 24539 max Sigma= 1089.013161692756 total t = 273021016463053.4 j = 92 Sigma[Inner] = 94.56345266927461
    dt= 20901782555.714123 timestep= 24549 max Sigma= 1088.7424964617558 total t = 273229997010808.78 j = 92 Sigma[Inner] = 94.47873194429549
    dt= 20910062226.185867 timestep= 24559 max Sigma= 1088.471809144607 total t = 273439060378881.5 j = 92 Sigma[Inner] = 94.39412743026118
    dt= 20918336652.561752 timestep= 24569 max Sigma= 1088.2011002713537 total t = 273648206514813.28 j = 92 Sigma[Inner] = 94.30963903977984
    dt= 20926605838.68455 timestep= 24579 max Sigma= 1087.9303703702976 total t = 273857435366184.2 j = 92 Sigma[Inner] = 94.22526668476604
    dt= 20934869788.371407 timestep= 24589 max Sigma= 1087.6596199680046 total t = 274066746880612.78 j = 92 Sigma[Inner] = 94.1410102764486
    dt= 20943128505.41415 timestep= 24599 max Sigma= 1087.3888495893111 total t = 274276141005755.62 j = 92 Sigma[Inner] = 94.05686972537802
    dt= 20951381993.5795 timestep= 24609 max Sigma= 1087.1180597573293 total t = 274485617689306.97 j = 92 Sigma[Inner] = 93.97284494143413
    dt= 20959630256.60945 timestep= 24619 max Sigma= 1086.8472509934525 total t = 274695176878998.66 j = 92 Sigma[Inner] = 93.88893583383353
    dt= 20967873298.221497 timestep= 24629 max Sigma= 1086.5764238173624 total t = 274904818522599.8 j = 92 Sigma[Inner] = 93.8051423111369
    dt= 20976111122.108925 timestep= 24639 max Sigma= 1086.305578747034 total t = 275114542567916.5 j = 92 Sigma[Inner] = 93.72146428125663
    dt= 20984343731.9411 timestep= 24649 max Sigma= 1086.0347162987407 total t = 275324348962791.8 j = 92 Sigma[Inner] = 93.63790165146395
    dt= 20992571131.36374 timestep= 24659 max Sigma= 1085.763836987062 total t = 275534237655105.1 j = 92 Sigma[Inner] = 93.5544543283962
    dt= 21000793323.999165 timestep= 24669 max Sigma= 1085.4929413248876 total t = 275744208592772.34 j = 92 Sigma[Inner] = 93.47112221806404
    dt= 21009010313.446617 timestep= 24679 max Sigma= 1085.2220298234236 total t = 275954261723745.4 j = 92 Sigma[Inner] = 93.38790522585859
    dt= 21017222103.282528 timestep= 24689 max Sigma= 1084.9511029921973 total t = 276164396996012.2 j = 92 Sigma[Inner] = 93.30480325655867
    dt= 21025428697.060726 timestep= 24699 max Sigma= 1084.680161339064 total t = 276374614357596.12 j = 92 Sigma[Inner] = 93.22181621433776
    dt= 21033630098.312725 timestep= 24709 max Sigma= 1084.4092053702122 total t = 276584913756555.97 j = 92 Sigma[Inner] = 93.13894400277088
    dt= 21041826310.548054 timestep= 24719 max Sigma= 1084.1382355901692 total t = 276795295140985.9 j = 92 Sigma[Inner] = 93.05618652484175
    dt= 21050017337.254395 timestep= 24729 max Sigma= 1083.8672525018053 total t = 277005758459014.94 j = 92 Sigma[Inner] = 92.97354368294945
    dt= 21058203181.89794 timestep= 24739 max Sigma= 1083.596256606342 total t = 277216303658806.78 j = 92 Sigma[Inner] = 92.8910153789155
    dt= 21066383847.9236 timestep= 24749 max Sigma= 1083.3252484033542 total t = 277426930688559.84 j = 92 Sigma[Inner] = 92.80860151399045
    dt= 21074559338.755283 timestep= 24759 max Sigma= 1083.0542283907778 total t = 277637639496506.78 j = 92 Sigma[Inner] = 92.72630198886067
    dt= 21082729657.796112 timestep= 24769 max Sigma= 1082.7831970649147 total t = 277848430030914.3 j = 92 Sigma[Inner] = 92.64411670365513
    dt= 21090894808.428688 timestep= 24779 max Sigma= 1082.5121549204375 total t = 278059302240083.25 j = 92 Sigma[Inner] = 92.56204555795192
    dt= 21099054794.015316 timestep= 24789 max Sigma= 1082.2411024503958 total t = 278270256072348.03 j = 92 Sigma[Inner] = 92.48008845078485
    dt= 21107209617.898308 timestep= 24799 max Sigma= 1081.97004014622 total t = 278481291476076.56 j = 92 Sigma[Inner] = 92.39824528065003
    dt= 21115359283.400146 timestep= 24809 max Sigma= 1081.6989684977284 total t = 278692408399670.12 j = 92 Sigma[Inner] = 92.31651594551231
    dt= 21123503793.82378 timestep= 24819 max Sigma= 1081.427887993131 total t = 278903606791563.03 j = 92 Sigma[Inner] = 92.23490034281174
    dt= 21131173451.69004 timestep= 24829 max Sigma= 1081.1568006369014 total t = 279114884947376.7 j = 92 Sigma[Inner] = 92.15339882553226
    dt= 21138528891.02148 timestep= 24839 max Sigma= 1080.8857143954494 total t = 279326237141253.22 j = 92 Sigma[Inner] = 92.07201353286868
    dt= 21145878947.40231 timestep= 24849 max Sigma= 1080.6146307304703 total t = 279537662859912.12 j = 92 Sigma[Inner] = 91.99074464233611
    dt= 21153223626.542694 timestep= 24859 max Sigma= 1080.3435501160313 total t = 279749162049555.25 j = 92 Sigma[Inner] = 91.90959203431376
    dt= 21160562934.128265 timestep= 24869 max Sigma= 1080.072473024508 total t = 279960734656441.6 j = 92 Sigma[Inner] = 91.82855558872475
    dt= 21167896875.820194 timestep= 24879 max Sigma= 1079.8013999265868 total t = 280172380626886.72 j = 92 Sigma[Inner] = 91.74763518504264
    dt= 21175225457.255367 timestep= 24889 max Sigma= 1079.5303312912718 total t = 280384099907262.72 j = 92 Sigma[Inner] = 91.66683070229719
    dt= 21182548684.046734 timestep= 24899 max Sigma= 1079.2592675858925 total t = 280595892443997.88 j = 92 Sigma[Inner] = 91.58614201908051
    dt= 21189866561.78357 timestep= 24909 max Sigma= 1078.988209276108 total t = 280807758183576.56 j = 92 Sigma[Inner] = 91.50556901355307
    dt= 21197179096.031734 timestep= 24919 max Sigma= 1078.717156825914 total t = 281019697072538.84 j = 92 Sigma[Inner] = 91.42511156344989
    dt= 21204486292.333904 timestep= 24929 max Sigma= 1078.4461106976469 total t = 281231709057480.34 j = 92 Sigma[Inner] = 91.34476954608633
    dt= 21211788156.209824 timestep= 24939 max Sigma= 1078.175071351991 total t = 281443794085051.97 j = 92 Sigma[Inner] = 91.26454283836418
    dt= 21219084693.156525 timestep= 24949 max Sigma= 1077.904039247984 total t = 281655952101959.75 j = 92 Sigma[Inner] = 91.18443131677769
    dt= 21226375908.64861 timestep= 24959 max Sigma= 1077.6330148430225 total t = 281868183054964.5 j = 92 Sigma[Inner] = 91.10443485741911
    dt= 21233661808.138424 timestep= 24969 max Sigma= 1077.3619985928663 total t = 282080486890881.6 j = 92 Sigma[Inner] = 91.0245533359849
    dt= 21240942397.056297 timestep= 24979 max Sigma= 1077.090990951647 total t = 282292863556581.2 j = 92 Sigma[Inner] = 90.94478662778137
    dt= 21248217680.81074 timestep= 24989 max Sigma= 1076.8199923718694 total t = 282505312998986.9 j = 92 Sigma[Inner] = 90.86513460773035
    dt= 21256214371.905556 timestep= 25000 max Sigma= 1076.5219049381658 total t = 282739091379448.9 j = 92 Sigma[Inner] = 90.77764970074638
    
