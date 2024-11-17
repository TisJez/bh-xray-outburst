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
