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
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import ScalarFormatter
%matplotlib inline

# Constants in CGS units
G = 6.67430e-8  # Gravitational constant (cm^3 g^-1 s^-2)
M_sun = 1.989e33  # Mass of the Sun (g)
M_wd = M_sun  # Mass of white dwarf (g)
M_bh = 9*M_sun # Mass of black hole (g)
M_smbh = 4.3e6*M_sun # Mass of SMBH (g)
k_B = 1.380649e-16  # Boltzmann constant (erg/K)
m_p = 1.6726219e-24  # Proton mass (g)
sigma_SB = 5.670374419e-5  # Stefan-Boltzmann constant (erg cm^-2 s^-1 K^-4)
#alpha = 0.4  # Alpha-viscosity parameter
r_cons = 8.31446261815324e7 # Molar gas constant in cgs

alpha_cold = 0.04
alpha_hot = 0.2

# Disk parameters
R_1 = 5e8  # Inner radius of the disk (cm)
R_K = 2.2e11  # Radius where initial mass is added (cm)
R_N = 4.2e11  # Outer radius of the disk (cm)
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
Δr = np.diff(r) #Differences in r
Δr = np.insert(Δr, 0, 0)
ΔX = np.diff(X)[0]  # Assuming uniform grid spacing
X_N = X_N+N_n*ΔX
N = N+N_n #redefine N to match new X array

#FOR WHITE DWARF MODEL ONLY
# Disk parameters
R_1_wd = 5e8  # Inner radius of the disk (cm)
R_K_wd = 2.1e10  # Radius where initial mass is added (cm)
R_N_wd = 8e10  # Outer radius of the disk (cm)
M_dot_wd = 5e16  # Mass transfer rate (g/s)
X_1_wd = X_func(R_1_wd)
X_K_wd = X_func(R_K_wd)
X_N_wd = X_func(R_N_wd)
# Simulation parameters
N_wd = 100  # Number of grid points for the simulation
N_n_wd = 3 # Extra grid points after bath outer radius
# Define the radial grid
X_wd = np.linspace(X_1_wd, X_N_wd, N_wd)
ΔX_wd = np.diff(X_wd)[0]  # Assuming uniform grid spacing
X_wd = np.linspace(X_1_wd,X_N_wd+N_n_wd*ΔX_wd,N_wd+N_n_wd)
r_wd = R_func(X_wd)
Δr_wd = np.diff(r_wd)  # Assuming uniform grid spacing
Δr_wd = np.insert(Δr_wd, 0, 0)
ΔX_wd = np.diff(X_wd)[0]  # Assuming uniform grid spacing
X_N_wd = X_N_wd+N_n_wd*ΔX_wd
N_wd = N_wd+N_n_wd #redefine N to match new X array

def omega(Star_Mass, R):
    return np.sqrt(G * Star_Mass/ (R**3))

def kinematic_viscosity(H, R, alpha, Star_mass):
    return (2/3) * alpha * omega(Star_mass,R) * H**2
```


```python
#Cell to call all outburst arrays

#WHITE DWARF
df_0 = pd.read_csv('Sigma_history_bath_array_new_Fcs_final_newopac_b.csv')
Sigma_history_wd = df_0.to_numpy().tolist()
df_1 = pd.read_csv('Temp_history_bath_array_new_Fcs_final_newopac_b.csv')
Temp_history_wd = df_1.to_numpy().tolist()
df_2 = pd.read_csv('H_history_bath_array_new_Fcs_final_newopac_b.csv')
H_history_wd = df_2.to_numpy().tolist()
df_3 = pd.read_csv('alpha_history_bath_array_new_Fcs_final_newopac_b.csv')
alpha_history_wd = df_3.to_numpy().tolist()
df_4 = pd.read_csv('t_history_bath_array_new_Fcs_final_newopac_b.csv')
t_history_wd = df_4.to_numpy().flatten().tolist()
df_5 = pd.read_csv('Sigma_transfer_history_bath_array_new_Fcs_final_newopac_b.csv')
Sigma_transfer_history_wd = df_5.to_numpy().flatten().tolist()

#Black Hole base
df_0 = pd.read_csv('Sigma_history_bath_array_new_Fcs_final_newopac_bb.csv')
Sigma_history_bh = df_0.to_numpy().tolist()
df_1 = pd.read_csv('Temp_history_bath_array_new_Fcs_final_newopac_bb.csv')
Temp_history_bh = df_1.to_numpy().tolist()
df_2 = pd.read_csv('H_history_bath_array_new_Fcs_final_newopac_bb.csv')
H_history_bh = df_2.to_numpy().tolist()
df_3 = pd.read_csv('alpha_history_bath_array_new_Fcs_final_newopac_bb.csv')
alpha_history_bh = df_3.to_numpy().tolist()
df_4 = pd.read_csv('t_history_bath_array_new_Fcs_final_newopac_bb.csv')
t_history_bh = df_4.to_numpy().flatten().tolist()
df_5 = pd.read_csv('Sigma_transfer_history_bath_array_new_Fcs_final_newopac_bb.csv')
Sigma_transfer_history_bh = df_5.to_numpy().flatten().tolist()

#Black Hole base_2
df_0 = pd.read_csv('Sigma_history_bath_array_new_Fcs_final_newopac_bhne.csv')
Sigma_history_bhne = df_0.to_numpy().tolist()
df_1 = pd.read_csv('Temp_history_bath_array_new_Fcs_final_newopac_bhne.csv')
Temp_history_bhne = df_1.to_numpy().tolist()
df_2 = pd.read_csv('H_history_bath_array_new_Fcs_final_newopac_bhne.csv')
H_history_bhne = df_2.to_numpy().tolist()
df_3 = pd.read_csv('alpha_history_bath_array_new_Fcs_final_newopac_bhne.csv')
alpha_history_bhne = df_3.to_numpy().tolist()
df_4 = pd.read_csv('t_history_bath_array_new_Fcs_final_newopac_bhne.csv')
t_history_bhne = df_4.to_numpy().flatten().tolist()
df_5 = pd.read_csv('Sigma_transfer_history_bath_array_new_Fcs_final_newopac_bhne.csv')
Sigma_transfer_history_bhne = df_5.to_numpy().flatten().tolist()

#Black Hole irradiation
df_0 = pd.read_csv('Sigma_history_bath_array_new_Fcs_final_newopac_3bbb.csv')
Sigma_history_ir = df_0.to_numpy().tolist()
df_1 = pd.read_csv('Temp_history_bath_array_new_Fcs_final_newopac_3bbb.csv')
Temp_history_ir = df_1.to_numpy().tolist()
df_2 = pd.read_csv('H_history_bath_array_new_Fcs_final_newopac_3bbb.csv')
H_history_ir = df_2.to_numpy().tolist()
df_3 = pd.read_csv('alpha_history_bath_array_new_Fcs_final_newopac_3bbb.csv')
alpha_history_ir = df_3.to_numpy().tolist()
df_4 = pd.read_csv('t_history_bath_array_new_Fcs_final_newopac_3bbb.csv')
t_history_ir = df_4.to_numpy().flatten().tolist()
df_5 = pd.read_csv('Sigma_transfer_history_bath_array_new_Fcs_final_newopac_3bbb.csv')
Sigma_transfer_history_ir = df_5.to_numpy().flatten().tolist()

#Black Hole evaporation #4dbb
df_0 = pd.read_csv('Sigma_history_bath_array_new_Fcs_final_newopac_ev.csv')
Sigma_history_ev = df_0.to_numpy().tolist()
df_1 = pd.read_csv('Temp_history_bath_array_new_Fcs_final_newopac_ev.csv')
Temp_history_ev = df_1.to_numpy().tolist()
df_2 = pd.read_csv('H_history_bath_array_new_Fcs_final_newopac_ev.csv')
H_history_ev = df_2.to_numpy().tolist()
df_3 = pd.read_csv('alpha_history_bath_array_new_Fcs_final_newopac_ev.csv')
alpha_history_ev = df_3.to_numpy().tolist()
df_4 = pd.read_csv('t_history_bath_array_new_Fcs_final_newopac_ev.csv')
t_history_ev = df_4.to_numpy().flatten().tolist()
df_5 = pd.read_csv('Sigma_transfer_history_bath_array_new_Fcs_final_newopac_ev.csv')
Sigma_transfer_history_ev = df_5.to_numpy().flatten().tolist()

#Back Hole irr + evap
df_0 = pd.read_csv('Sigma_history_bath_array_new_Fcs_final_newopac_irev.csv')
Sigma_history_irev = df_0.to_numpy().tolist()
df_1 = pd.read_csv('Temp_history_bath_array_new_Fcs_final_newopac_irev.csv')
Temp_history_irev = df_1.to_numpy().tolist()
df_2 = pd.read_csv('H_history_bath_array_new_Fcs_final_newopac_irev.csv')
H_history_irev = df_2.to_numpy().tolist()
df_3 = pd.read_csv('alpha_history_bath_array_new_Fcs_final_newopac_irev.csv')
alpha_history_irev = df_3.to_numpy().tolist()
df_4 = pd.read_csv('t_history_bath_array_new_Fcs_final_newopac_irev.csv')
t_history_irev = df_4.to_numpy().flatten().tolist()
df_5 = pd.read_csv('Sigma_transfer_history_bath_array_new_Fcs_final_newopac_irev.csv')
Sigma_transfer_history_irev = df_5.to_numpy().flatten().tolist()
```


```python
#Cell to create Luminosities for all outburst arrays


#WHITE DWARF OUTBURST#
H_history_wd = np.array(H_history_wd)[:130000]
Sigma_history_wd = np.array(Sigma_history_wd)[:130000]
alpha_history_wd = np.array(alpha_history_wd)[:130000]
Temp_history_wd = np.array(Temp_history_wd)[:130000]
t_history_wd = np.array(t_history_wd)[:130000]

#BLACK HOLE OUTBURST#
H_history_bh = np.array(H_history_bh)
Sigma_history_bh = np.array(Sigma_history_bh)
alpha_history_bh = np.array(alpha_history_bh)
Temp_history_bh = np.array(Temp_history_bh)
t_history_bh = np.array(t_history_bh)

#BLACK HOLE OUTBURST 2
H_history_bhne = np.array(H_history_bhne)
Sigma_history_bhne = np.array(Sigma_history_bhne)
alpha_history_bhne = np.array(alpha_history_bhne)
Temp_history_bhne = np.array(Temp_history_bhne)
t_history_bhne = np.array(t_history_bhne)


#BH IRRADIATION OUTBURST#
H_history_ir = np.array(H_history_ir)
Sigma_history_ir = np.array(Sigma_history_ir)
alpha_history_ir = np.array(alpha_history_ir)
Temp_history_ir = np.array(Temp_history_ir)
t_history_ir = np.array(t_history_ir)

#BH EVAPORATION OUTBURST#
H_history_ev = np.array(H_history_ev)
Sigma_history_ev = np.array(Sigma_history_ev)
alpha_history_ev = np.array(alpha_history_ev)
Temp_history_ev = np.array(Temp_history_ev)
t_history_ev = np.array(t_history_ev)

#BH IRR + EVAP OUTBURST#
H_history_irev = np.array(H_history_irev)
Sigma_history_irev = np.array(Sigma_history_irev)
alpha_history_irev = np.array(alpha_history_irev)
Temp_history_irev = np.array(Temp_history_irev)
t_history_irev = np.array(t_history_irev)




#GENERAL FUNCTIONS DO NOT EDIT FURTHER
#Total luminosity of disk in a given timestep t_index
def L_rad(Sigma_history, H_history, alpha_history, r, t_index, Star_Mass):
    M_dot_array = 2 * np.pi * r * Sigma_history[t_index] * 3 * kinematic_viscosity(H_history[t_index], r, alpha_history[t_index],Star_Mass) / r
    L_rad0 = (3/2)*G*Star_Mass*M_dot_array*Δr/(r**2)
    return sum(L_rad0)

#Luminosity across hte disk in a given timestep
def L_rad_arr(Sigma_history, H_history, alpha_history, r, t_index, Star_Mass):
    M_dot_array = 2 * np.pi * r * Sigma_history[t_index] * 3 * kinematic_viscosity(H_history[t_index], r, alpha_history[t_index],Star_Mass) / r
    L_rad0 = (3/2)*G*Star_Mass*M_dot_array*Δr/(r**2)
    return L_rad0

def T_eff(Sigma_history, H_history, alpha_history, r, t_index, Star_Mass):
    M_dot_array = 2 * np.pi * r * Sigma_history[t_index] * 3 * kinematic_viscosity(H_history[t_index], r, alpha_history[t_index],Star_Mass) / r
    T_eff =  (((3*G*Star_Mass*M_dot_array) / (8*np.pi*sigma_SB*r**3)) * (1-np.sqrt(r[0]/r)))**(1/4)
    return T_eff



#CREATE ARRAYS

#WHITE DWARF
L_array_wd = np.empty(len(t_history_wd))
L_rad_array_wd = []
for i in range(len(L_array_wd)):
    L_array_wd[i] = L_rad(Sigma_history_wd, H_history_wd, alpha_history_wd, r_wd, i, M_wd)
for i in range(len(L_array_wd)):
    L_rad_array_wd.append(L_rad_arr(Sigma_history_wd, H_history_wd, alpha_history_wd, r_wd, i, M_wd))
T_eff_array_wd = []
for i in range(len(Sigma_history_wd)):
    T_eff_array_wd.append(T_eff(Sigma_history_wd, H_history_wd, alpha_history_wd, r_wd, i, M_wd))
T_eff_array_wd = np.vstack(T_eff_array_wd)
L_rad_array_wd = np.vstack(L_rad_array_wd)

#BLACK HOLE
L_array_bh = np.empty(len(t_history_bh))
L_rad_array_bh = []
for i in range(len(L_array_bh)):
    L_array_bh[i] = L_rad(Sigma_history_bh, H_history_bh, alpha_history_bh, r, i, M_bh)
for i in range(len(L_array_bh)):
    L_rad_array_bh.append(L_rad_arr(Sigma_history_bh, H_history_bh, alpha_history_bh, r, i, M_bh))
T_eff_array_bh = []
for i in range(len(Sigma_history_bh)):
    T_eff_array_bh.append(T_eff(Sigma_history_bh, H_history_bh, alpha_history_bh, r, i, M_bh))
T_eff_array_bh = np.vstack(T_eff_array_bh)
L_rad_array_bh = np.vstack(L_rad_array_bh)

#BLACK HOLE 2
L_array_bhne = np.empty(len(t_history_bhne))
L_rad_array_bhne = []
for i in range(len(L_array_bhne)):
    L_array_bhne[i] = L_rad(Sigma_history_bhne, H_history_bhne, alpha_history_bhne, r, i, M_bh)
for i in range(len(L_array_bhne)):
    L_rad_array_bhne.append(L_rad_arr(Sigma_history_bhne, H_history_bhne, alpha_history_bhne, r, i, M_bh))
T_eff_array_bhne = []
for i in range(len(Sigma_history_bhne)):
    T_eff_array_bhne.append(T_eff(Sigma_history_bhne, H_history_bhne, alpha_history_bhne, r, i, M_bh))
T_eff_array_bhne = np.vstack(T_eff_array_bhne)
L_rad_array_bhne = np.vstack(L_rad_array_bhne)

#IRRADIATION
L_array_ir = np.empty(len(t_history_ir))
L_rad_array_ir = []
for i in range(len(L_array_ir)):
    L_array_ir[i] = L_rad(Sigma_history_ir, H_history_ir, alpha_history_ir, r, i, M_bh)
for i in range(len(L_array_ir)):
    L_rad_array_ir.append(L_rad_arr(Sigma_history_ir, H_history_ir, alpha_history_ir, r, i, M_bh))
T_eff_array_ir = []
for i in range(len(Sigma_history_ir)):
    T_eff_array_ir.append(T_eff(Sigma_history_ir, H_history_ir, alpha_history_ir, r, i, M_bh))
T_eff_array_ir = np.vstack(T_eff_array_ir)
L_rad_array_ir = np.vstack(L_rad_array_ir)

#EVAPORATION
L_array_ev = np.empty(len(t_history_ev))
L_rad_array_ev = []
for i in range(len(L_array_ev)):
    L_array_ev[i] = L_rad(Sigma_history_ev, H_history_ev, alpha_history_ev, r, i, M_bh)
for i in range(len(L_array_ev)):
    L_rad_array_ev.append(L_rad_arr(Sigma_history_ev, H_history_ev, alpha_history_ev, r, i, M_bh))
T_eff_array_ev = []
for i in range(len(Sigma_history_ev)):
    T_eff_array_ev.append(T_eff(Sigma_history_ev, H_history_ev, alpha_history_ev, r, i, M_bh))
T_eff_array_ev = np.vstack(T_eff_array_ev)
L_rad_array_ev = np.vstack(L_rad_array_ev)

#IRR + EVAP
L_array_irev = np.empty(len(t_history_irev))
L_rad_array_irev = []
for i in range(len(L_array_irev)):
    L_array_irev[i] = L_rad(Sigma_history_irev, H_history_irev, alpha_history_irev, r, i, M_bh)
for i in range(len(L_array_irev)):
    L_rad_array_irev.append(L_rad_arr(Sigma_history_irev, H_history_irev, alpha_history_irev, r, i, M_bh))
T_eff_array_irev = []
for i in range(len(Sigma_history_irev)):
    T_eff_array_irev.append(T_eff(Sigma_history_irev, H_history_irev, alpha_history_irev, r, i, M_bh))
T_eff_array_irev = np.vstack(T_eff_array_irev)
L_rad_array_irev = np.vstack(L_rad_array_irev)
```


```python
# Function to plot the mass transfer history
def plot_mass_transfer(ax, Sigma_history, H_history, alpha_history, t_history, star, r, Star_Mass):
    y_values = np.array([2 * np.pi * r[1] * sigma_a[1] * 3 * kinematic_viscosity(H_a[1], r[1], alpha_a[1], Star_Mass) / r[1] 
                         for sigma_a, H_a, alpha_a in zip(Sigma_history, H_history, alpha_history)])
    x_values = np.array(t_history)/86400
    
    if star == 'wd':
        a1 = ax.plot(x_values, y_values, linewidth=2.5,linestyle = 'dashed')
    elif star == 'bh':
        a1 = ax.plot(x_values, y_values, linewidth=2.5,linestyle = (5, (10, 3)), color = 'k')        
    elif star == 'ir':
        a1 = ax.plot(x_values, y_values, linewidth=2.5,linestyle ='dashed', color = 'r')        
    elif star == 'ev':
        a1 = ax.plot(x_values, y_values, linewidth=2.5,linestyle ='dotted', color = 'g')    
    elif star == 'irev':
        a1 = ax.plot(x_values, y_values, linewidth=3, color = 'purple')      
    return a1

# Function to luminosity
def plot_Lum(ax, L_array, t_history, star, r):
    y_values = L_array
    x_values = t_history/86400
    if star == 'wd':
        a1 = ax.plot(x_values, y_values, linewidth=2.5,linestyle = 'dashed')
    elif star == 'bh':
        a1 = ax.plot(x_values, y_values, linewidth=2.5,linestyle = (5, (10, 3)), color = 'k')        
    elif star == 'ir':
        a1 = ax.plot(x_values, y_values, linewidth=2.5,linestyle ='dashed', color = 'r')        
    elif star == 'ev':
        a1 = ax.plot(x_values, y_values, linewidth=2.5,linestyle ='dotted', color = 'g')    
    elif star == 'irev':
        a1 = ax.plot(x_values, y_values, linewidth=3, color = 'purple')   
    return a1

#Function to plot radius
def plot_radius(ax, Temp_history, t_history, star, r):
    first_indices = np.full(Temp_history.shape[0], 1)  # Default to 1, indicating no index found
    # Iterate over each timestep
    for timestep in range(Temp_history.shape[0]):
    # For each timestep, ignore the first three radial positions and find the first index under 5,000
        radial_indices = np.where(Temp_history[timestep, 3:] < 5000)[0]
    # If there are any such indices, store the first one adjusted for the ignored positions
        if radial_indices.size > 0:
            first_indices[timestep] = radial_indices[0] + 3  # Adjust index by 3 to account for ignored positions
    # Since we're interested in timesteps starting from the third, the rest can be ignored for plotting
    # However, let's keep the full array for now to see the full context
    r_f_i = r[first_indices]
    y_values = r_f_i/1e10
    print(y_values)
    x_values = np.array(t_history)/86400
    if star == 'wd':
        a1 = ax.plot(x_values, y_values, linewidth=2.5,linestyle = 'dashed')
    elif star == 'bh':
        a1 = ax.plot(x_values, y_values, linewidth=2.5,linestyle = (5, (10, 3)), color = 'k')        
    elif star == 'ir':
        a1 = ax.plot(x_values, y_values, linewidth=2.5,linestyle ='dashed', color = 'r')        
    elif star == 'ev':
        a1 = ax.plot(x_values, y_values, linewidth=2.5,linestyle ='dotted', color = 'g')    
    elif star == 'irev':
        a1 = ax.plot(x_values, y_values, linewidth=3, color = 'purple')    
    return a1

font = {'weight' : 'normal',
        'size'   : 32}
plt.rc('font', **font)

fig, axs = plt.subplots(3, 1, figsize=(20, 16), gridspec_kw={'hspace': 0})

ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]

#Axis 1
ax1.set_xlabel('Time (days)')
ax1.set_ylabel(r'$\dot{M}_{in}$ (g s⁻¹)')
ax1.set_yscale('log')
ax1.set_ylim(1e12,2e16)
ax1.set_xlim(300,700)

#Axis 2
ax2.set_xlabel('t (days)')
ax2.set_ylabel(r'$L_{tot}$ (erg s$^{-1}$)')
ax2.set_yscale('log')
#ax1.set_ylim(1e12,2e16)
ax2.set_xlim(300,700)

#Axis 2
ax3.set_xlabel(r'$t$ (days)')
ax3.set_ylabel(r'$R_{out}$ (10$^{10}$cm)')
#ax3.set_yscale('log')
#ax1.set_ylim(1e12,2e16)
ax3.set_xlim(300,700)

#Hide tick labels of all x-axis except bottom row
ax1.set_xticklabels([])
ax2.set_xticklabels([])

#Set up major and minor inward ticks for all axes
def setup_ticks(ax):
    ax.tick_params(axis='both', which='major', direction='in', length=16, width=2, colors='black')
    ax.tick_params(axis='both', which='minor', direction='in', length=8, width=2, colors='gray')
    ax.tick_params(axis='both', which='both', top=True, right=True, direction='in')
    ax.tick_params(labeltop=False, labelright=False)  # Hide labels on top and right

setup_ticks(ax1)
setup_ticks(ax2)
setup_ticks(ax3)

# Call the plotting function
plot_mass_transfer(ax1,Sigma_history_wd, H_history_wd, alpha_history_wd, t_history_wd,'wd',r_wd,M_wd)
plot_Lum(ax2, L_array_wd, t_history_wd,'wd',r_wd)
plot_radius(ax3, Temp_history_wd, t_history_wd,'wd',r_wd)
plt.legend(frameon=False)
```


```python
fig, axs = plt.subplots(3, 1, figsize=(20, 16), gridspec_kw={'hspace': 0})

ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]

#Axis 1
ax1.set_xlabel('Time (days)')
ax1.set_ylabel(r'$\dot{M}_{in}$ (g s⁻¹)')
ax1.set_yscale('log')
ax1.set_ylim(1e12,2e16)
ax1.set_xlim(500,1000)

#Axis 2
ax2.set_xlabel('t (days)')
ax2.set_ylabel(r'$L_{tot}$ (erg s$^{-1}$)')
ax2.set_yscale('log')
ax2.set_ylim(1e30,1e35)
ax2.set_xlim(500,1000)

#Axis 2
ax3.set_xlabel(r'$t$ (days)')
ax3.set_ylabel(r'$R_{out}$ (10$^{10}$cm)')
#ax3.set_yscale('log')
ax3.set_ylim(0.1,2.1)
ax3.set_xlim(500,1000)

#Hide tick labels of all x-axis except bottom row
ax1.set_xticklabels([])
ax2.set_xticklabels([])

#Set up major and minor inward ticks for all axes
setup_ticks(ax1)
setup_ticks(ax2)
setup_ticks(ax3)

# Call the plotting function
plot_mass_transfer(ax1,Sigma_history_bh, H_history_bh, alpha_history_bh, t_history_bh,'wd',r,M_bh)
plot_Lum(ax2, L_array_bh, t_history_bh,'wd',r)
plot_radius(ax3, Temp_history_bh, t_history_bh,'wd',r)
plt.legend(frameon=False)
```


```python
#Combining white dwarf and black hole base
font = {'weight' : 'normal',
        'size'   : 32}
plt.rc('font', **font)
fig, axs = plt.subplots(3, 1, figsize=(20, 16), gridspec_kw={'hspace': 0})

ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]

#Axis 1
ax1.set_xlabel('t (years)')
ax1.set_ylabel(r'$\dot{M}_{in}$ (g s⁻¹)')
ax1.set_yscale('log')
ax1.set_ylim(1e12,2e16)
ax1.set_xlim(0,2000)

#Axis 2
ax2.set_xlabel('t (years)')
ax2.set_ylabel(r'$L_{tot}$ (erg s$^{-1}$)')
ax2.set_yscale('log')
ax2.set_ylim(1e31,1e35)
ax2.set_xlim(0,2000)

#Axis 2
ax3.set_xlabel(r'$t$ (days)')
ax3.set_ylabel(r'$R_{out}$ (10$^{10}$cm)')
ax3.set_yscale('log')
ax3.set_ylim(0.1,7)
ax3.set_xlim(0,2000)

#Hide tick labels of all x-axis except bottom row
ax1.set_xticklabels([])
ax2.set_xticklabels([])

#Set up major and minor inward ticks for all axes
setup_ticks(ax1)
setup_ticks(ax2)
setup_ticks(ax3)

# Call the plotting function
plot_mass_transfer(ax1,Sigma_history_bhne, H_history_bhne, alpha_history_bhne, t_history_bhne,'irev',r,M_bh)
plot_Lum(ax2, L_array_bhne, t_history_bhne,'irev',r)
plot_radius(ax3, Temp_history_bhne, t_history_bhne,'irev',r)
plot_mass_transfer(ax1,Sigma_history_wd, H_history_wd, alpha_history_wd, t_history_wd,'wd',r_wd,M_wd)
plot_Lum(ax2, L_array_wd, t_history_wd,'wd',r_wd)
plot_radius(ax3, Temp_history_wd, t_history_wd,'wd',r_wd)
plt.legend(frameon=False)
```


```python
# Function to plot the mass transfer history in years
def plot_mass_transfer_y(ax, Sigma_history, H_history, alpha_history, t_history, star, index,Star_Mass):
    
    y_values = np.array([2 * np.pi * r[index] * sigma_a[index] * 3 * kinematic_viscosity(H_a[index], r[index], alpha_a[index],Star_Mass) / r[index] 
                         for sigma_a, H_a, alpha_a in zip(Sigma_history, H_history, alpha_history)])
    x_values = np.array(t_history)/31536000
    
    if star == 'wd':
        a1 = ax.plot(x_values, y_values, linewidth=2.5)
    elif star == 'bh':
        a1 = ax.plot(x_values, y_values, linewidth=2.5,linestyle = (5, (10, 3)), color = 'k', label = 'no effects', alpha = 0.5)        
    elif star == 'ir':
        a1 = ax.plot(x_values, y_values, linewidth=2.5,linestyle ='dashed', color = 'r', label = 'irradiation', alpha = 0.7)        
    elif star == 'ev':
        a1 = ax.plot(x_values, y_values, linewidth=2.5,linestyle ='dotted', color = 'b', label = 'evaporation', alpha = 0.9)    
    elif star == 'irev':
        a1 = ax.plot(x_values, y_values, linewidth=3, color = 'purple', label = 'irr + evap')      
    return a1

# Function to luminosity
def plot_Lum_y(ax, L_array, t_history, star):
    y_values = L_array
    x_values = t_history/31536000
    if star == 'wd':
        a1 = ax.plot(x_values, y_values, linewidth=2.5)
    elif star == 'bh':
        a1 = ax.plot(x_values, y_values, linewidth=2.5,linestyle = (5, (10, 3)), color = 'k', label = 'no effects', alpha = 0.5)        
    elif star == 'ir':
        a1 = ax.plot(x_values, y_values, linewidth=2.5,linestyle ='dashed', color = 'r', label = 'irradiation', alpha = 0.7)        
    elif star == 'ev':
        a1 = ax.plot(x_values, y_values, linewidth=2.5,linestyle ='dotted', color = 'b', label = 'evaporation', alpha = 0.9)    
    elif star == 'irev':
        a1 = ax.plot(x_values, y_values, linewidth=3, color = 'purple', label = 'irr + evap')   
    return a1

#Function to plot radius
def plot_radius_y(ax, Temp_history, t_history, star,r):
    first_indices = np.full(Temp_history.shape[0], 1)  # Default to 1, indicating no index found
    # Iterate over each timestep
    for timestep in range(Temp_history.shape[0]):
    # For each timestep, ignore the first two radial positions and find the first index under 5,000
        radial_indices = np.where(Temp_history[timestep, 2:] < 5000)[0]
    # If there are any such indices, store the first one adjusted for the ignored positions
        if radial_indices.size > 0:
            first_indices[timestep] = radial_indices[0] + 2  # Adjust index by 2 to account for ignored positions
    # Since we're interested in timesteps starting from the third, the rest can be ignored for plotting
    # However, let's keep the full array for now to see the full context
    r_f_i = r[first_indices]
    y_values = r_f_i/1e10
    print(y_values)
    x_values = np.array(t_history)/31536000
    if star == 'wd':
        a1 = ax.plot(x_values, y_values, linewidth=2.5)
    elif star == 'bh':
        a1 = ax.plot(x_values, y_values, linewidth=2.5,linestyle = (5, (10, 3)), color = 'k', label = 'no effects', alpha = 0.5)        
    elif star == 'ir':
        a1 = ax.plot(x_values, y_values, linewidth=2.5,linestyle ='dashed', color = 'r', label = 'irradiation', alpha = 0.7)        
    elif star == 'ev':
        a1 = ax.plot(x_values, y_values, linewidth=2.5,linestyle ='dotted', color = 'b', label = 'evaporation', alpha = 0.9)    
    elif star == 'irev':
        a1 = ax.plot(x_values, y_values, linewidth=3, color = 'purple', label = 'irr + evap')    
    return a1

fig, axs = plt.subplots(3, 1, figsize=(20, 16), gridspec_kw={'hspace': 0})

ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]

#Axis 1
ax1.set_xlabel('Time (days)')
ax1.set_ylabel(r'$\dot{M}_{in}$ (g s⁻¹)')
ax1.set_yscale('log')
#ax1.set_ylim(1e12,2e16)
ax1.set_xlim(0,10)

#Axis 2
ax2.set_xlabel('t (days)')
ax2.set_ylabel(r'$L_{tot}$ (erg s$^{-1}$)')
ax2.set_yscale('log')
#ax2.set_ylim(1e32,1e35)
ax2.set_xlim(0,10)

#Axis 2
ax3.set_xlabel(r'$t$ (years)')
ax3.set_ylabel(r'$R_{out}$ (10$^{10}$cm)')
#ax3.set_yscale('log')
#ax3.set_ylim(0.1,0.7)
ax3.set_xlim(0,10)

#Hide tick labels of all x-axis except bottom row
ax1.set_xticklabels([])
ax2.set_xticklabels([])

#Set up major and minor inward ticks for all axes
setup_ticks(ax1)
setup_ticks(ax2)
setup_ticks(ax3)

# Call the plotting function
plot_mass_transfer_y(ax1,Sigma_history_ir, H_history_ir, alpha_history_ir, t_history_ir,'irev',1,M_bh)
plot_Lum_y(ax2, L_array_ir, t_history_ir,'irev')
plot_radius_y(ax3, Temp_history_ir, t_history_ir,'irev',r)
plt.legend(frameon=False)
```


```python
fig, axs = plt.subplots(3, 1, figsize=(20, 16), gridspec_kw={'hspace': 0})

ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]

#Axis 1
ax1.set_xlabel('Time (days)')
ax1.set_ylabel(r'$\dot{M}_{in}$ (g s⁻¹)')
ax1.set_yscale('log')
#ax1.set_ylim(1e12,2e16)
ax1.set_xlim(0,10)

#Axis 2
ax2.set_xlabel('t (days)')
ax2.set_ylabel(r'$L_{tot}$ (erg s$^{-1}$)')
ax2.set_yscale('log')
#ax2.set_ylim(1e32,1e35)
ax2.set_xlim(0,10)

#Axis 2
ax3.set_xlabel(r'$t$ (years)')
ax3.set_ylabel(r'$R_{out}$ (10$^{10}$cm)')
#ax3.set_yscale('log')
#ax3.set_ylim(0.1,0.7)
ax3.set_xlim(0,10)

#Hide tick labels of all x-axis except bottom row
ax1.set_xticklabels([])
ax2.set_xticklabels([])

#Set up major and minor inward ticks for all axes
setup_ticks(ax1)
setup_ticks(ax2)
setup_ticks(ax3)

# Call the plotting function
plot_mass_transfer_y(ax1,Sigma_history_ev, H_history_ev, alpha_history_ev, t_history_ev,'irev',5,M_bh)
plot_Lum_y(ax2, L_array_ev, t_history_ev,'irev')
plot_radius_y(ax3, Temp_history_ev, t_history_ev,'irev',r)
plt.legend(frameon=False)
```


```python
plt.rcParams.update({'font.size': 32})

fig, axs = plt.subplots(3, 1, figsize=(20, 16), gridspec_kw={'hspace': 0})

ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]

#Axis 1
ax1.set_xlabel('Time (days)')
ax1.set_ylabel(r'$\dot{M}_{in}$ (g s⁻¹)')
ax1.set_yscale('log')
#ax1.set_ylim(1e12,2e16)
ax1.set_xlim(0,20)

#Axis 2
ax2.set_xlabel('t (days)')
ax2.set_ylabel(r'$L_{tot}$ (erg s$^{-1}$)')
ax2.set_yscale('log')
#ax2.set_ylim(1e32,1e35)
ax2.set_xlim(0,20)

#Axis 2
ax3.set_xlabel(r'$t$ (years)')
ax3.set_ylabel(r'$R_{out}$ (10$^{10}$cm)')
#ax3.set_yscale('log')
#ax3.set_ylim(0.1,0.7)
ax3.set_xlim(0,20)

#Hide tick labels of all x-axis except bottom row
ax1.set_xticklabels([])
ax2.set_xticklabels([])

#Set up major and minor inward ticks for all axes
setup_ticks(ax1)
setup_ticks(ax2)
setup_ticks(ax3)

# Call the plotting function
plot_mass_transfer_y(ax1,Sigma_history_irev, H_history_irev, alpha_history_irev, t_history_irev,'irev',5,M_bh)
plot_Lum_y(ax2, L_array_irev, t_history_irev,'irev')
plot_radius_y(ax3, Temp_history_irev, t_history_irev,'irev',r)
plt.legend(frameon=False)
```


```python
plt.rcParams.update({'font.size': 56})

fig, axs = plt.subplots(3, 1, figsize=(40, 32), gridspec_kw={'hspace': 0})

ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]

#Axis 1
ax1.set_xlabel('Time (days)')
ax1.set_ylabel(r'$\dot{M}_{in}$ (g s⁻¹)')
ax1.set_yscale('log')
#ax1.set_ylim(1e12,2e16)
ax1.set_xlim(0,20)

#Axis 2
ax2.set_xlabel('t (days)')
ax2.set_ylabel(r'$L_{tot}$ (erg s$^{-1}$)')
ax2.set_yscale('log')
#ax2.set_ylim(1e32,1e35)
ax2.set_xlim(0,20)

#Axis 2
ax3.set_xlabel(r'$t$ (years)')
ax3.set_ylabel(r'$R_{out}$ (10$^{10}$cm)')
#ax3.set_yscale('log')
ax3.set_ylim(0.1,2.9)
ax3.set_xlim(0,20)

#Hide tick labels of all x-axis except bottom row
ax1.set_xticklabels([])
ax2.set_xticklabels([])

#new ticks to account for increase in scale
def setup_ticks_l(ax):
    ax.tick_params(axis='both', which='major', direction='in', length=28, width=4, colors='black')
    ax.tick_params(axis='both', which='minor', direction='in', length=14, width=4, colors='gray')
    ax.tick_params(axis='both', which='both', top=True, right=True, direction='in')
    ax.tick_params(labeltop=False, labelright=False)  # Hide labels on top and right
    
#Set up major and minor inward ticks for all axes
setup_ticks_l(ax1)
setup_ticks_l(ax2)
setup_ticks_l(ax3)

# Call the plotting function
plot_mass_transfer_y(ax1,Sigma_history_bhne, H_history_bhne, alpha_history_bhne, t_history_bhne,'bh',1,M_bh)
plot_Lum_y(ax2, L_array_bhne, t_history_bhne,'bh')
plot_radius_y(ax3, Temp_history_bhne, t_history_bhne,'bh',r)
plot_mass_transfer_y(ax1,Sigma_history_ir, H_history_ir, alpha_history_ir, t_history_ir,'ir',1,M_bh)
plot_Lum_y(ax2, L_array_ir, t_history_ir,'ir')
plot_radius_y(ax3, Temp_history_ir, t_history_ir,'ir',r)
plot_mass_transfer_y(ax1,Sigma_history_ev, H_history_ev, alpha_history_ev, t_history_ev,'ev',2,M_bh)
plot_Lum_y(ax2, L_array_ev, t_history_ev,'ev')
plot_radius_y(ax3, Temp_history_ev, t_history_ev,'ev',r)
plot_mass_transfer_y(ax1,Sigma_history_irev, H_history_irev, alpha_history_irev, t_history_irev,'irev',2,M_bh)
plot_Lum_y(ax2, L_array_irev, t_history_irev,'irev')
plot_radius_y(ax3, Temp_history_irev, t_history_irev,'irev',r)

ax1.legend(frameon=False, loc='upper right', fontsize ='29')
```


```python
plt.rcParams.update({'font.size': 56})

fig, axs = plt.subplots(3, 1, figsize=(40, 32), gridspec_kw={'hspace': 0})

ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]

#Axis 1
ax1.set_xlabel('Time (days)')
ax1.set_ylabel(r'$\dot{M}_{in}$ (g s⁻¹)')
ax1.set_yscale('log')
#ax1.set_ylim(1e12,2e16)
ax1.set_xlim(11,13)

#Axis 2
ax2.set_xlabel('t (days)')
ax2.set_ylabel(r'$L_{tot}$ (erg s$^{-1}$)')
ax2.set_yscale('log')
#ax2.set_ylim(1e32,1e35)
ax2.set_xlim(11,13)

#Axis 2
ax3.set_xlabel(r'$t$ (years)')
ax3.set_ylabel(r'$R_{out}$ (cm/10$^{10}$)')
#ax3.set_yscale('log')
ax3.set_ylim(0.1,2.9)
ax3.set_xlim(11,13)

#Hide tick labels of all x-axis except bottom row
ax1.set_xticklabels([])
ax2.set_xticklabels([])

#new ticks to account for increase in scale
def setup_ticks_l(ax):
    ax.tick_params(axis='both', which='major', direction='in', length=28, width=4, colors='black')
    ax.tick_params(axis='both', which='minor', direction='in', length=14, width=4, colors='gray')
    ax.tick_params(axis='both', which='both', top=True, right=True, direction='in')
    ax.tick_params(labeltop=False, labelright=False)  # Hide labels on top and right
    
#Set up major and minor inward ticks for all axes
setup_ticks_l(ax1)
setup_ticks_l(ax2)
setup_ticks_l(ax3)

# Call the plotting function
plot_mass_transfer_y(ax1,Sigma_history_bhne, H_history_bhne, alpha_history_bhne, t_history_bhne,'bh',1,M_bh)
plot_Lum_y(ax2, L_array_bhne, t_history_bhne,'bh')
plot_radius_y(ax3, Temp_history_bhne, t_history_bhne,'bh')
plot_mass_transfer_y(ax1,Sigma_history_ir, H_history_ir, alpha_history_ir, t_history_ir,'ir',1,M_bh)
plot_Lum_y(ax2, L_array_ir, t_history_ir,'ir')
plot_radius_y(ax3, Temp_history_ir, t_history_ir,'ir')
plot_mass_transfer_y(ax1,Sigma_history_ev, H_history_ev, alpha_history_ev, t_history_ev,'ev',2,M_bh)
plot_Lum_y(ax2, L_array_ev, t_history_ev,'ev')
plot_radius_y(ax3, Temp_history_ev, t_history_ev,'ev')
plot_mass_transfer_y(ax1,Sigma_history_irev, H_history_irev, alpha_history_irev, t_history_irev,'irev',2,M_bh)
plot_Lum_y(ax2, L_array_irev, t_history_irev,'irev')
plot_radius_y(ax3, Temp_history_irev, t_history_irev,'irev')

ax1.legend(frameon=False, loc='upper right', fontsize ='29')
```


```python
import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 150.0

font = {'weight' : 'normal',
        'size'   : 24}

indices_ir = np.linspace(5000, len(L_rad_array_ir)-1, 180, endpoint=False).astype(int)

n_angles = 50
angles = np.linspace(0, 2 * np.pi, n_angles)
R_mesh, Theta = np.meshgrid(r[:50], angles)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(12, 8))
ax.xaxis.grid(False)
ax.set_yticks(r[::4])
ax.set_yticklabels([])  # Remove radial labels
ax.set_xticklabels([])  # Remove angular labels
c = ax.pcolormesh(Theta, R_mesh, np.zeros_like(R_mesh), shading='auto', cmap='hot',vmin=np.min(L_rad_array_ir), vmax=np.max(L_rad_array_ir[5000:]))
cb = plt.colorbar(c, label='Luminosity (erg s$^{-1}$)')

def update_circle_lum_ir(frame):
    # Update the temperature values for the current frame/timestep
    frame_index = indices_ir[frame]
    Temp = np.tile(L_rad_array_ir[frame_index, :][:50], (n_angles, 1))
    c.set_array(Temp.flatten())
    ax.set_title(f'Time (days) = {int(t_history_ir[frame_index]/86400)}')
    return c,

ani_circle_lum_ir = FuncAnimation(fig, update_circle_lum_ir, frames=len(indices_ir), blit=True)

# To save the animation, uncomment the next line (requires ffmpeg or pillow)
# ani.save('temperature_animation.mp4', writer='ffmpeg')

plt.show()
```


```python
from IPython.display import HTML
HTML(ani_circle_lum_ir.to_jshtml())
```


```python
import matplotlib.colors as mcolors
import matplotlib

matplotlib.rcParams['animation.embed_limit'] = 150.0

font = {'weight' : 'normal',
        'size'   : 24}

indices_ir = np.linspace(5000, len(L_rad_array_ir)-1, 180, endpoint=False).astype(int)

# Ensure all your luminosity values are > 0 before taking the log
L_rad_array_ir[L_rad_array_ir <= 0] = np.nan
log_L_rad_array_ir = np.log10(L_rad_array_ir)  # Applying log10

n_angles = 50
angles = np.linspace(0, 2 * np.pi, n_angles)
R_mesh, Theta = np.meshgrid(r, angles)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(12, 8))
ax.xaxis.grid(False)
ax.set_yticks(r[::4])
ax.set_yticklabels([])  # Remove radial labels
ax.set_xticklabels([])  # Remove angular labels

# Use LogNorm for the normalization of the color scale
norm = mcolors.LogNorm(vmin=28, vmax=33)
c = ax.pcolormesh(Theta, R_mesh, np.zeros_like(R_mesh), shading='auto', cmap='hot', norm=norm)

# Create colorbar with logarithmic ticks
cb = plt.colorbar(c, label=r'log$_{10}(L)$')
cb_ticks = np.linspace(28, 33, 6)  # Modify this as needed for your actual data range
cb.set_ticks(cb_ticks)
cb_ticklabels = [f"{x:.0f}" for x in cb_ticks]  
cb.set_ticklabels(cb_ticklabels)


# ... Rest of your code for animation here ...

# Adjust update function to use the log luminosity data
def update_circle_lum_ir(frame):
    frame_index = indices_ir[frame]
    Temp_log = np.log10(L_rad_array_ir[frame_index, :])  # Logarithmic scale
    Temp_log = np.tile(Temp_log, (n_angles, 1))  # Tiling the log luminosity data
    c.set_array(Temp_log.flatten())
    time_days = int((t_history_ir[frame_index]-t_history_ir[5000])/86400)
    if time_days < 730:
        ax.set_title(f'Time (days) = {time_days}',fontsize = '28')
    else:
        ax.set_title(f'Time (years) = {round(time_days/365,2)}',fontsize = '28')
    return c,

ani_circle_lum_ir = FuncAnimation(fig, update_circle_lum_ir, frames=len(indices_ir), blit=True)

# To save the animation, uncomment the next line (requires ffmpeg or pillow)
# ani.save('temperature_animation.mp4', writer='ffmpeg')

plt.show()
```


```python
from IPython.display import HTML
HTML(ani_circle_lum_ir.to_jshtml())
```


```python
ani_circle_lum_ir.save('temp_radial_grid_irradiated_bh.gif', fps=60, writer='pillow')
```


```python
#Finding peaks of the data
from scipy.signal import find_peaks

Sigma_wd_index1 = Sigma_history_wd[:,1]
Sigma_bhne_i1 = Sigma_history_bhne[:,1]
Sigma_ir_i1 = Sigma_history_ir[:,1]
Sigma_ev_i1 = Sigma_history_ev[:,2]
Sigma_irev_i1 = Sigma_history_irev[:,2]
```


```python
wd_peaks, _ = find_peaks(Sigma_wd_index1, height=30, distance=100)
wd_peaks_times = t_history_wd[wd_peaks]
bhne_peaks, _ = find_peaks(Sigma_bhne_i1, height=1, distance=1000)
bhne_peaks_times = t_history_bhne[bhne_peaks]
ir_peaks, _ = find_peaks(Sigma_ir_i1, height=1, distance=1000)
ir_peaks_times = t_history_ir[ir_peaks]
ev_peaks, _ = find_peaks(Sigma_ev_i1, height=1, distance=1000)
ev_peaks_times = t_history_ev[ev_peaks]
irev_peaks, _ = find_peaks(Sigma_irev_i1, height=1, distance=1000)
irev_peaks_times = t_history_irev[irev_peaks]
bhne_peaks_times = bhne_peaks_times[:-1]
ir_peaks_times = ir_peaks_times[:-1]

wd_peaks_times,bhne_peaks_times,ir_peaks_times,ev_peaks_times,irev_peaks_times
```


```python
#BLACK HOLES ONLY

outburst_times_nowd = {
    'BH No Effects': bhne_peaks_times,
    'Evaporation': ev_peaks_times,
    'Irradiation': ir_peaks_times,
    'Irr & Evap': irev_peaks_times
}

averages_nowd = {}
standard_errors_nowd = {}

for model_nowd in outburst_times_nowd:
    periods_nowd = np.diff(outburst_times_nowd[model_nowd])# Calculate periods between successive peaks
    average_period_nowd = np.mean(periods_nowd)  # Average period
    std_deviation_nowd = np.std(periods_nowd, ddof=1)  # Sample standard deviation
    std_error_nowd = std_deviation_nowd / np.sqrt(len(periods_nowd))  # Standard error

    averages_nowd[model_nowd] = average_period_nowd
    standard_errors_nowd[model_nowd] = std_error_nowd

averages_nowd, standard_errors_nowd

#WHITE DWARF ONLY

outburst_times_wd = {
    'White Dwarf': wd_peaks_times,
}

averages_wd = {}
standard_errors_wd = {}

for model_wd in outburst_times_wd:
    periods_wd = np.diff(outburst_times_wd[model_wd])# Calculate periods between successive peaks
    average_period_wd = np.mean(periods_wd)  # Average period
    std_deviation_wd = np.std(periods_wd, ddof=1)  # Sample standard deviation
    std_error_wd = std_deviation_wd / np.sqrt(len(periods_wd))  # Standard error

    averages_wd[model_wd] = average_period_wd
    standard_errors_wd[model_wd] = std_error_wd

averages_wd, standard_errors_wd

outburst_times = {
    'White Dwarf': wd_peaks_times,
    'BH No Effects': bhne_peaks_times,
    'Evaporation': ev_peaks_times,
    'Irradiation': ir_peaks_times,
    'Irr & Evap': irev_peaks_times
}

averages = {}
standard_errors = {}

for model in outburst_times:
    periods = np.diff(outburst_times[model])  # Calculate periods between successive peaks
    average_period = np.mean(periods)  # Average period
    std_deviation = np.std(periods, ddof=1)  # Sample standard deviation
    std_error = std_deviation / np.sqrt(len(periods))  # Standard error

    averages[model] = average_period
    standard_errors[model] = std_error

averages, standard_errors
```


```python
plt.rcParams.update({'font.size': 24})

# Extract the model names, average periods, and errors
models = list(averages.keys())
average_periods = [averages[model] for model in models]
errors = [standard_errors[model] for model in models]

# Create the plot
plt.figure(figsize=(30, 10))
# Plot the error bars
plt.errorbar(models, np.array(average_periods)/86400, yerr=np.array(errors)/86400, fmt='o', capsize=12, color='orange', ecolor='purple', elinewidth=5, capthick=3,markersize=10)
plt.plot(models, np.array(average_periods)/86400, linewidth = 1.5, color = 'red', linestyle ='dashed',alpha=0.5)
# Add the axis labels and title
plt.xlabel('Model', fontsize=36)
plt.ylabel('Average Period (days)', fontsize=36)
plt.title('Average Period Between Outbursts for Different Models', fontsize=40)

plt.yscale('log')
plt.ylim(.9e2,2e3)
# Add grid for better readability
#plt.grid(True, which='both', linestyle='--', linewidth=0.5)
setup_ticks(plt)
# Display the plot
plt.show()
```


```python
# Extract the model names, average periods, and errors
models_nowd = list(averages_nowd.keys())
average_periods_nowd = [averages_nowd[model_nowd] for model_nowd in models_nowd]
errors_nowd = [standard_errors_nowd[model_nowd] for model_nowd in models_nowd]

# Create the plot
plt.figure(figsize=(30, 15))
# Plot the error bars
plt.errorbar(models_nowd, np.array(average_periods_nowd)/31536000 , yerr=np.array(errors_nowd)/31536000, fmt='o', capsize=8, color='orange', ecolor='purple', elinewidth=5, capthick=8,markersize=10)
plt.plot(models_nowd, np.array(average_periods_nowd)/31536000, linewidth = 2, color = 'red', linestyle ='dashed',alpha=0.5)
# Add the axis labels and title
plt.xlabel('Model', fontsize=36)
plt.ylabel(r'Average Period (yrs)', fontsize=36)
plt.title('Average Period Between Outbursts for Different Models', fontsize=40)
setup_ticks(plt)
# Add grid for better readability
#plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Display the plot
plt.show()
```


```python
plt.rcParams.update({'font.size': 32})

models_wd = list(averages_wd.keys())
average_periods_wd = [averages_wd[model_wd] for model_wd in models_wd]
errors_wd = [standard_errors_wd[model_wd] for model_wd in models_wd]

from matplotlib import gridspec
fig = plt.figure(figsize=(24,20))
gs = gridspec.GridSpec(2, 2, width_ratios=[0.4, 1], height_ratios=[1, 1], hspace=0.1)

# Axes for top big figure
ax1 = fig.add_subplot(gs[0, :])

# Axes for the bottom subplots with shared x-axis
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])  # Share x-axis with ax1

#Axis 1 parameters
ax1.errorbar(models, np.array(average_periods)/86400, yerr=np.array(errors)/86400, fmt='o', capsize=12, color='orange', ecolor='purple', elinewidth=5, capthick=3,markersize=10)
ax1.plot(models, np.array(average_periods)/86400, linewidth = 1.5, color = 'red', linestyle ='dashed',alpha=0.5)
#ax1.set_xlabel('Model', fontsize=36)
ax1.set_ylabel(r'$\bar{T}$ (days)', fontsize=36)
#ax1.set_title('Average Period Between Outbursts for Different Models', fontsize=40)
ax1.set_yscale('log')
ax1.set_ylim(.9e2,2e3)
setup_ticks(ax1)

#Axis 2 parameters
ax2.errorbar(models_wd, np.array(average_periods_wd)/86400, yerr=np.array(errors_wd)/86400, fmt='o', capsize=16, color='orange', ecolor='purple', elinewidth=5, capthick=8,markersize=10)
#ax2.set_xlabel('Model', fontsize=36)
ax2.set_ylabel(r'$\bar{T}$ (days)', fontsize=36)
ax2.set_ylim(114.85,116.1)
setup_ticks(ax2)

#Axis 3 parameters
ax3.errorbar(models_nowd, np.array(average_periods_nowd)/31536000 , yerr=np.array(errors_nowd)/31536000, fmt='o', capsize=16, color='orange', ecolor='purple', elinewidth=5, capthick=8,markersize=10)
ax3.plot(models_nowd, np.array(average_periods_nowd)/31536000, linewidth = 2, color = 'red', linestyle ='dashed',alpha=0.5)
#ax3.set_xlabel('Model', fontsize=36)
ax3.set_ylabel(r'$\bar{T}$ (yrs)', fontsize=36)
#ax3.set_title('Average Period Between Outbursts for Different Models', fontsize=40)
setup_ticks(ax3)
```


```python
# Compute evenly spaced indices for 180 frames
indices_irev = np.linspace(0, len(Sigma_history_irev)-1, 180, endpoint=False).astype(int)

fig, ax = plt.subplots(figsize=(16, 6))
line, = ax.plot([], [], lw=2)

# Initialization function: plot the background of each frame
def init_s_irev():
#    ax.clear()
    ax.set_xlim(np.min(r), R_N)
    ax.set_ylim(0.3, np.max(Sigma_history_irev))
    ax.set_xlabel(r'$R$ (cm)')
    ax.set_ylabel(r'$\Sigma$ (g cm⁻²)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    setup_ticks(ax)
    # Initialize Sigma line with specific style
    global line_s
    line_s, = ax.plot([], [], lw=3, color='k')
    return (line,)

def init_t_irev():
#    ax.clear()
    ax.set_xlim(np.min(r), R_N)
    ax.set_ylim(30, np.max(Temp_history_irev))
    ax.set_xlabel(r'$R$ (cm)')
    ax.set_ylabel(r'$T$ (K)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    setup_ticks(ax)
    # Initialize t line with specific style
    global line_t
    line_t, = ax.plot([], [], lw=3, color='r')
    return (line,)

def init_h_irev():
#    ax.clear()
    ax.set_xlim(np.min(r), R_N)
    ax.set_ylim(np.min(H_history_irev), np.max(H_history_irev))
    ax.set_xlabel(r'$R$ (cm)')
    ax.set_ylabel(r'$H$ (cm)')
    ax.set_xscale('log')    
    ax.set_yscale('log')
    setup_ticks(ax)
    # Initialize H line with specific style
    global line_h
    line_h, = ax.plot([], [], lw=3, color='b')
    return (line,)

# Animation function. This calls a subset of frames.
def animate_s_irev(i):
    # Use the selected index for the current frame
    frame_index = indices_irev[i]
    line_s.set_data(r, Sigma_history_irev[frame_index, :])
    ax.set_title(f'$t$ (yrs) = {round(t_history_irev[frame_index]/31536000,2)}')
    return (line,)

def animate_t_irev(i):
    # Use the selected index for the current frame
    frame_index = indices_irev[i]
    line_t.set_data(r, Temp_history_irev[frame_index, :])
    ax.set_title(f'$t$ (yrs) = {round(t_history_irev[frame_index]/31536000,2)}')
    return (line,)

def animate_h_irev(i):
    # Use the selected index for the current frame
    frame_index = indices_irev[i]
    line_h.set_data(r, H_history_irev[frame_index, :])
    ax.set_title(f'$t$ (yrs) = {round(t_history_irev[frame_index]/31536000,2)}')
    return (line,)


# Call the animator. blit=True means only re-draw the parts that have changed.
ani_s = FuncAnimation(fig, animate_s_irev, init_func=init_s_irev, frames=len(indices_irev), interval=20, blit=True)
ani_t = FuncAnimation(fig, animate_t_irev, init_func=init_t_irev, frames=len(indices_irev), interval=20, blit=True)
ani_h = FuncAnimation(fig, animate_h_irev, init_func=init_h_irev, frames=len(indices_irev), interval=20, blit=True)

# To display the animation in a Jupyter notebook, use the following:
from IPython.display import HTML

# To save the animation as an MP4 file
# ani.save('surface_density_animation.mp4', fps=30)  # fps = frames per second

# Show the plot (not needed if you use HTML display for Jupyter notebooks)
plt.show()
```


```python
HTML(ani_s.to_jshtml())
```


```python
HTML(ani_t.to_jshtml())
```


```python
#Supermassive Black Hole
df_0 = pd.read_csv('Sigma_history_bath_array_new_Fcs_final_newopac_smbh_1.csv')
Sigma_history_smbh_1 = df_0.to_numpy().tolist()
df_1 = pd.read_csv('Temp_history_bath_array_new_Fcs_final_newopac_smbh_1.csv')
Temp_history_smbh_1 = df_1.to_numpy().tolist()
df_2 = pd.read_csv('H_history_bath_array_new_Fcs_final_newopac_smbh_1.csv')
H_history_smbh_1 = df_2.to_numpy().tolist()
df_3 = pd.read_csv('alpha_history_bath_array_new_Fcs_final_newopac_smbh_1.csv')
alpha_history_smbh_1 = df_3.to_numpy().tolist()
df_4 = pd.read_csv('t_history_bath_array_new_Fcs_final_newopac_smbh_1.csv')
t_history_smbh_1 = df_4.to_numpy().flatten().tolist()
df_5 = pd.read_csv('Sigma_transfer_history_bath_array_new_Fcs_final_newopac_smbh_1.csv')
Sigma_transfer_history_smbh_1 = df_5.to_numpy().flatten().tolist()

#smbh_1
H_history_smbh_1 = np.array(H_history_smbh_1)
Sigma_history_smbh_1 = np.array(Sigma_history_smbh_1)
alpha_history_smbh_1 = np.array(alpha_history_smbh_1)
Temp_history_smbh_1 = np.array(Temp_history_smbh_1)
t_history_smbh_1 = np.array(t_history_smbh_1)


#R of smbh_1
# Disk parameters
R_1_smbh_1 = 4e12  # Inner radius of the disk (cm)
R_K_smbh_1 = 1e15  # Radius where initial mass is added (cm)
R_N_smbh_1 = 2e15  # Outer radius of the disk (cm)
X_1_smbh_1 = X_func(R_1_smbh_1)
X_K_smbh_1 = X_func(R_K_smbh_1)
X_N_smbh_1 = X_func(R_N_smbh_1)
N_smbh_1 = 100  # Number of grid points for the simulation
N_n_smbh_1 = 3 # Extra grid points after bath outer radius
# Define the radial grid
X_smbh_1 = np.linspace(X_1_smbh_1, X_N_smbh_1, N_smbh_1)
ΔX_smbh_1 = np.diff(X_smbh_1)[0]  # Assuming uniform grid spacing
X_smbh_1 = np.linspace(X_1_smbh_1,X_N_smbh_1+N_n_smbh_1*ΔX_smbh_1,N_smbh_1+N_n_smbh_1)
r_smbh_1 = R_func(X_smbh_1)
Δr_smbh_1 = np.diff(r_smbh_1)  # Assuming uniform grid spacing
Δr_smbh_1 = np.insert(Δr_smbh_1, 0, 0)
ΔX_smbh_1 = np.diff(X_smbh_1)[0]  # Assuming uniform grid spacing
X_N_smbh_1 = X_N_smbh_1+N_n_smbh_1*ΔX_smbh_1
N_smbh_1 = N_smbh_1+N_n_smbh_1 #redefine N to match new X array

#smbh_1
L_array_smbh_1 = np.empty(len(t_history_smbh_1))
L_rad_array_smbh_1 = []
for i in range(len(L_array_smbh_1)):
    L_array_smbh_1[i] = L_rad(Sigma_history_smbh_1, H_history_smbh_1, alpha_history_smbh_1, r_smbh_1, i,M_smbh)
for i in range(len(L_array_smbh_1)):
    L_rad_array_smbh_1.append(L_rad_arr(Sigma_history_smbh_1, H_history_smbh_1, alpha_history_smbh_1, r_smbh_1, i,M_smbh))
T_eff_array_smbh_1 = []
for i in range(len(Sigma_history_smbh_1)):
    T_eff_array_smbh_1.append(T_eff(Sigma_history_smbh_1, H_history_smbh_1, alpha_history_smbh_1, r_smbh_1, i,M_smbh))
T_eff_array_smbh_1 = np.vstack(T_eff_array_smbh_1)
L_rad_array_smbh_1 = np.vstack(L_rad_array_smbh_1)
```


```python
# Function to plot the mass transfer history in years
def plot_mass_transfer_smbh(ax, Sigma_history, H_history, alpha_history, t_history, index, r,Star_Mass,tag,labeltype):
    
    y_values = np.array([2 * np.pi * r[index] * sigma_a[index] * 3 * kinematic_viscosity(H_a[index], r[index], alpha_a[index],Star_Mass) / r[index] 
                         for sigma_a, H_a, alpha_a in zip(Sigma_history, H_history, alpha_history)])
    x_values = np.array(t_history)/31536000
    
    if tag == 1 and labeltype == 0:
        a1 = ax.plot(x_values, y_values, linewidth=3.5,alpha=1, label='4')#, color = 'red') 
    elif tag == 1 and labeltype == 1:
        a1 = ax.plot(x_values, y_values, linewidth=3.5,alpha=1, label='0.04')#, color = 'red') 
    elif tag == 2 and labeltype == 0:
        a1 = ax.plot(x_values, y_values, linewidth=3,alpha=0.9, label='5')#, color = 'red') 
    elif tag == 3 and labeltype == 0:
        a1 = ax.plot(x_values, y_values, linewidth=2.5,alpha=0.8, label='6')#, color = 'red') 
    elif tag == 4 and labeltype == 0:
        a1 = ax.plot(x_values, y_values, linewidth=2,alpha=0.7, label='7')#, color = 'red') 
    elif tag == 5 and labeltype == 0:
        a1 = ax.plot(x_values, y_values, linewidth=3,alpha=0.9,linestyle ='dashed', label='4')#, color = 'red') 
    elif tag == 5 and labeltype == 1:
        a1 = ax.plot(x_values, y_values, linewidth=3,alpha=0.9,linestyle ='dashed', label='0.02')#, color = 'red') 
        
    return a1

# Function to luminosity
def plot_Lum_smbh(ax, L_array, t_history,tag):
    y_values = L_array
    x_values = t_history/31536000

    if tag == 1:
        a1 = ax.plot(x_values, y_values, linewidth=3.5,alpha=1)#, color = 'red') 
    elif tag == 2:
        a1 = ax.plot(x_values, y_values, linewidth=3,alpha=0.9)#, color = 'red') 
    elif tag == 3:
        a1 = ax.plot(x_values, y_values, linewidth=2.5,alpha=0.8)#, color = 'red') 
    elif tag == 4:
        a1 = ax.plot(x_values, y_values, linewidth=2,alpha=0.7)#, color = 'red') 
    elif tag == 5:
        a1 = ax.plot(x_values, y_values, linewidth=3,alpha=0.9,linestyle ='dashed')#, color = 'red') 
    
    return a1

#Function to plot radius
def plot_radius_smbh(ax, Temp_history, t_history,r,tag):
    first_indices = np.full(Temp_history.shape[0], 1)  # Default to 1, indicating no index found
    # Iterate over each timestep
    for timestep in range(Temp_history.shape[0]):
    # For each timestep, ignore the first two radial positions and find the first index under 5,000
        radial_indices = np.where(Temp_history[timestep, 2:] < 2000)[0]
    # If there are any such indices, store the first one adjusted for the ignored positions
        if radial_indices.size > 0:
            first_indices[timestep] = radial_indices[0] + 2  # Adjust index by 2 to account for ignored positions
    # Since we're interested in timesteps starting from the third, the rest can be ignored for plotting
    # However, let's keep the full array for now to see the full context
    r_f_i = r[first_indices]
    y_values = r_f_i/1e13
   # print(y_values)
    x_values = np.array(t_history)/31536000

    if tag == 1:
        a1 = ax.plot(x_values, y_values, linewidth=3.5,alpha=1)#, color = 'red') 
    elif tag == 2:
        a1 = ax.plot(x_values, y_values, linewidth=3,alpha=0.9)#, color = 'red') 
    elif tag == 3:
        a1 = ax.plot(x_values, y_values, linewidth=2.5,alpha=0.8)#, color = 'red') 
    elif tag == 4:
        a1 = ax.plot(x_values, y_values, linewidth=2,alpha=0.7)#, color = 'red') 
    elif tag == 5:
        a1 = ax.plot(x_values, y_values, linewidth=3,alpha=0.9,linestyle ='dashed')#, color = 'red') 
    return a1

plt.rcParams.update({'font.size': 32})

fig, axs = plt.subplots(3, 1, figsize=(20, 16), gridspec_kw={'hspace': 0})

ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]

#Axis 1
ax1.set_xlabel(r'$t$ ($10^5$ years)')
ax1.set_ylabel(r'$\dot{M}_{in}$ (g s⁻¹)')
ax1.set_yscale('log')
#ax1.set_ylim(1e12,2e16)
ax1.set_xlim(0,1)

#Axis 2
ax2.set_xlabel(r'$t$ ($10^5$ years)')
ax2.set_ylabel(r'$L_{tot}$ (erg s$^{-1}$)')
ax2.set_yscale('log')
#ax2.set_ylim(1e36,1e38)
ax2.set_xlim(0,1)

#Axis 2
ax3.set_xlabel(r'$t$ ($10^5$ years)')
ax3.set_ylabel(r'$R_{out}$ (10$^{13}$cm)')
#ax3.set_yscale('log')
#ax3.set_ylim(0.1,0.7)
ax3.set_xlim(0,1)

#Hide tick labels of all x-axis except bottom row
ax1.set_xticklabels([])
ax2.set_xticklabels([])

#Set up major and minor inward ticks for all axes
setup_ticks(ax1)
setup_ticks(ax2)
setup_ticks(ax3)

smbh_index_min = 120
smbh_index_max = 260
s_smbh = Sigma_history_smbh_1[smbh_index_min:smbh_index_max]
h_smbh = H_history_smbh_1[smbh_index_min:smbh_index_max]
a_smbh = alpha_history_smbh_1[smbh_index_min:smbh_index_max]
T_smbh = Temp_history_smbh_1[smbh_index_min:smbh_index_max]
t_smbh = t_history_smbh_1[smbh_index_min:smbh_index_max]
t_smbh = (t_smbh - t_smbh[0])/1e5
L_smbh = L_array_smbh_1[smbh_index_min:smbh_index_max]


# Call the plotting function
plot_mass_transfer_smbh(ax1,s_smbh, h_smbh, a_smbh, t_smbh,2,r_smbh_1,M_smbh,1,0)
plot_Lum_smbh(ax2, L_smbh, t_smbh,1)
plot_radius_smbh(ax3, T_smbh, t_smbh,r_smbh,1)
#ax1.legend(frameon=False, loc='upper left', fontsize ='29')
```


```python
#R of smbh_2
# Disk parameters
R_1_smbh_2 = 5e12  # Inner radius of the disk (cm)
R_K_smbh_2 = 1e15  # Radius where initial mass is added (cm)
R_N_smbh_2 = 2e15  # Outer radius of the disk (cm)
X_1_smbh_2 = X_func(R_1_smbh_2)
X_K_smbh_2 = X_func(R_K_smbh_2)
X_N_smbh_2 = X_func(R_N_smbh_2)
N_smbh_2 = 100  # Number of grid points for the simulation
N_n_smbh_2 = 3 # Extra grid points after bath outer radius
# Define the radial grid
X_smbh_2 = np.linspace(X_1_smbh_2, X_N_smbh_2, N_smbh_2)
ΔX_smbh_2 = np.diff(X_smbh_2)[0]  # Assuming uniform grid spacing
X_smbh_2 = np.linspace(X_1_smbh_2,X_N_smbh_2+N_n_smbh_2*ΔX_smbh_2,N_smbh_2+N_n_smbh_2)
r_smbh_2 = R_func(X_smbh_2)
Δr_smbh_2 = np.diff(r_smbh_2)  # Assuming uniform grid spacing
Δr_smbh_2 = np.insert(Δr_smbh_2, 0, 0)
ΔX_smbh_2 = np.diff(X_smbh_2)[0]  # Assuming uniform grid spacing
X_N_smbh_2 = X_N_smbh_2+N_n_smbh_2*ΔX_smbh_2
N_smbh_2 = N_smbh_2+N_n_smbh_2 #redefine N to match new X array
#R of smbh_3
# Disk parameters
R_1_smbh_3 = 6e12  # Inner radius of the disk (cm)
R_K_smbh_3 = 1e15  # Radius where initial mass is added (cm)
R_N_smbh_3 = 2e15  # Outer radius of the disk (cm)
X_1_smbh_3 = X_func(R_1_smbh_3)
X_K_smbh_3 = X_func(R_K_smbh_3)
X_N_smbh_3 = X_func(R_N_smbh_3)
N_smbh_3 = 100  # Number of grid points for the simulation
N_n_smbh_3 = 3 # Extra grid points after bath outer radius
# Define the radial grid
X_smbh_3 = np.linspace(X_1_smbh_3, X_N_smbh_3, N_smbh_3)
ΔX_smbh_3 = np.diff(X_smbh_3)[0]  # Assuming uniform grid spacing
X_smbh_3 = np.linspace(X_1_smbh_3,X_N_smbh_3+N_n_smbh_3*ΔX_smbh_3,N_smbh_3+N_n_smbh_3)
r_smbh_3 = R_func(X_smbh_3)
Δr_smbh_3 = np.diff(r_smbh_3)  # Assuming uniform grid spacing
Δr_smbh_3 = np.insert(Δr_smbh_3, 0, 0)
ΔX_smbh_3 = np.diff(X_smbh_3)[0]  # Assuming uniform grid spacing
X_N_smbh_3 = X_N_smbh_3+N_n_smbh_3*ΔX_smbh_3
N_smbh_3 = N_smbh_3+N_n_smbh_3 #redefine N to match new X array
#R of smbh_4
# Disk parameters
R_1_smbh_4 = 7e12  # Inner radius of the disk (cm)
R_K_smbh_4 = 1e15  # Radius where initial mass is added (cm)
R_N_smbh_4 = 2e15  # Outer radius of the disk (cm)
X_1_smbh_4 = X_func(R_1_smbh_4)
X_K_smbh_4 = X_func(R_K_smbh_4)
X_N_smbh_4 = X_func(R_N_smbh_4)
N_smbh_4 = 100  # Number of grid points for the simulation
N_n_smbh_4 = 3 # Extra grid points after bath outer radius
# Define the radial grid
X_smbh_4 = np.linspace(X_1_smbh_4, X_N_smbh_4, N_smbh_4)
ΔX_smbh_4 = np.diff(X_smbh_4)[0]  # Assuming uniform grid spacing
X_smbh_4 = np.linspace(X_1_smbh_4,X_N_smbh_4+N_n_smbh_4*ΔX_smbh_4,N_smbh_4+N_n_smbh_4)
r_smbh_4 = R_func(X_smbh_4)
Δr_smbh_4 = np.diff(r_smbh_4)  # Assuming uniform grid spacing
Δr_smbh_4 = np.insert(Δr_smbh_4, 0, 0)
ΔX_smbh_4 = np.diff(X_smbh_4)[0]  # Assuming uniform grid spacing
X_N_smbh_4 = X_N_smbh_4+N_n_smbh_4*ΔX_smbh_4
N_smbh_4 = N_smbh_4+N_n_smbh_4 #redefine N to match new X array
#R of smbh_a2
# Disk parameters
R_1_smbh_a2 = 4e12  # Inner radius of the disk (cm)
R_K_smbh_a2 = 1e15  # Radius where initial mass is added (cm)
R_N_smbh_a2 = 2e15  # Outer radius of the disk (cm)
X_1_smbh_a2 = X_func(R_1_smbh_a2)
X_K_smbh_a2 = X_func(R_K_smbh_a2)
X_N_smbh_a2 = X_func(R_N_smbh_a2)
N_smbh_a2 = 100  # Number of grid points for the simulation
N_n_smbh_a2 = 3 # Extra grid points after bath outer radius
# Define the radial grid
X_smbh_a2 = np.linspace(X_1_smbh_a2, X_N_smbh_a2, N_smbh_a2)
ΔX_smbh_a2 = np.diff(X_smbh_a2)[0]  # Assuming uniform grid spacing
X_smbh_a2 = np.linspace(X_1_smbh_a2,X_N_smbh_a2+N_n_smbh_a2*ΔX_smbh_a2,N_smbh_a2+N_n_smbh_a2)
r_smbh_a2 = R_func(X_smbh_a2)
Δr_smbh_a2 = np.diff(r_smbh_a2)  # Assuming uniform grid spacing
Δr_smbh_a2 = np.insert(Δr_smbh_a2, 0, 0)
ΔX_smbh_a2 = np.diff(X_smbh_a2)[0]  # Assuming uniform grid spacing
X_N_smbh_a2 = X_N_smbh_a2+N_n_smbh_a2*ΔX_smbh_a2
N_smbh_a2 = N_smbh_a2+N_n_smbh_a2 #redefine N to match new X array

#Supermassive Black Hole
df_0 = pd.read_csv('Sigma_history_bath_array_new_Fcs_final_newopac_smbh_2.csv')
Sigma_history_smbh_2 = df_0.to_numpy().tolist()
df_1 = pd.read_csv('Temp_history_bath_array_new_Fcs_final_newopac_smbh_2.csv')
Temp_history_smbh_2 = df_1.to_numpy().tolist()
df_2 = pd.read_csv('H_history_bath_array_new_Fcs_final_newopac_smbh_2.csv')
H_history_smbh_2 = df_2.to_numpy().tolist()
df_3 = pd.read_csv('alpha_history_bath_array_new_Fcs_final_newopac_smbh_2.csv')
alpha_history_smbh_2 = df_3.to_numpy().tolist()
df_4 = pd.read_csv('t_history_bath_array_new_Fcs_final_newopac_smbh_2.csv')
t_history_smbh_2 = df_4.to_numpy().flatten().tolist()
df_5 = pd.read_csv('Sigma_transfer_history_bath_array_new_Fcs_final_newopac_smbh_2.csv')
Sigma_transfer_history_smbh_2 = df_5.to_numpy().flatten().tolist()

#smbh_2
H_history_smbh_2 = np.array(H_history_smbh_2)
Sigma_history_smbh_2 = np.array(Sigma_history_smbh_2)
alpha_history_smbh_2 = np.array(alpha_history_smbh_2)
Temp_history_smbh_2 = np.array(Temp_history_smbh_2)
t_history_smbh_2 = np.array(t_history_smbh_2)

#smbh_2
L_array_smbh_2 = np.empty(len(t_history_smbh_2))
L_rad_array_smbh_2 = []
for i in range(len(L_array_smbh_2)):
    L_array_smbh_2[i] = L_rad(Sigma_history_smbh_2, H_history_smbh_2, alpha_history_smbh_2, r_smbh_2, i,M_smbh)
for i in range(len(L_array_smbh_2)):
    L_rad_array_smbh_2.append(L_rad_arr(Sigma_history_smbh_2, H_history_smbh_2, alpha_history_smbh_2, r_smbh_2, i,M_smbh))
T_eff_array_smbh_2 = []
for i in range(len(Sigma_history_smbh_2)):
    T_eff_array_smbh_2.append(T_eff(Sigma_history_smbh_2, H_history_smbh_2, alpha_history_smbh_2, r_smbh_2, i,M_smbh))
T_eff_array_smbh_2 = np.vstack(T_eff_array_smbh_2)
L_rad_array_smbh_2 = np.vstack(L_rad_array_smbh_2)

#Supermassive Black Hole
df_0 = pd.read_csv('Sigma_history_bath_array_new_Fcs_final_newopac_smbh_3.csv')
Sigma_history_smbh_3 = df_0.to_numpy().tolist()
df_1 = pd.read_csv('Temp_history_bath_array_new_Fcs_final_newopac_smbh_3.csv')
Temp_history_smbh_3 = df_1.to_numpy().tolist()
df_2 = pd.read_csv('H_history_bath_array_new_Fcs_final_newopac_smbh_3.csv')
H_history_smbh_3 = df_2.to_numpy().tolist()
df_3 = pd.read_csv('alpha_history_bath_array_new_Fcs_final_newopac_smbh_3.csv')
alpha_history_smbh_3 = df_3.to_numpy().tolist()
df_4 = pd.read_csv('t_history_bath_array_new_Fcs_final_newopac_smbh_3.csv')
t_history_smbh_3 = df_4.to_numpy().flatten().tolist()
df_5 = pd.read_csv('Sigma_transfer_history_bath_array_new_Fcs_final_newopac_smbh_3.csv')
Sigma_transfer_history_smbh_3 = df_5.to_numpy().flatten().tolist()

#smbh_3
H_history_smbh_3 = np.array(H_history_smbh_3)
Sigma_history_smbh_3 = np.array(Sigma_history_smbh_3)
alpha_history_smbh_3 = np.array(alpha_history_smbh_3)
Temp_history_smbh_3 = np.array(Temp_history_smbh_3)
t_history_smbh_3 = np.array(t_history_smbh_3)

#smbh_3
L_array_smbh_3 = np.empty(len(t_history_smbh_3))
L_rad_array_smbh_3 = []
for i in range(len(L_array_smbh_3)):
    L_array_smbh_3[i] = L_rad(Sigma_history_smbh_3, H_history_smbh_3, alpha_history_smbh_3, r_smbh_3, i,M_smbh)
for i in range(len(L_array_smbh_3)):
    L_rad_array_smbh_3.append(L_rad_arr(Sigma_history_smbh_3, H_history_smbh_3, alpha_history_smbh_3, r_smbh_3, i,M_smbh))
T_eff_array_smbh_3 = []
for i in range(len(Sigma_history_smbh_3)):
    T_eff_array_smbh_3.append(T_eff(Sigma_history_smbh_3, H_history_smbh_3, alpha_history_smbh_3, r_smbh_3, i,M_smbh))
T_eff_array_smbh_3 = np.vstack(T_eff_array_smbh_3)
L_rad_array_smbh_3 = np.vstack(L_rad_array_smbh_3)

#Supermassive Black Hole
df_0 = pd.read_csv('Sigma_history_bath_array_new_Fcs_final_newopac_smbh_4.csv')
Sigma_history_smbh_4 = df_0.to_numpy().tolist()
df_1 = pd.read_csv('Temp_history_bath_array_new_Fcs_final_newopac_smbh_4.csv')
Temp_history_smbh_4 = df_1.to_numpy().tolist()
df_2 = pd.read_csv('H_history_bath_array_new_Fcs_final_newopac_smbh_4.csv')
H_history_smbh_4 = df_2.to_numpy().tolist()
df_3 = pd.read_csv('alpha_history_bath_array_new_Fcs_final_newopac_smbh_4.csv')
alpha_history_smbh_4 = df_3.to_numpy().tolist()
df_4 = pd.read_csv('t_history_bath_array_new_Fcs_final_newopac_smbh_4.csv')
t_history_smbh_4 = df_4.to_numpy().flatten().tolist()
df_5 = pd.read_csv('Sigma_transfer_history_bath_array_new_Fcs_final_newopac_smbh_4.csv')
Sigma_transfer_history_smbh_4 = df_5.to_numpy().flatten().tolist()

#smbh_4
H_history_smbh_4 = np.array(H_history_smbh_4)
Sigma_history_smbh_4 = np.array(Sigma_history_smbh_4)
alpha_history_smbh_4 = np.array(alpha_history_smbh_4)
Temp_history_smbh_4 = np.array(Temp_history_smbh_4)
t_history_smbh_4 = np.array(t_history_smbh_4)

#smbh_4
L_array_smbh_4 = np.empty(len(t_history_smbh_4))
L_rad_array_smbh_4 = []
for i in range(len(L_array_smbh_4)):
    L_array_smbh_4[i] = L_rad(Sigma_history_smbh_4, H_history_smbh_4, alpha_history_smbh_4, r_smbh_4, i,M_smbh)
for i in range(len(L_array_smbh_4)):
    L_rad_array_smbh_4.append(L_rad_arr(Sigma_history_smbh_4, H_history_smbh_4, alpha_history_smbh_4, r_smbh_4, i,M_smbh))
T_eff_array_smbh_4 = []
for i in range(len(Sigma_history_smbh_4)):
    T_eff_array_smbh_4.append(T_eff(Sigma_history_smbh_4, H_history_smbh_4, alpha_history_smbh_4, r_smbh_4, i,M_smbh))
T_eff_array_smbh_4 = np.vstack(T_eff_array_smbh_4)
L_rad_array_smbh_4 = np.vstack(L_rad_array_smbh_4)

#Supermassive Black Hole
df_0 = pd.read_csv('Sigma_history_bath_array_new_Fcs_final_newopac_smbh_a2.csv')
Sigma_history_smbh_a2 = df_0.to_numpy().tolist()
df_1 = pd.read_csv('Temp_history_bath_array_new_Fcs_final_newopac_smbh_a2.csv')
Temp_history_smbh_a2 = df_1.to_numpy().tolist()
df_2 = pd.read_csv('H_history_bath_array_new_Fcs_final_newopac_smbh_a2.csv')
H_history_smbh_a2 = df_2.to_numpy().tolist()
df_3 = pd.read_csv('alpha_history_bath_array_new_Fcs_final_newopac_smbh_a2.csv')
alpha_history_smbh_a2 = df_3.to_numpy().tolist()
df_4 = pd.read_csv('t_history_bath_array_new_Fcs_final_newopac_smbh_a2.csv')
t_history_smbh_a2 = df_4.to_numpy().flatten().tolist()
df_5 = pd.read_csv('Sigma_transfer_history_bath_array_new_Fcs_final_newopac_smbh_a2.csv')
Sigma_transfer_history_smbh_a2 = df_5.to_numpy().flatten().tolist()

#smbh_a2
H_history_smbh_a2 = np.array(H_history_smbh_a2)
Sigma_history_smbh_a2 = np.array(Sigma_history_smbh_a2)
alpha_history_smbh_a2 = np.array(alpha_history_smbh_a2)
Temp_history_smbh_a2 = np.array(Temp_history_smbh_a2)
t_history_smbh_a2 = np.array(t_history_smbh_a2)

#smbh_a2
L_array_smbh_a2 = np.empty(len(t_history_smbh_a2))
L_rad_array_smbh_a2 = []
for i in range(len(L_array_smbh_a2)):
    L_array_smbh_a2[i] = L_rad(Sigma_history_smbh_a2, H_history_smbh_a2, alpha_history_smbh_a2, r_smbh_1, i,M_smbh)
for i in range(len(L_array_smbh_a2)):
    L_rad_array_smbh_a2.append(L_rad_arr(Sigma_history_smbh_a2, H_history_smbh_a2, alpha_history_smbh_a2, r_smbh_1, i,M_smbh))
T_eff_array_smbh_a2 = []
for i in range(len(Sigma_history_smbh_a2)):
    T_eff_array_smbh_a2.append(T_eff(Sigma_history_smbh_a2, H_history_smbh_a2, alpha_history_smbh_a2, r_smbh_1, i,M_smbh))
T_eff_array_smbh_a2 = np.vstack(T_eff_array_smbh_a2)
L_rad_array_smbh_a2 = np.vstack(L_rad_array_smbh_a2)
```


```python
plt.rcParams.update({'font.size': 32})

fig, axs = plt.subplots(3, 1, figsize=(20, 16), gridspec_kw={'hspace': 0})

ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]

#Axis 1
ax1.set_xlabel(r'$t$ ($10^5$ years)')
ax1.set_ylabel(r'$\dot{M}_{in}$ (g s⁻¹)')
ax1.set_yscale('log')
#ax1.set_ylim(1e12,2e16)
ax1.set_xlim(0,1)

#Axis 2
ax2.set_xlabel(r'$t$ ($10^5$ years)')
ax2.set_ylabel(r'$L_{tot}$ (erg s$^{-1}$)')
ax2.set_yscale('log')
#ax2.set_ylim(1e36,1e38)
ax2.set_xlim(0,1)

#Axis 3
ax3.set_xlabel(r'$t$ ($10^5$ years)')
ax3.set_ylabel(r'$R_{out}$ (10$^{13}$cm)')
#ax3.set_yscale('log')
#ax3.set_ylim(0.1,0.7)
ax3.set_xlim(0,1)

#Hide tick labels of all x-axis except bottom row
ax1.set_xticklabels([])
ax2.set_xticklabels([])

#Set up major and minor inward ticks for all axes
setup_ticks(ax1)
setup_ticks(ax2)
setup_ticks(ax3)

smbh_index_min = 120
smbh_index_max = 310
s_smbh_1 = Sigma_history_smbh_1[smbh_index_min:smbh_index_max]
h_smbh_1 = H_history_smbh_1[smbh_index_min:smbh_index_max]
a_smbh_1 = alpha_history_smbh_1[smbh_index_min:smbh_index_max]
T_smbh_1 = Temp_history_smbh_1[smbh_index_min:smbh_index_max]
t_smbh_1 = t_history_smbh_1[smbh_index_min:smbh_index_max]
t_smbh_1 = (t_smbh_1 - t_smbh_1[0])/1e5
L_smbh_1 = L_array_smbh_1[smbh_index_min:smbh_index_max]
smbh_2_index_min = 120
smbh_2_index_max = 250
s_smbh_2 = Sigma_history_smbh_2[smbh_2_index_min:smbh_2_index_max]
h_smbh_2 = H_history_smbh_2[smbh_2_index_min:smbh_2_index_max]
a_smbh_2 = alpha_history_smbh_2[smbh_2_index_min:smbh_2_index_max]
T_smbh_2 = Temp_history_smbh_2[smbh_2_index_min:smbh_2_index_max]
t_smbh_2 = t_history_smbh_2[smbh_2_index_min:smbh_2_index_max]
t_smbh_2 = (t_smbh_2 - t_smbh_2[0])/1e5
L_smbh_2 = L_array_smbh_2[smbh_2_index_min:smbh_2_index_max]
smbh_3_index_min = 150
smbh_3_index_max = 250
s_smbh_3 = Sigma_history_smbh_3[smbh_3_index_min:smbh_3_index_max]
h_smbh_3 = H_history_smbh_3[smbh_3_index_min:smbh_3_index_max]
a_smbh_3 = alpha_history_smbh_3[smbh_3_index_min:smbh_3_index_max]
T_smbh_3 = Temp_history_smbh_3[smbh_3_index_min:smbh_3_index_max]
t_smbh_3 = t_history_smbh_3[smbh_3_index_min:smbh_3_index_max]
t_smbh_3 = (t_smbh_3 - t_smbh_3[0])/1e5
L_smbh_3 = L_array_smbh_3[smbh_3_index_min:smbh_3_index_max]
smbh_4_index_min = 150
smbh_4_index_max = 250
s_smbh_4 = Sigma_history_smbh_4[smbh_4_index_min:smbh_4_index_max]
h_smbh_4 = H_history_smbh_4[smbh_4_index_min:smbh_4_index_max]
a_smbh_4 = alpha_history_smbh_4[smbh_4_index_min:smbh_4_index_max]
T_smbh_4 = Temp_history_smbh_4[smbh_4_index_min:smbh_4_index_max]
t_smbh_4 = t_history_smbh_4[smbh_4_index_min:smbh_4_index_max]
t_smbh_4 = (t_smbh_4 - t_smbh_4[0])/1e5
L_smbh_4 = L_array_smbh_4[smbh_4_index_min:smbh_4_index_max]
smbh_a2_index_min = 140
smbh_a2_index_max = 200
s_smbh_a2 = Sigma_history_smbh_a2[smbh_a2_index_min:smbh_a2_index_max]
h_smbh_a2 = H_history_smbh_a2[smbh_a2_index_min:smbh_a2_index_max]
a_smbh_a2 = alpha_history_smbh_a2[smbh_a2_index_min:smbh_a2_index_max]
T_smbh_a2 = Temp_history_smbh_a2[smbh_a2_index_min:smbh_a2_index_max]
t_smbh_a2 = t_history_smbh_a2[smbh_a2_index_min:smbh_a2_index_max]
t_smbh_a2 = (t_smbh_a2 - t_smbh_a2[0])/1e5
L_smbh_a2 = L_array_smbh_a2[smbh_a2_index_min:smbh_a2_index_max]
# Call the plotting function
plot_mass_transfer_smbh(ax1,s_smbh_1, h_smbh_1, a_smbh_1, t_smbh_1,2,r_smbh_1,M_smbh,1,0)
plot_Lum_smbh(ax2, L_smbh_1, t_smbh_1,1)
plot_radius_smbh(ax3, T_smbh_1, t_smbh_1,r_smbh_1,1)
plot_mass_transfer_smbh(ax1,s_smbh_2, h_smbh_2, a_smbh_2, t_smbh_2,2,r_smbh_2,M_smbh,2,0)
plot_Lum_smbh(ax2, L_smbh_2, t_smbh_2,2)
plot_radius_smbh(ax3, T_smbh_2, t_smbh_2,r_smbh_2,2)
plot_mass_transfer_smbh(ax1,s_smbh_3, h_smbh_3, a_smbh_3, t_smbh_3,2,r_smbh_3,M_smbh,3,0)
plot_Lum_smbh(ax2, L_smbh_3, t_smbh_3,3)
plot_radius_smbh(ax3, T_smbh_3, t_smbh_3,r_smbh_3,3)
plot_mass_transfer_smbh(ax1,s_smbh_4, h_smbh_4, a_smbh_4, t_smbh_4,2,r_smbh_4,M_smbh,4,0)
plot_Lum_smbh(ax2, L_smbh_4, t_smbh_4,4)
plot_radius_smbh(ax3, T_smbh_4, t_smbh_4,r_smbh_4,4)
#plot_mass_transfer_smbh(ax1,s_smbh_a2, h_smbh_a2, a_smbh_a2, t_smbh_a2,2,r_smbh_a2,M_smbh)
#plot_Lum_smbh(ax2, L_smbh_a2, t_smbh_a2)
#plot_radius_smbh(ax3, T_smbh_a2, t_smbh_a2,r_smbh_a2)
ax1.legend(frameon=False, loc='upper left', fontsize ='20',title=r'$R_{in}$ ($10^{12}$cm)')
```


```python
plt.rcParams.update({'font.size': 32})

fig, axs = plt.subplots(3, 1, figsize=(20, 16), gridspec_kw={'hspace': 0})

ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]

#Axis 1
ax1.set_xlabel(r'$t$ ($10^5$ years)')
ax1.set_ylabel(r'$\dot{M}_{in}$ (g s⁻¹)')
ax1.set_yscale('log')
#ax1.set_ylim(1e12,2e16)
ax1.set_xlim(0,1)

#Axis 2
ax2.set_xlabel(r'$t$ ($10^5$ years)')
ax2.set_ylabel(r'$L_{tot}$ (erg s$^{-1}$)')
ax2.set_yscale('log')
#ax2.set_ylim(1e36,1e38)
ax2.set_xlim(0,1)

#Axis 3
ax3.set_xlabel(r'$t$ ($10^5$ years)')
ax3.set_ylabel(r'$R_{out}$ (10$^{13}$cm)')
#ax3.set_yscale('log')
#ax3.set_ylim(0.1,0.7)
ax3.set_xlim(0,1)

#Hide tick labels of all x-axis except bottom row
ax1.set_xticklabels([])
ax2.set_xticklabels([])

#Set up major and minor inward ticks for all axes
setup_ticks(ax1)
setup_ticks(ax2)
setup_ticks(ax3)

smbh_index_min = 120
smbh_index_max = 310
s_smbh_1 = Sigma_history_smbh_1[smbh_index_min:smbh_index_max]
h_smbh_1 = H_history_smbh_1[smbh_index_min:smbh_index_max]
a_smbh_1 = alpha_history_smbh_1[smbh_index_min:smbh_index_max]
T_smbh_1 = Temp_history_smbh_1[smbh_index_min:smbh_index_max]
t_smbh_1 = t_history_smbh_1[smbh_index_min:smbh_index_max]
t_smbh_1 = (t_smbh_1 - t_smbh_1[0])/1e5
L_smbh_1 = L_array_smbh_1[smbh_index_min:smbh_index_max]
smbh_a2_index_min = 120
smbh_a2_index_max = 310
s_smbh_a2 = Sigma_history_smbh_a2[smbh_a2_index_min:smbh_a2_index_max]
h_smbh_a2 = H_history_smbh_a2[smbh_a2_index_min:smbh_a2_index_max]
a_smbh_a2 = alpha_history_smbh_a2[smbh_a2_index_min:smbh_a2_index_max]
T_smbh_a2 = Temp_history_smbh_a2[smbh_a2_index_min:smbh_a2_index_max]
t_smbh_a2 = t_history_smbh_a2[smbh_a2_index_min:smbh_a2_index_max]
t_smbh_a2 = (t_smbh_a2 - t_smbh_a2[0])/1e5
L_smbh_a2 = L_array_smbh_a2[smbh_a2_index_min:smbh_a2_index_max]

# Call the plotting function
plot_mass_transfer_smbh(ax1,s_smbh_1, h_smbh_1, a_smbh_1, t_smbh_1,2,r_smbh_1,M_smbh,1,1)
plot_Lum_smbh(ax2, L_smbh_1, t_smbh_1,1)
plot_radius_smbh(ax3, T_smbh_1, t_smbh_1,r_smbh_1,1)
plot_mass_transfer_smbh(ax1,s_smbh_a2, h_smbh_a2, a_smbh_a2, t_smbh_a2,2,r_smbh_a2,M_smbh,5,1)
plot_Lum_smbh(ax2, L_smbh_a2, t_smbh_a2,5)
plot_radius_smbh(ax3, T_smbh_a2, t_smbh_a2,r_smbh_a2,5)

ax1.legend(frameon=False, loc='upper left', fontsize ='20',title=r'$\alpha_{cold}$')
```


```python
S_smbh1_i1 = Sigma_history_smbh_1[100:800,2]
S_smbh2_i1 = Sigma_history_smbh_2[100:800,2]
S_smbh3_i1 = Sigma_history_smbh_3[100:800,2]
S_smbh4_i1 = Sigma_history_smbh_4[100:800,2]
S_smbh5_i1 = Sigma_history_smbh_a2[100:800,2]

smbh1_peaks, _ = find_peaks(S_smbh1_i1, height=100, distance=30)
smbh1_peaks_times = t_history_smbh_1[smbh1_peaks]/(31536000*1e5)
smbh2_peaks, _ = find_peaks(S_smbh2_i1, height=100, distance=30)
smbh2_peaks_times = t_history_smbh_2[smbh2_peaks]/(31536000*1e5)
smbh3_peaks, _ = find_peaks(S_smbh3_i1, height=100, distance=30)
smbh3_peaks_times = t_history_smbh_3[smbh3_peaks]/(31536000*1e5)
smbh4_peaks, _ = find_peaks(S_smbh4_i1, height=100, distance=30)
smbh4_peaks_times = t_history_smbh_4[smbh4_peaks]/(31536000*1e5)
smbh5_peaks, _ = find_peaks(S_smbh5_i1, height=100, distance=30)
smbh5_peaks_times = t_history_smbh_a2[smbh5_peaks]/(31536000*1e5)


smbh1_peaks_times,smbh2_peaks_times,smbh3_peaks_times,smbh4_peaks_times,smbh5_peaks_times
```


```python
outburst_times_smbh = {
    r'$R_{in-1}$, $\alpha_{c-1}$': smbh1_peaks_times,
    r'$R_{in-2}$, $\alpha_{c-1}$': smbh2_peaks_times,
    r'$R_{in-3}$, $\alpha_{c-1}$': smbh3_peaks_times,
    r'$R_{in-4}$, $\alpha_{c-1}$': smbh4_peaks_times,
    r'$R_{in-1}$, $\alpha_{c-2}$': smbh5_peaks_times
}

averages_smbh = {}
standard_errors_smbh = {}

for model_smbh in outburst_times_smbh:
    periods_smbh = np.diff(outburst_times_smbh[model_smbh])  # Calculate periods between successive peaks
    average_period_smbh = np.mean(periods_smbh)  # Average period
    std_deviation_smbh = np.std(periods_smbh, ddof=1)  # Sample standard deviation
    std_error_smbh = std_deviation_smbh / np.sqrt(len(periods_smbh))  # Standard error

    averages_smbh[model_smbh] = average_period_smbh
    standard_errors_smbh[model_smbh] = std_error_smbh

averages_smbh, standard_errors_smbh
```


```python
# Extract the model names, average periods, and errors
models_smbh = list(averages_smbh.keys())
average_periods_smbh = [averages_smbh[model_smbh] for model_smbh in models_smbh]
errors_smbh = [standard_errors_smbh[model_smbh] for model_smbh in models_smbh]
plt.rcParams.update({'font.size': 40})
# Create the plot
plt.figure(figsize=(30, 15))
# Plot the error bars
plt.errorbar(models_smbh, np.array(average_periods_smbh) , yerr=np.array(errors_smbh), fmt='o', capsize=8, color='red', ecolor='blue', elinewidth=5, capthick=8,markersize=10)
plt.plot(models_smbh, np.array(average_periods_smbh), linewidth = 2, color = 'orange', linestyle ='dashed',alpha=0.5)
# Add the axis labels and title
plt.xlabel('SMBH Model', fontsize=42)
plt.ylabel(r'Outburst $\bar{T}_{SMBH}$ ($10^5$yrs)', fontsize=42)
#plt.title('Average Period Between Outbursts for Different Models', fontsize=40)
setup_ticks_l(plt)
# Add grid for better readability
#plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Display the plot
plt.show()
```


```python
import matplotlib.colors as mcolors
import matplotlib

matplotlib.rcParams['animation.embed_limit'] = 150.0

font = {'weight' : 'normal',
        'size'   : 24}

indices_irev = np.linspace(5000, int(len(L_rad_array_irev)/3), 180, endpoint=False).astype(int)

# Ensure all your luminosity values are > 0 before taking the log
L_rad_array_irev[L_rad_array_irev <= 0] = np.nan
log_L_rad_array_irev = np.log10(L_rad_array_irev)  # Applying log10

n_angles = 50
angles = np.linspace(0, 2 * np.pi, n_angles)
R_mesh, Theta = np.meshgrid(r[:30], angles)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(12, 8))
ax.xaxis.grid(False)
ax.set_yticks(r[::4])
ax.set_yticklabels([])  # Remove radial labels
ax.set_xticklabels([])  # Remove angular labels

# Use LogNorm for the normalization of the color scale
norm = mcolors.LogNorm(vmin=29, vmax=34)
c = ax.pcolormesh(Theta, R_mesh, np.zeros_like(R_mesh), shading='auto', cmap='hot', norm=norm)

# Create colorbar with logarithmic ticks
cb = plt.colorbar(c, label=r'log$_{10}(L)$')
cb_ticks = np.linspace(29, 34, 6)  # Modify this as needed for your actual data range
cb.set_ticks(cb_ticks)
cb_ticklabels = [f"{x:.0f}" for x in cb_ticks]  
cb.set_ticklabels(cb_ticklabels)


# ... Rest of your code for animation here ...

# Adjust update function to use the log luminosity data
def update_circle_lum_irev(frame):
    frame_index = indices_irev[frame]
    Temp_log = np.log10(L_rad_array_irev[frame_index, :][:30])  # Logarithmic scale
    Temp_log = np.tile(Temp_log, (n_angles, 1))  # Tiling the log luminosity data
    c.set_array(Temp_log.flatten())
    time_days = int((t_history_irev[frame_index]-t_history_irev[5000])/86400)
   # if time_days < 730:
   #     ax.set_title(f'Time (days) = {time_days}',fontsize = '28')
   # else:
   #     ax.set_title(f'Time (years) = {round(time_days/365,2)}',fontsize = '28')
    ax.set_title(r'Quiescence',fontsize = '28')
    return c,

def update_circle_lum_irev2(frame):
    frame_index = indices_irev[frame]
    Temp_log = np.log10(L_rad_array_irev[frame_index, :][:30])  # Logarithmic scale
    Temp_log = np.tile(Temp_log, (n_angles, 1))  # Tiling the log luminosity data
    c.set_array(Temp_log.flatten())
    time_days = int((t_history_irev[frame_index]-t_history_irev[5000])/86400)
   # if time_days < 730:
   #     ax.set_title(f'Time (days) = {time_days}',fontsize = '28')
   # else:
   #     ax.set_title(f'Time (years) = {round(time_days/365,2)}',fontsize = '28')
    ax.set_title(r'Outburst',fontsize = '28')
    return c,

def update_circle_lum_irev3(frame):
    frame_index = indices_irev[frame]
    Temp_log = np.log10(L_rad_array_irev[frame_index, :][:30])  # Logarithmic scale
    Temp_log = np.tile(Temp_log, (n_angles, 1))  # Tiling the log luminosity data
    c.set_array(Temp_log.flatten())
    time_days = int((t_history_irev[frame_index]-t_history_irev[5000])/86400)
   # if time_days < 730:
   #     ax.set_title(f'Time (days) = {time_days}',fontsize = '28')
   # else:
   #     ax.set_title(f'Time (years) = {round(time_days/365,2)}',fontsize = '28')
    ax.set_title(r'Outburst Fall',fontsize = '28')
    return c,

def update_circle_lum_irev_time(frame):
    frame_index = indices_irev[frame]
    Temp_log = np.log10(L_rad_array_irev[frame_index, :][:30])  # Logarithmic scale
    Temp_log = np.tile(Temp_log, (n_angles, 1))  # Tiling the log luminosity data
    c.set_array(Temp_log.flatten())
    time_days = int((t_history_irev[frame_index]-t_history_irev[5000])/86400)
    if time_days < 730:
        ax.set_title(f'Time (days) = {time_days}',fontsize = '28')
    else:
        ax.set_title(f'Time (years) = {round(time_days/365,2)}',fontsize = '28')
    return c,

ani_circle_lum_irev_t = FuncAnimation(fig, update_circle_lum_irev_time, frames=len(indices_irev), blit=True)
ani_circle_lum_irev = FuncAnimation(fig, update_circle_lum_irev, frames=len(indices_irev), blit=True)
ani_circle_lum_irev2 = FuncAnimation(fig, update_circle_lum_irev2, frames=len(indices_irev), blit=True)
ani_circle_lum_irev3 = FuncAnimation(fig, update_circle_lum_irev3, frames=len(indices_irev), blit=True)

# To save the animation, uncomment the next line (requires ffmpeg or pillow)
# ani.save('temperature_animation.mp4', writer='ffmpeg')

plt.show()
```


```python
from IPython.display import HTML
HTML(ani_circle_lum_irev.to_jshtml())
```


```python
from IPython.display import HTML
HTML(ani_circle_lum_irev2.to_jshtml())
```


```python
from IPython.display import HTML
HTML(ani_circle_lum_irev3.to_jshtml())
```


```python
from IPython.display import HTML
HTML(ani_circle_lum_irev_t.to_jshtml())
```


```python
ani_circle_lum_irev_t.save('temp_radial_grid_irev_bh.gif', fps=60, writer='pillow')
```


```python
from tabulate import tabulate

data = [['White Dwarf', r'M_{\odot}', r'5{\times}10^{16}',r'5{\times}10^{8}',r'8{\times}10^{10}','None','115.0 \pm 0.5 days'],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        []]

headers = ['Star Type', 'Mass', 'Accretion Rate', r'R_{in}', r'R_{out}', 'Effects', r'\bar{T}']

print(tabulate(data, headers=headers, tablefmt="grid"))
```


```python
import pandas as pd

# Your data
data = {"Type": ["White Dwarf", "Black Hole", "Black Hole"],
        "Mass": [r'M_{\odot}', r'9M_{\odot}', r'9M_{\odot}'],
        "Accretion Rate": [r'5{\times}10^{16}', r'1{\times}10^{17}', r'1{\times}10^{17}']}

df = pd.DataFrame(data)
print(df)
```


```python

```
