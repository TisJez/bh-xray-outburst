```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
%matplotlib inline

#universal constants in SI units

c = 299792458
G = 6.6743e-11
sigma = 5.670374419e-8
h = 6.62607015e-34
k_B = 1.380649e-23
solarmass = 2e30
m_p = 1.67262192e-27

#constants converted to KATO units (CGS)

solarmass = solarmass*1000
c = c*100
m_p = m_p*1000
k_B = k_B*1e7
sigma = sigma*1000
G = G*1000
h = h*1e7
m_H = m_p


#M_dot mass accretion
M_dot_crit = 1.5e17 #m g s^-1

#schwarzschild radius #chris says its actually just G*M/(c**2)
def r_g(M):
    return 2*G*M/(c**2)
    
#variable constants in SI

#bhmass = 10*solarmass
#Rg = (bhmass*G)/(c**2)
#Rin = 6*Rg
#Rout = Rg*10**5
#masschange = 10**15


#KATO UNITS
# g, cm, K
```


```python
def r_hat(r, M):
    return r/r_g(M)
     
def m(M):
    return M/solarmass

def m_dot(M_dot):
    return M_dot/M_dot_crit
    
def f(r, M):
    a = np.sqrt(3*r_g(M)/r)
    b = 1-a
    return b


#test values
M = 3*solarmass
M_dot = M_dot_crit/100

#alpha = 0.1 #s-s dimensionless value
nu = 0.1 #efficiency
#m_dot16 = (4*np.pi*G*m(M)*m_p/(c*nu*sigma))/1e16 #eddington limit of accretion mass/(10**16 g/s)



#Opacity values

#kappa_es = .4 cm^2g^-1
#kappa_ff = kappa_0*rho*T^-3.5
#kappa_0 = 6.4x10^22
#optical_depth = tau_* = sqrt(kappa_es*kappa_ff)*rho*H

#radiation constant 'a' in textbooks
a_rad = 4*sigma/c
```


```python
X_kapp = .96
Z_kapp = 1-X_kapp
Z_star = 5e-10
#Z_star = .1

#opacity equations

kappa_Th = 0.2*(1+X_kapp)

def kappa_e(rho, T):
    a = 0.2*(1+X_kapp)*(1/(1+(2.7e11)*(rho/(T**2))))
    b = 1/(1+(T/(4.5e8))**0.86)
    c1 = a*b
    return c1

def kappa_K(rho, T):
    a = (4e25)*(1+X_kapp)*(Z_kapp+0.001)
    b = rho/(T**3.5)
    c1 = a*b
    return c1

def kappa_Hminus(rho, T):
    a = (1.1e-25)*(np.sqrt(Z_kapp*rho))
    b = T**7.7
    c1 = a*b
    return c1

kappa_M = 0.1*Z_kapp

def kappa_rad(rho, T):
    a = kappa_e(rho,T) + kappa_K(rho,T)
    b = 1/kappa_Hminus(rho,T) + 1/a
    c1 = 1/b
    d = kappa_M + c1
    return d

def kappa_cond(rho, T):
    a = 2.6e-7
    b = Z_star
    c0 = (T**2)/(rho**2)
    c1 = 1+(rho/2e6)**(2/3)
    d = a*b*c0*c1
    return d

def kappa_tot(rho, T):
    a0 = 1/kappa_rad(rho,T)
    a1 = 1/kappa_cond(rho,T)
    a = a0 + a1
    b = 1/a
    return b
```


```python
#kappa derivatives

def kappa_tot_drho(rho,T,drho):
    
    drho = 1e-5
    
    a = kappa_tot(rho+drho,T) - kappa_tot(rho,T)
    b = drho
    return a/b

def kappa_cond_drho(rho,T,drho):
    
    drho = 1e-5
    
    a = kappa_cond(rho+drho,T) - kappa_cond(rho,T)
    b = drho
    return a/b

def kappa_tot_dT(rho,T,dT):
    
    dT = 1e-5
    
    a = kappa_tot(rho,T+dT) - kappa_tot(rho,T)
    b = dT
    return a/b
```


```python
#old kappa values


def kappa_ff(rho,T):
    a = kappa_0*rho*T_c**-3.5
    return a

def kappa_old(rho,T):
    kappa_es = 0.4
    kappa_0 = 6.4e22
    a = kappa_es + kappa_0*rho*T**-3.5
    return a

def kappa_old_drho(rho,T,drho):
    
    drho = 1e-5
    
    a = kappa_old(rho+drho,T) - kappa_old(rho,T)
    b = drho
    return a/b
```
