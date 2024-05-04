import numpy as np
from scipy.special import softmax 
import inversionmethods as IM
import matplotlib.pyplot as plt
pi=np.pi
import numpy.linalg as la

import time


# ------------------------------------- #
# ---- Spectral Function Inversion ---- #
# ------------------------------------- #
kernel='ScalarParticle'
Nomega = 400; Nk = 250; Ntau = 50;
GeV_to_fminv = 1./5.068
fm_to_GeVinv = 5.068
omegas = np.linspace(0, 4, Nomega) 
taus = fm_to_GeVinv*0.085*np.array(range(Ntau)) #np.linspace(0,4, Ntau)

mass_rho=.77 # rho mass in GeV
mass_pi=.14 # pion mass in GeV
g2_rhopipi=5.45**2 # rho-pion-pion vertex/interaction strength (squared)
w0 = 1.3 # GeV
delta = 0.2 # GeV
alpha_s = 0.3 # strong force coupling parameter
# define rho particle decay rate
GG_rho = g2_rhopipi / (48*np.pi) * mass_rho * ( 1 - 4*mass_pi**2 / omegas**2 )**(3/2) * np.heaviside(omegas - 2*mass_pi , 0.5)
GG_rho [ np.isnan(GG_rho) ] =0.0
# define lorentzian peak around the pole
F2_rho = mass_rho**2 / g2_rhopipi
pole = F2_rho* GG_rho * mass_rho / (( omegas**2 - mass_rho**2 )**2 + GG_rho**2 * mass_rho**2 )
cut = 1/(8*np.pi)*(1 + alpha_s/np.pi) / (1 + np.exp( (w0 - omegas) / delta))
x0 = 2 / np.pi * (pole + cut)
x0 /= np.sum(x0)
noise_level=.0001
Nsamples=1000
gamma=.01

#  -------------------------------------------------------------------

# A -- Matrix
# b -- noiseless observation
# bn -- noisy observation
A, b, bn = IM.generate_data(taus, omegas, x0, noise_level, Nsamples, kernel, gamma)

# Save the data into a .npz file
np.savez('PhysicsData.npz', A=A, b=b, bn=bn, x0=x0)

# Running the above code, one has access to A, b in R^{50}, and the initial vector x0 in R^{400}



# Uncomment the code below for visuals

# plt.figure()
# plt.plot(b, color='black')
# plt.title(r"Our observation $b \in \mathbb{R}^{50}$")

# plt.show()

# plt.figure()
# plt.imshow(np.log(A))
# plt.title(r"The matrix A")

# plt.show()

# plt.plot(x0)
# plt.title(r"The true measure $x_0 \in \Delta_{400}$ we wish to recover ")

# plt.show()

