#%%
import numpy as np
import matplotlib.pyplot as plt

def plot_normalized_dynamic_structure_factor():
    """
    This function generates a plot of the dynamic structure factor S(q, ω) 
    as a function of energy transfer ω, showing a plasmon peak and a 
    particle excitation peak. The total area under the curve is normalized to 1.
    """
    # Define omega range and parameters for S(q, omega) (dynamic structure factor)
    omega = np.linspace(-10, 10, 500)  # Energy transfer range
    q = 1.0  # Momentum transfer (fixed for this plot)

    # Example dynamic structure factor as a sum of Gaussian peaks to simulate plasmon and particle excitations
    # Parameters for plasmon peak
    omega_plasmon = 2.0  # Plasmon frequency
    sigma_plasmon = 0.5  # Width of plasmon peak

    # Parameters for particle excitation broadening
    omega_particle = -3.0  # Center of particle excitation
    sigma_particle = 2.0   # Broader particle excitation peak

    # Gaussian peak for plasmon and broader peak for particle excitation
    S_plasmon = np.exp(-(omega - omega_plasmon)**2 / (2 * sigma_plasmon**2))
    S_particle = 0.8 * np.exp(-(omega - omega_particle)**2 / (2 * sigma_particle**2))

    # Total S(q, omega) distribution
    S_q_omega = S_plasmon + S_particle

    # Normalize S(q, omega) so that its integral is 1 (like a probability distribution)
    integral = np.trapz(S_q_omega, omega)  # Compute the integral over the omega range
    S_q_omega_normalized = S_q_omega / integral  # Normalize the function

    # Plot the normalized distribution
    plt.figure(figsize=(8, 6))
    plt.plot(omega, S_q_omega_normalized, label=r'Normalized $S(q, \omega)$', color='b')
    plt.title(r"Normalized $S(q, \omega)$ as a Function of $\omega$")
    plt.xlabel(r'Energy Transfer $\omega$')
    plt.ylabel(r'Normalized $S(q, \omega)$')
    plt.axvline(omega_plasmon, color='r', linestyle='--', label="Plasmon peak")
    plt.axvline(omega_particle, color='g', linestyle='--', label="Particle excitation")
    plt.legend()
    plt.show()

# Call the function to generate the normalized plot
plot_normalized_dynamic_structure_factor()
# %%
