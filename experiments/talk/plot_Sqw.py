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
    plt.plot(omega, S_q_omega_normalized, label=r'Distribution $S(q, \omega)$', color='b')
    plt.xlabel(r'Energy Transfer $\omega$')
    plt.ylabel(r'Normalized $S(q, \omega)$')
    plt.axvline(omega_plasmon, color='r', linestyle='--', label="Plasmon peak")
    plt.axvline(omega_particle, color='g', linestyle='--', label="Particle excitation")
    plt.legend()
    plt.show()

# Call the function to generate the normalized plot
plot_normalized_dynamic_structure_factor()
# %%
import matplotlib.pyplot as plt

# Data provided by the user
lambda_values = [1.0e-12, 1.0e-11, 1.0e-10, 1.0e-9, 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 0.0001, 0.001, 0.01]
KLLS_values = [2008.0, 1082.0, 892.0, 570.0, 376.0, 242.0, 164.0, 132.0, 94.0, 60.0, 40.0]
PDCO_values = [3896.0, 4084.0, 7368.0, 11738.0, 7520.0, 6060.0, 3628.0, 2070.0, 1144.0, 646.0, 398.0]

# Plotting without the failed convergence points and adjusting the labels as requested
plt.figure(figsize=(10, 6))

# Plotting both solvers with distinct colors
plt.plot(lambda_values, KLLS_values, marker='o', color='blue', label='KLLS', linewidth=2)
plt.plot(lambda_values[3:], PDCO_values[3:], marker='o', color='green', label='PDCO', linewidth=2)

# Log scale for lambda and reversing the x-axis
plt.xscale('log')
plt.gca().invert_xaxis()
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# Labels and removing the title
plt.xlabel('Regularization (λ)', fontsize=14)
plt.ylabel('Matrix-Vector Products', fontsize=14)

# Adding legend
plt.legend()

# Show the plot with grid
plt.grid(False)
plt.show()
# %%
