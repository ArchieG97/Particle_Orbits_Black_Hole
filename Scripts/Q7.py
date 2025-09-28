import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import quad
import os

# Output directory
output_dir = r"C:\CATAM_BlackHoleOrbits\Figures_BH"
os.makedirs(output_dir, exist_ok=True)

def f_turning(r, b):
    return 1 - (b**2 * (1 - 1/r) / r**2)

def deflection_angle(b):
    b_crit = 3 * np.sqrt(3) / 2
    if b <= b_crit:
        return np.nan  # captured, no deflection

    # Solve for r_min
    r_min = brentq(f_turning, 1.5001, b, args=(b,))

    # Integrand includes the factor b
    integrand = lambda r: b / (r**2 * np.sqrt(1 - (b**2 * (1 - 1/r) / r**2)))

    # Perform the integral
    I, _ = quad(integrand, r_min, np.inf, limit=200)

    # Return the net deflection
    return 2 * I - np.pi

# Impact parameters
b_values = np.linspace(5, 50, 100)
delta_phi_numeric = np.array([deflection_angle(b) for b in b_values])
delta_phi_approx = 2 / b_values  # Approximation

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(b_values, delta_phi_numeric, 'k-', label='Numerical')
plt.plot(b_values, delta_phi_approx, 'b--', label=r'Approx. $2/b$')
plt.xlabel('$b$')
plt.ylabel(r'$\Delta\phi(b)$')
plt.title('Photon Deflection Angle numerical comparison to analytic approximation')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Q7_deflection_angle_large_b.pdf'), dpi=300)
plt.show()
