# Re-import necessary libraries since execution state was reset
import numpy as np
import matplotlib.pyplot as plt

# Redefine the constants
A = 1.0
B = 2.0
nu = 0.3
G = 1.5
rho = 1.2

# Define range for r (avoiding r=0 to prevent division by zero)
r = np.linspace(0.1, 10, 500)

# Compute theta_r based on the correct equation
theta_r_corrected = A + B / r**3 - ((3 + nu) / 20) * ((4 / 3) * G * rho ** 2* np.pi)* r**2

# Plot theta_r against r
plt.figure(figsize=(8, 5))
plt.plot(r, theta_r_corrected, label=r'$\theta_r = A + \frac{B}{r^3} - \left(\frac{3 + \nu}{20} \right) \left(\frac{4}{3} G \rho^2 \pi r^2 \right)$', color='r')
plt.xlabel(r'$r$')
plt.ylabel(r'$\theta_r$')
plt.title(r'Plot of $\theta_r$ against $r$')
plt.legend()
plt.grid(True)
plt.show()