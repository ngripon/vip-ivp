import matplotlib.pyplot as plt
import vip_ivp as vip
import numpy as np

# Parameters
GRAVITY = -9.81
v0 = 30
th0 = np.radians(45)
mu = 0.1  # Coefficient of air drag

# Compute initial conditions
v0 = [v0 * np.cos(th0), v0 * np.sin(th0)]
x0 = [0, 0]

# Create the system
velocity = [vip.state(x) for x in v0]
position = [vip.state(x, derivative=v) for x, v in zip(x0, velocity)]
v_norm = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
velocity[0].der = -mu * velocity[0] * v_norm
velocity[1].der = GRAVITY - mu * velocity[1] * v_norm

# Terminate on touching the ground
hit_ground = position[1].crosses(0, "falling")
vip.when(hit_ground, vip.terminate)

vip.solve(10)

# Plot results
x, y = [x.values for x in position]
plt.plot(x, y)
plt.title("Projectile motion")
plt.xlabel("X (m)")
plt.ylabel("Height (m)")
plt.grid()
plt.show()
