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
acceleration = vip.loop_node(2)
velocity = vip.integrate(acceleration, v0)
position = vip.integrate(velocity, x0)
v_norm = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
acceleration.loop_into([-mu * velocity[0] * v_norm,
                        GRAVITY - mu * velocity[1] * v_norm])
# Terminate on touching the ground
position[1].on_crossing(0, direction="falling", terminal=True)

vip.solve(10, time_step=0.01)

# Plot results
x, y = position.values
plt.plot(x, y)
plt.title("Projectile motion")
plt.xlabel("X (m)")
plt.ylabel("Height (m)")
plt.grid()
plt.show()
