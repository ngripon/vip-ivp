import vip_ivp as vip
import matplotlib.pyplot as plt

# System parameters
m = 1.0  # Mass (kg)
k = 10.0  # Spring constant (N/m)
c = 0.5  # Damping coefficient (N·s/m)

x0 = 1.0  # Initial position (m)
v0 = 0.0  # Initial velocity (m/s)

# Build the system
x, v = vip.n_order_state(x0, v0)
v.der = -(c * v + k * x) / m  # Set acceleration value

# Solve the system
vip.solve(10)

# Plot the variable evolution
vip.plot(x, v)

# Create a phase space diagram
plt.plot(x.values, v.values)
plt.xlabel('Position x (m)')
plt.ylabel('Velocity v (m/s)')
plt.title('Phase Space Diagram')
plt.grid()
plt.show()

# Export the results to pandas
dataframe = vip.export_to_df()
print(dataframe)
