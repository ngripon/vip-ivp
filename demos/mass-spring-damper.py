import vip_ivp as vip

# System parameters
m = 1.0  # Mass (kg)
k = 10.0  # Spring constant (N/m)
c = 0.5  # Damping coefficient (N·s/m)

x0 = 1.0  # Initial position (m)
v0 = 0.0  # Initial velocity (m/s)

# Build the system
a = vip.loop_node()  # Acceleration
v = vip.integrate(a, v0)  # Velocity
x = vip.integrate(v, x0)  # Displacement
a.loop_into(-(c * v + k * x) / m) # Set acceleration value

# Create a variable to count the number of oscillations
count = vip.create_source(0)
# Create event that triggers when x crosses 0 (from negative to positive)
x.on_crossing(0, count.action_set_to(count + 1), direction="rising")

# Choose results to plot
x.to_plot("Displacement (m)")
count.to_plot("Oscillations count")
# Solve the system
vip.solve(10, time_step=0.01)
