import vip_ivp as vip

# System parameters
m = 300.0  # Mass (kg)
c = 1500  # Damping coefficient (N.s/m)
k = 25000  # Spring stiffness (N/m)
displacement_x0 = 0.2  # Initial value of displacement (m)

# Create simulation
# System equation is : m * acc + c * vel + k * disp = 0 <=> acc = - 1 / m * (c * vel + k * disp)
# We do not have access to velocity and displacement at this stage, so we create a loop node.
acceleration = vip.loop_node()
velocity = vip.integrate(acceleration, 0)
displacement = vip.integrate(velocity, displacement_x0)
# Now we can set the acceleration
acceleration.loop_into(-(c * velocity + k * displacement) / m)

# Choose results to plot
displacement.to_plot("Displacement (m)")
velocity.to_plot("Velocity (m/s)")

# Solve the system
t_simulation = 10  # s
time_step = 0.001
vip.solve(t_simulation, time_step=time_step)
