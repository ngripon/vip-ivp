import vip_ivp as vip

# Parameters
initial_height = 1  # m
GRAVITY = -9.81
k = 0.7  # Bouncing coefficient
v_min = 0.01  # Minimum velocity need to bounce

# Create the system
acceleration = vip.temporal(GRAVITY)
velocity = vip.integrate(acceleration, x0=0)
height = vip.integrate(velocity, x0=initial_height)

# Create the bouncing event
hit_ground = height.crosses(0, "falling")
velocity.reset_on(hit_ground, -0.8 * velocity)
vip.terminate_on(hit_ground & (abs(velocity) <= v_min))

# Add variables to plot
height.to_plot("Height (m)")

# Solve the system
vip.solve(20, time_step=0.005)
