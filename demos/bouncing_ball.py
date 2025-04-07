import vip_ivp as vip

# Parameters
initial_height = 1  # m
GRAVITY = -9.81
k = 0.7  # Bouncing coefficient
v_min = 0.01  # Minimum velocity need to bounce

# Create the system
acceleration = vip.create_source(GRAVITY)
velocity = vip.integrate(acceleration, 0)
height = vip.integrate(velocity, initial_height)

# Create the bouncing event
bounce = vip.where(abs(velocity) > v_min, velocity.set_value(-k * velocity), vip.terminate)
height.on_crossing(0, bounce, terminal=False, direction="falling")

# Add variables to plot
height.to_plot("Height (m)")

# Solve the system
vip.solve(20, time_step=0.001)
