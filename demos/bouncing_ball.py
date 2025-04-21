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
bounce = vip.where(abs(velocity) > v_min, velocity.action_reset_to(-k * velocity), vip.action_terminate)
height.on_crossing(0, bounce, terminal=False, direction="falling")

# Add variables to plot
height.to_plot("Height (m)")

# Solve the system
vip.solve(20, time_step=0.001)
