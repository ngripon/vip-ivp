import vip_ivp as vip

# Parameters
initial_height = 1  # m
GRAVITY = -9.81
k = 0.7  # Bouncing coefficient
v_min = 0.01  # Minimum velocity need to bounce

# Create the system
height, velocity = vip.n_order_state(initial_height, 0, derivative=GRAVITY)

# Create the bouncing event
hit_ground = height.crosses(0, "falling")
vip.when(hit_ground, velocity.reinit(-k * velocity))
vip.when(hit_ground & (abs(velocity) <= v_min), vip.terminate)

# Solve the system
vip.solve(20, step_eval=0.005)

# Post-processing
vip.plot(height)
