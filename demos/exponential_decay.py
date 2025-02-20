import vip_ivp as vip

# Exponential decay : dN/dt = - λ * N
d_n = vip.loop_node()
n = vip.integrate(d_n, 1)
d_n.loop_into(-0.5 * n)

# Choose which variables to plot
n.to_plot("Quantity")
d_n.to_plot("Derivative")

# Solve the system. The plot will automatically show.
vip.solve(10, time_step=0.001)
