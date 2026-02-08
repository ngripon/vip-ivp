import vip_ivp as vip

# Exponential decay : dN/dt = - λ * N
n = vip.state(1)
n.der=-0.5 * n

# Solve the system.
vip.solve(10)

# Plot the results
vip.plot(n, n.der)
