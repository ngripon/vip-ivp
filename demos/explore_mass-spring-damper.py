import vip_ivp as vip


def mass_spring_damper_system(m=1, c=1, k=1, x0=0.2):
    acceleration = vip.loop_node()
    velocity = vip.integrate(acceleration, 0)
    displacement = vip.integrate(velocity, x0)
    acceleration.loop_into(-(c * velocity + k * displacement) / m)
    return velocity, displacement


# Solve the system
t_simulation = 10  # s
vip.explore(mass_spring_damper_system, t_simulation)
