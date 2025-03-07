import vip_ivp as vip


def mass_spring_damper_system(m=1, c=1, k=1, x0=0.2):
    acceleration = vip.loop_node()
    velocity = vip.integrate(acceleration, 0)
    displacement = vip.integrate(velocity, x0)
    acceleration.loop_into(-(c * velocity + k * displacement) / m)
    return displacement


t_simulation = 50  # s
time_step = 0.001  # s
vip.explore(mass_spring_damper_system, t_simulation, time_step=time_step, title="Mass-Spring-Damper mechanism")
