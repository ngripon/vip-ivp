import matplotlib.pyplot as plt

from experimental.physics.bonds import *
from experimental.physics.mechanical import Inertia, Spring
from tests.test_known_solutions import ABSOLUTE_TOLERANCE


def test_spawn_two_connections():
    x0 = 1
    k1 = 1
    k2 = 5
    m = 1

    k_eq = k1 + k2

    # Mass damper mechanism
    mass1 = Inertia(m, False, port1=Mechanical1DEffort(0))
    spring1 = Spring(k1, x0=x0, port1=mass1.port2, port2=Mechanical1DFlow(0))
    spring2 = Spring(k2, x0=x0, port1=mass1.port2, port2=Mechanical1DFlow(0))
    vip.solve(10, time_step=0.001)

    vip.new_system()
    acceleration = vip.loop_node()
    velocity = vip.integrate(acceleration, 0)
    displacement = vip.integrate(velocity, x0)
    acceleration.loop_into(-(k_eq * displacement) / m)
    vip.solve(10, time_step=0.001)

    error_array = mass1.port2.flow.values - velocity.values

    assert all(error_array < ABSOLUTE_TOLERANCE)


def test_receive_two_connections():
    x0 = 1
    k1 = 1
    k2 = 5
    m = 1

    k_eq = k1 + k2

    # Mass damper mechanism
    spring1 = Spring(k1, x0=-x0, port1=Mechanical1DFlow(0))
    spring2 = Spring(k2, x0=-x0, port1=Mechanical1DFlow(0))
    mass = Inertia(m, False, port1=spring1.port2 + spring2.port2)
    vip.solve(10, time_step=0.001)

    vip.new_system()
    acceleration = vip.loop_node()
    velocity = vip.integrate(acceleration, 0)
    displacement = vip.integrate(velocity, x0)
    acceleration.loop_into(-(k_eq * displacement) / m)
    vip.solve(10, time_step=0.001)

    # plt.plot(velocity.t, velocity.values)
    # plt.plot(mass.port1.flow.t, mass.port1.flow.values)
    # plt.show()

    error_array = mass.port1.flow.values - velocity.values

    assert all(error_array < ABSOLUTE_TOLERANCE)
