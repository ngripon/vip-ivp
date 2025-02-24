from numbers import Number
from typing import Union

import matplotlib.pyplot as plt

import vip_ivp as vip

from experimental.physics.bonds import Mechanical1DEffort, Mechanical1DFlow, Bond


class Inertia:
    def __init__(self, mass: float, gravity: bool, speed0: float = 0, port1: Mechanical1DEffort = None,
                 port2: Mechanical1DEffort = None):
        if port1 is not None:
            self.port1 = port1
        else:
            self.port1 = Mechanical1DFlow()
        if port2 is not None:
            self.port2 = port1
        else:
            self.port2 = Mechanical1DFlow()
        acc = self.port2.force + self.port1.force / mass
        if gravity:
            acc += 9.81
        speed = vip.integrate(acc, speed0)
        self.port2.speed = speed
        self.port1.speed = speed


class Spring:
    def __init__(self, port1: Mechanical1DFlow, stiffness: float, x0: float = 0):
        self.port2 = Mechanical1DEffort()
        x = vip.integrate(port1.flow - self.port2.flow, x0)
        effort_value = stiffness * x
        port1.effort = -effort_value
        self.port2.effort = effort_value


if __name__ == '__main__':
    # 100 spring system
    n_springs = 100
    mass_list = []
    spring_list = []
    current_effort = Mechanical1DEffort(0)
    for i in range(n_springs):
        mass = Inertia( 1, False, 1 if i == 0 else 0, port1=current_effort)
        spring = Spring(mass.port2, 1)
        current_effort = spring.port2

        mass_list.append(mass.port2)
        spring_list.append(spring.port2)

    vip.solve(500, time_step=0.01)
    # Plot

    for mass in mass_list[:1]:
        plt.plot(mass.flow.t, mass.flow)
    plt.show()
