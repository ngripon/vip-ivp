from numbers import Number
from typing import Union

import matplotlib.pyplot as plt

import vip_ivp as vip


class Bond:
    def __init__(self, flow, effort):
        self._flow = flow
        self._effort = effort

    @property
    def effort(self):
        return self._effort

    @effort.setter
    def effort(self, value: vip.TemporalVar):
        self._effort = value

    @property
    def flow(self):
        return self._flow

    @flow.setter
    def flow(self, value: vip.TemporalVar):
        self._flow = value

    @property
    def power(self):
        return self.flow * self.effort


class BondFlow(Bond):
    def __init__(self, value: float = 0):
        super().__init__(value, vip.loop_node())

    @Bond.effort.setter
    def effort(self, value):
        self.effort.loop_into(value)


class BondEffort(Bond):
    def __init__(self, value: float = 0):
        super().__init__(vip.loop_node(), value)

    @Bond.flow.setter
    def flow(self, value):
        self.flow.loop_into(value)


class Inertia:
    def __init__(self, port1: BondEffort, mass: float, gravity: bool, speed0: float = 0):
        self.port2 = BondFlow()
        acc = self.port2.effort + port1.effort / mass
        if gravity:
            acc += 9.81
        speed = vip.integrate(acc, speed0)
        self.port2.flow = speed
        port1.flow = speed


class Spring:
    def __init__(self, port1: BondFlow, stiffness: float, x0: float = 0):
        self.port2 = BondEffort()
        x = vip.integrate(port1.flow - self.port2.flow, x0)
        effort_value = stiffness * x
        port1.effort = -effort_value
        self.port2.effort = effort_value


if __name__ == '__main__':
    # 100 spring system
    n_springs = 100
    mass_list = []
    spring_list = []
    current_effort = BondEffort(0)
    for i in range(n_springs):
        mass = Inertia(current_effort, 1, False, 1 if i == 0 else 0)
        spring = Spring(mass.port2, 1)
        current_effort = spring.port2

        mass_list.append(mass.port2)
        spring_list.append(spring.port2)

    vip.solve(500, time_step=0.01)
    # Plot

    for mass in mass_list[:1]:
        plt.plot(mass.flow.t, mass.flow)
    plt.show()
