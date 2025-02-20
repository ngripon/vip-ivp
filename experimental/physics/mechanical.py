from numbers import Number
from typing import Union

import matplotlib.pyplot as plt

import vip_ivp as vip
from vip_ivp import LoopNode


class Mechanical1DBond:
    def __init__(self):
        self.speed = None
        self.force = None
        self.loop_node = None

    @property
    def effort(self):
        return self.force

    @effort.setter
    def effort(self, value: vip.TemporalVar):
        self.force = value

    @property
    def flow(self):
        return self.speed

    @flow.setter
    def flow(self, value: vip.TemporalVar):
        self.speed = value

    @property
    def power(self):
        return self.speed * self.force


class Mechanical1DBondFlow(Mechanical1DBond):
    def __init__(self, value: float = 0):
        super().__init__()
        self.speed = value
        self.force = vip.loop_node()

    @Mechanical1DBond.effort.setter
    def effort(self, value):
        self.force.loop_into(value)


class Mechanical1DBondEffort(Mechanical1DBond):
    def __init__(self, value: float = 0):
        super().__init__()
        self.speed = vip.loop_node()
        self.force = value

    @Mechanical1DBond.flow.setter
    def flow(self, value):
        self.speed.loop_into(value)


class Inertia:
    def __init__(self, port1: Mechanical1DBondEffort, mass: float, gravity: bool, speed0: float = 0):
        self.port2 = Mechanical1DBondFlow()
        acc = self.port2.effort + port1.effort / mass
        if gravity:
            acc += 9.81
        speed = vip.integrate(acc, speed0)
        self.port2.flow = speed
        port1.flow = speed


class Spring:
    def __init__(self, port1: Mechanical1DBondFlow, stiffness: float, x0: float = 0):
        self.port2 = Mechanical1DBondEffort()
        x = vip.integrate(port1.speed - self.port2.speed, x0)
        effort_value = stiffness * x
        port1.effort = -effort_value
        self.port2.effort = effort_value


if __name__ == '__main__':
    # 100 spring system
    n_springs = 100
    mass_list = []
    spring_list = []
    current_effort = Mechanical1DBondEffort(0)
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
