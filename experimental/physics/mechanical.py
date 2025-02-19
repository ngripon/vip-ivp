from numbers import Number
from typing import Union

import matplotlib.pyplot as plt

import vip_ivp as vip


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
    def __init__(self, value: float, loop_node):
        super().__init__()
        self.speed = value
        self.force = None
        self.loop_node = loop_node

    @classmethod
    def from_effort(cls, bond: "Mechanical1DBondEffort"):
        if isinstance(bond, Mechanical1DBondEffort):
            loop_node = vip.loop_node(bond.force)
        elif isinstance(bond, Number):
            loop_node = vip.loop_node(bond)
        else:
            raise Exception(f"Incompatible type: {bond} of type {type(bond)}.")
        new_bond = cls(0, loop_node)
        return new_bond, loop_node

    @Mechanical1DBond.effort.setter
    def effort(self, value):
        self.force = value
        self.loop_node.loop_into(self.force)


class Mechanical1DBondEffort(Mechanical1DBond):
    def __init__(self, value: float, loop_node):
        super().__init__()
        self.speed = None
        self.force = value
        self.loop_node = loop_node

    @classmethod
    def from_flow(cls, bond: "Mechanical1DBondFlow"):
        if isinstance(bond, Mechanical1DBondFlow):
            loop_node = vip.loop_node(bond.speed)
        elif isinstance(bond, Number):
            loop_node = vip.loop_node(bond)
        else:
            raise Exception(f"Incompatible type: {bond} of type {type(bond)}.")
        new_bond = cls(0, loop_node)
        return new_bond, loop_node

    @Mechanical1DBond.flow.setter
    def flow(self, value):
        self.speed = value
        self.loop_node.loop_into(-self.flow)


def set_effort(bond: Union[Mechanical1DBond, float], effort: vip.TemporalVar):
    if isinstance(bond, Mechanical1DBond):
        bond.effort = effort


def set_flow(bond: Union[Mechanical1DBond, float], flow: vip.TemporalVar):
    if isinstance(bond, Mechanical1DBond):
        bond.flow = flow


class Inertia:
    def __init__(self, port1: Union[Mechanical1DBondEffort, float], mass: float, gravity: bool, speed0: float = 0):
        output_flow, effort = Mechanical1DBondFlow.from_effort(port1)
        acc = effort / mass
        if gravity:
            acc += 9.81
        speed = vip.integrate(acc, speed0)
        set_flow(output_flow, speed)
        set_flow(port1, speed)
        self.port2 = output_flow


class Spring:
    def __init__(self, port1: Union[Mechanical1DBondFlow, float], stiffness: float, x0: float = 0):
        output_effort, flow = Mechanical1DBondEffort.from_flow(port1)
        x = vip.integrate(flow, x0)
        effort_value = stiffness * x
        set_effort(port1, -effort_value)
        set_effort(output_effort, effort_value)
        self.port2 = output_effort


if __name__ == '__main__':
    # 100 spring system
    n_springs = 100
    mass_list = []
    spring_list = []
    current_effort = 0
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
