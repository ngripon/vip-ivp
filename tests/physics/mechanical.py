from typing import Union

import matplotlib.pyplot as plt

import main as vip


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
        else:
            loop_node = vip.loop_node(bond)
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
        else:
            loop_node = vip.loop_node(bond)
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


def inertia(input_effort: Mechanical1DBondEffort, mass, gravity: bool, speed0: float = 0) -> Mechanical1DBondFlow:
    output_flow, effort = Mechanical1DBondFlow.from_effort(input_effort)
    acc = effort / mass
    if gravity:
        acc += 9.81
    speed = vip.integrate(acc, speed0)
    set_flow(output_flow, speed)
    set_flow(input_effort, speed)
    return output_flow


def spring(input_flow: Mechanical1DBondFlow, stiffness: float, x0: float = 0) -> Mechanical1DBondEffort:
    output_effort, flow = Mechanical1DBondEffort.from_flow(input_flow)
    x = vip.integrate(flow, x0)
    effort_value = stiffness * x
    set_effort(input_flow, -effort_value)
    set_effort(output_effort, effort_value)
    return output_effort


if __name__ == '__main__':

    # Double spring system
    # mass_output = inertia(0, 1, 0, 1)
    # spring_output = spring(mass_output, 1, 0)
    # mass_2_output = inertia(spring_output, 1, 0)
    # spring2_output=spring(mass_2_output, 1)

    # # 100 spring system
    n_springs = 100
    mass_list = []
    spring_list = []
    current = 0
    for i in range(n_springs):
        mass_output = inertia(current, 1, False, 1 if i == 0 else 0)
        current = spring(mass_output, 1)
        mass_list.append(mass_output)
        spring_list.append(current)

    vip.solve(500, time_step=0.01)

    for mass in mass_list[:1]:
        plt.plot(mass.flow.t, mass.flow)
    plt.show()

