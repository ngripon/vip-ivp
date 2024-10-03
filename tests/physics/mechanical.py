import operator
import random
import time
from typing import Union

import matplotlib.pyplot as plt

from main import Solver, TemporalVar
from tests.test_valid_systems import solver


class Mechanical1DBond:
    def __init__(self):
        self.speed = None
        self.force = None
        self.loop_node = None

    @property
    def effort(self):
        return self.force

    @effort.setter
    def effort(self, value: TemporalVar):
        self.force = value

    @property
    def flow(self):
        return self.speed

    @flow.setter
    def flow(self, value: TemporalVar):
        self.speed = value

    @property
    def power(self):
        return self.speed * self.force


class Mechanical1DBondFlow(Mechanical1DBond):
    def __init__(self, value: float, loop_node, solver):
        super().__init__()
        self.speed = value
        self.force = None
        self.loop_node = loop_node

    @classmethod
    def from_effort(cls, bond: "Mechanical1DBondEffort", solver):
        if isinstance(bond, Mechanical1DBondEffort):
            loop_node = solver.loop_node(bond.force)
        else:
            loop_node = solver.loop_node(bond)
        new_bond = cls(0, loop_node, solver)
        return new_bond, loop_node

    @Mechanical1DBond.effort.setter
    def effort(self, value):
        self.force = value
        self.loop_node.loop_into(self.force)


class Mechanical1DBondEffort(Mechanical1DBond):
    def __init__(self, value: float, loop_node, solver):
        super().__init__()
        self.speed = None
        self.force = value
        self.loop_node = loop_node

    @classmethod
    def from_flow(cls, bond: "Mechanical1DBondFlow", solver):
        if isinstance(bond, Mechanical1DBondFlow):
            loop_node = solver.loop_node(bond.speed)
        else:
            loop_node = solver.loop_node(bond)
        new_bond = cls(0, loop_node, solver)
        return new_bond, loop_node

    @Mechanical1DBond.flow.setter
    def flow(self, value):
        self.speed = value
        self.loop_node.loop_into(-self.flow)


def set_effort(bond: Union[Mechanical1DBond, float], effort: TemporalVar):
    if isinstance(bond, Mechanical1DBond):
        bond.effort = effort


def set_flow(bond: Union[Mechanical1DBond, float], flow: TemporalVar):
    if isinstance(bond, Mechanical1DBond):
        bond.flow = flow


def inertia(input_effort: Mechanical1DBondEffort, mass, gravity:bool, solver, speed0: float = 0) -> Mechanical1DBondFlow:
    output_flow, effort = Mechanical1DBondFlow.from_effort(input_effort, solver)
    acc = effort / mass
    if gravity:
        acc += 9.81
    speed = solver.integrate(acc, speed0)
    set_flow(output_flow, speed)
    set_flow(input_effort, speed)
    return output_flow


def spring(input_flow: Mechanical1DBondFlow, stiffness: float,
           solver: Solver, x0: float = 0) -> Mechanical1DBondEffort:
    output_effort, flow = Mechanical1DBondEffort.from_flow(input_flow, solver)
    x = solver.integrate(flow, x0)
    effort_value = stiffness * x
    set_effort(input_flow, -effort_value)
    set_effort(output_effort, effort_value)
    return output_effort


if __name__ == '__main__':
    solver = Solver()

    # Double spring system
    # mass_output = inertia(0, 1, 0, solver, 1)
    # spring_output = spring(mass_output, 1, solver, 0)
    # mass_2_output = inertia(spring_output, 1, 0, solver)
    # spring2_output=spring(mass_2_output, 1, solver)

    # # 100 spring system
    n_springs=100
    mass_list = []
    spring_list=[]
    current = 0
    for i in range(n_springs):
        mass_output = inertia(current, 1, 0, solver, 1 if i == 0 else 0)
        current = spring(mass_output, 1, solver)
        mass_list.append(mass_output)
        spring_list.append(current)

    solver.solve(500, time_step=0.01)

    for mass in mass_list[:1]:
        plt.plot(solver.t, mass.flow.values)
    # plt.plot(solver.t, mass_2_output.effort.values)
    # plt.plot(solver.t, spring_output.effort.values)
    # plt.plot(solver.t, mass_2_output.loop_node.values)
    plt.show()

    # n = 100
    # objects = []
    # for i in range(n):
    #     if i == 0 or i == n - 1:
    #         objects.append(0)
    #     else:
    #         objects.append(inertia(10, False, solver))
    # for i in range(n - 1):
    #     object1, object2 = objects[i], objects[i + 1]
    #     if i == 0:
    #         x0 = 1
    #     else:
    #         x0 = 0
    #     spring(object1, object2, 1, solver, x0)
    #
    # solver.solve(500, time_step=0.1)
    # for i in range(n):
    #     if i != 0 and i != n - 1:
    #         plt.plot(solver.t, objects[i].flow.values)
    # # plt.plot(solver.t, objects[100].flow.values)
    # plt.show()
