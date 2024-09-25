import random
from typing import Union

import matplotlib.pyplot as plt

from main import Solver, TemporalVar
from tests.test_valid_systems import solver


class Mechanical1DBond:
    def __init__(self, solver):
        self.speed = None
        self.force = solver.loop_node()

    @property
    def flow(self):
        return self.speed

    @flow.setter
    def flow(self, value):
        self.speed = value

    @property
    def effort(self):
        return self.force

    @effort.setter
    def effort(self, value):
        self.force.loop_into(value)


def inertia(mass, gravity, solver) -> Mechanical1DBond:
    bond = Mechanical1DBond(solver)
    acc = bond.force / mass
    if gravity:
        acc += 9.81
    bond.flow = solver.integrate(acc, 0)
    return bond


def spring(bond1: Union[Mechanical1DBond, float], bond2: Union[Mechanical1DBond, float], stiffness: float,
           solver: Solver,
           x0: float = 0) -> None:
    speed1, speed2 = _get_flow(bond1), _get_flow(bond2)
    x = solver.integrate(speed1 - speed2, x0)
    effort_value = stiffness * x
    _set_effort(bond2, effort_value)
    _set_effort(bond1, -effort_value)


def _get_flow(input_value: Union[Mechanical1DBond, float]):
    if isinstance(input_value, Mechanical1DBond):
        return input_value.flow
    else:
        return input_value


def _set_effort(bond: Union[Mechanical1DBond, float], effort: TemporalVar):
    if isinstance(bond, Mechanical1DBond):
        bond.effort = effort


if __name__ == '__main__':
    solver = Solver()

    n = 100
    objects = []
    for i in range(n):
        if i == 0 or i == n - 1:
            objects.append(0)
        else:
            objects.append(inertia(10, False, solver))
    for i in range(n - 1):
        object1, object2 = objects[i], objects[i + 1]
        if i == 0:
            x0 = 1
        else:
            x0 = 0
        spring(object1, object2, 1, solver, x0)

    solver.solve(500, time_step=0.1)
    for i in range(n):
        if i != 0 and i != n - 1:
            plt.plot(solver.t, objects[i].flow.values)
    # plt.plot(solver.t, objects[100].flow.values)
    plt.show()
