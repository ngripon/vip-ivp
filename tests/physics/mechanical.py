import matplotlib.pyplot as plt

from main import Solver
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


def inertia(mass, solver) -> Mechanical1DBond:
    bond = Mechanical1DBond(solver)
    acc = bond.force / mass + 9.81
    bond.flow = solver.integrate(acc, 0)
    return bond


def spring(bond1: Mechanical1DBond, bond2: Mechanical1DBond, stiffness, solver: Solver, x0=0) -> None:
    speed1, speed2 = bond1.speed, bond2.speed
    x = solver.integrate(speed1 - speed2, x0)
    bond2.effort = stiffness * x
    bond1.effort = -bond2.force


if __name__ == '__main__':
    solver = Solver()
    bond1 = inertia(10, solver)
    bond2 = inertia(5, solver)
    spring(bond1, bond2, 1, solver, 10)

    solver.solve(50)
    plt.plot(bond1.effort.t, bond1.effort.values)
    plt.show()
