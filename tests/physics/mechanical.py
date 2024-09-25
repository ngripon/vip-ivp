from main import Solver
from tests.test_valid_systems import solver


class Mechanical1DBond:
    def __init__(self, solver):
        self.speed = None
        self.force = solver.loop_node()

    @property
    def flow(self):
        return self.speed

    @property
    def effort(self):
        return self.force


def inertia(mass, solver):
    ...


def spring(bond1: Mechanical1DBond, bond2: Mechanical1DBond, stiffness, solver: Solver, x0=0):
    speed1, speed2 = bond1.speed, bond2.speed
    x = solver.integrate(speed1 - speed2, x0)
    bond2.force.loop_into(stiffness * x)
    bond1.force.loop_into(-bond2.force)
