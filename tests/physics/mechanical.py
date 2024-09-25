from main import Solver


class Mechanical1DBond:
    def __init__(self):
        self.speed = None
        self.force = None

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
    bond2.force = stiffness * x
    bond1.force = -bond2.force
