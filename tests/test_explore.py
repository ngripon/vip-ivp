from typing import Sequence

import numpy as np
import pytest
from matplotlib import pyplot as plt

from main import *


@pytest.fixture
def solver():
    return Solver()


def test_single_line(solver):
    def f(k=2, c=3, m=5, x0=0, v0=0):
        acc = solver.loop_node(1 / m)
        vit = solver.integrate(acc, v0)
        pos = solver.integrate(vit, x0)
        acc.loop_into(1 / m * (-c * vit - k * pos))
        return pos

    t_final = 50
    solver.explore(f, t_final, bounds=((0, 10), (0, 10), (0.1, 10), (-5, 5), (-5, 5)))


def test_multiple_lines(solver):
    def f(k=2, c=3, m=5, x0=1, v0=1):
        acc = solver.loop_node(1 / m)
        vit = solver.integrate(acc, v0)
        pos = solver.integrate(vit, x0)
        acc.loop_into(1 / m * (-c * vit - k * pos))
        return pos, vit, acc

    t_final = 50
    solver.explore(f, t_final, bounds=((-10, 10), (-10, 10), (0, 10)))


def test_multiple_plots(solver):
    def f(k=2, c=3, m=5, x0=1, v0=1):
        acc = solver.loop_node(1 / m)
        vit = solver.integrate(acc, v0)
        pos = solver.integrate(vit, x0)
        acc.loop_into(1 / m * (-c * vit - k * pos))
        return (pos, vit), (acc,)

    t_final = 50
    solver.explore(f, t_final, bounds=((-10, 10), (-10, 10), (0, 10)))


def test_plant_controller(solver):
    def plant_controller(kp=1, ki=0):
        def controller(error):
            i_err = solver.integrate(ki * error, x0=0)
            return i_err + kp * error

        def plant(x):
            m = 1
            k = 1
            c = 1
            v0 = 0
            x0 = 5
            acc = solver.loop_node(1 / m * x)
            vit = solver.integrate(acc, v0)
            pos = solver.integrate(vit, x0)
            acc.loop_into(1 / m * (-c * vit - k * pos + x))
            return pos

        target = 1
        error = solver.loop_node(target)
        x = controller(error)
        y = plant(x)
        error.loop_into(-y)
        return y

    solver.explore(plant_controller, 50)


def test_mass_spring_bond_graph(solver):
    def f(k=1, m=1):
        def inertia(force: TemporalVar, mass: float):
            acc = force / mass + 9.81
            vit = solver.integrate(acc, 0)
            return vit

        def spring(speed1, speed2, stiffness: float):
            x = solver.integrate(speed1 - speed2, 0)
            force2 = k * x
            force1 = -force2
            return force1, force2

        speed = solver.loop_node(0)

        force, _ = spring(speed, 0, k)
        speed.loop_into(inertia(force, m))

        return speed

    solver.explore(f, 50)
