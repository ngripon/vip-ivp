import math
import operator

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.random.mtrand import Sequence

from main import *


@pytest.fixture
def solver():
    return Solver()


def test_operator_overloading(solver):
    acc = solver.loop_node(0)
    vit = solver.integrate(acc, 0)
    pos = solver.integrate(vit, 0)
    acc.loop_into(-pos * vit - pos / vit % vit // pos + abs(pos ** vit))

    acc(0,[1,1])
    solver.solve(10)


def test_pendulum(solver):
    dd_th = solver.loop_node(0)
    d_th = solver.integrate(dd_th, 0)
    th = solver.integrate(d_th, np.pi / 2)
    dd_th.loop_into(-9.81 / 1 * np.sin(th))

    solver.solve(10)
    plt.plot(th.t, th.values)
    plt.show()


def test_source(solver):
    u = solver.create_source(lambda t: 5 * np.sin(5 * t))
    dd_th = solver.loop_node(u)
    d_th = solver.integrate(dd_th, 0)
    th = solver.integrate(d_th, np.pi / 2)
    dd_th.loop_into(-9.81 / 1 * np.sin(th))
    solver.solve(10)


def test_loop(solver):
    acc = solver.loop_node(0.1)
    vit = solver.integrate(acc, 0)
    pos = solver.integrate(vit, 5)
    acc.loop_into(1 / 10 * (-1 * vit - 1 * pos))
    acc.loop_into(5)
    solver.solve(50)


def test_integrate_scalar(solver):
    x = solver.integrate(5, 1)
    solver.solve(10)


def test_plant_controller(solver):
    def controller(error):
        ki = 1
        kp = 1
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

    solver.solve(50)


def test_mass_spring_bond_graph(solver):
    def inertia(forces: Sequence[TemporalVar], mass: float):
        acc = np.sum(forces) / mass + 9.81
        vit = solver.integrate(acc, 0)
        return vit

    def spring(speed1, speed2, stiffness: float):
        x = solver.integrate(speed1 - speed2, 0)
        force2 = k * x
        force1 = -force2
        return force1, force2

    k = 1
    mass = 1
    speed1 = solver.loop_node(0)
    force1, force2 = spring(speed1, 0, k)
    vit = inertia((force1,), mass)
    speed1.loop_into(vit)

    solver.solve(50)
