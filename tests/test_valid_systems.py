import math

import matplotlib.pyplot as plt
import numpy as np
import pytest

from main import *


@pytest.fixture
def solver():
    return Solver()


def test_operator_overloading(solver):
    pos, vit, acc = solver.create_derivatives((0, 0))
    acc.set_value(-pos * vit - pos / vit % vit // pos + abs(pos ** vit))
    solver.solve(10)


def test_pendulum(solver):
    th, d_th, dd_th = solver.create_derivatives((0, np.pi / 2))
    dd_th.set_value(-9.81 / 1 * np.sin(th))
    solver.solve(10)
    # plt.plot(th.t, th.values)
    # plt.show()


def test_source(solver):
    u = solver.create_source(lambda t: 5 * np.sin(5 * t))
    th, d_th, dd_th = solver.create_derivatives((0, np.pi / 2))
    dd_th.set_value(-9.81 / 1 * np.sin(th) + u)
    solver.solve(10)
    print(u.values)
    plt.plot(u.t, u.values)
    plt.plot(th.t, th.values)
    plt.show()

def test_loop(solver):
    acc = solver.loop_node(0.1)
    vit = solver.integrate(acc, 0)
    pos = solver.integrate(vit, 5)
    acc.loop_into(1 / 10 * (-1 * vit - 1 * pos))
    acc.loop_into(5)
    solver.solve(50)

def test_integrate_scalar(solver):
    x=solver.integrate(5,1)
    solver.solve(10)

def plant_controller(solver):
    def controller(error):
        ki = 1
        kp = 1
        i_err = solver.integrate(ki*error,x0=0)
        return i_err + kp * error

    def plant(x):
        m = 1
        k = 1
        c = 1
        v0 = 0
        x0 = 5
        acc=solver.loop_node(1/m*x)
        vit=solver.integrate(acc,v0)
        pos=solver.integrate(vit,x0)
        acc.loop_into(1 / m * (-c * vit - k * pos + x))
        return pos

    target = 1
    error = solver.loop_node(target)
    x = controller(error)
    y = plant(x)
    error.loop_into(-y)

    solver.solve(50)
