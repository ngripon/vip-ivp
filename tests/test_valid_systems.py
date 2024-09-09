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
    u=solver.create_source(lambda t:5*np.sin(5*t))
    th, d_th, dd_th = solver.create_derivatives((0, np.pi / 2))
    dd_th.set_value(-9.81 / 1 * np.sin(th)+u)
    solver.solve(10)
    print(u.values)
    plt.plot(u.t, u.values)
    plt.plot(th.t, th.values)
    plt.show()
