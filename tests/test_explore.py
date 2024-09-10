import pytest
from matplotlib import pyplot as plt

from main import Solver


@pytest.fixture
def solver():
    return Solver()


def test_single_line(solver):
    def f(k=2, c=3, m=5, x0=1, v0=1):
        acc = solver.loop_node(1 / m)
        vit = solver.integrate(acc, v0)
        pos = solver.integrate(vit, x0)
        acc.loop_into(1 / m * (-c * vit - k * pos))
        return pos

    t_final = 50
    solver.explore(f, t_final, bounds=((-10, 10), (-10, 10), (0, 10)))


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
