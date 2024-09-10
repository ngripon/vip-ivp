import pytest

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
        return (pos, vit), (vit,)

    t_final = 50
    solver.explore(f, t_final, bounds=((-10, 10), (-10, 10), (0, 10)))
