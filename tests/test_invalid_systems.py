import pytest

from main import *


@pytest.fixture
def solver():
    return Solver()


def test_algebraic_loop(solver):
    x = solver.loop_node(1)
    ix = solver.integrate(x, 0)
    x.loop_into(x + ix)

    with pytest.raises(RecursionError):
        solver.solve(10)
