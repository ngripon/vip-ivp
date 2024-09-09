import pytest

from main import *


@pytest.fixture
def solver():
    return Solver()


def test_algebraic_loop(solver):
    x=solver.loop_node(1)
    x.loop_into(x)

    with pytest.raises(RecursionError):
        solver.solve(10)
    #
    # y, dy, ddy = solver.create_derivatives((0, 0))
    # ddy.set_value(y + ddy)
    # with pytest.raises(RecursionError):
    #     solver.solve(10)



