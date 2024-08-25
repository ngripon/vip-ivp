import pytest

from main import *


def test_algebraic_loop():
    x, dx = solver.create_variables(0)
    with pytest.raises(RecursionError):
        dx.set_value(dx)

    dx.set_value(5 * dx)
    with pytest.raises(RecursionError):
        solver.solve(10)

    y,dy,ddy=solver.create_variables((0,0))
    ddy.set_value(y+ddy)
    with pytest.raises(RecursionError):
        solver.solve(10)


def test_uninitialized_feed():
    x,dx=solver.create_variables(0)
    with pytest.raises(ValueError) as e:
        solver.solve(10)
    print(e)


