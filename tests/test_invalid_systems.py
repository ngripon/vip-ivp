import pytest

from main import *


def test_algebraic_loop():
    x, dx = solver.create_variables(0)
    with pytest.raises(RecursionError):
        dx.set_value(dx)
    dx.set_value(5 * dx)
    with pytest.raises(RecursionError):
        solver.solve(10)
