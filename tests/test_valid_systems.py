import pytest

from main import *


@pytest.fixture
def solver():
    return Solver()


def test_operator_overloading(solver):
    pos, vit, acc = solver.create_variables((0, 0))
    acc.set_value(-pos * vit - pos / vit % vit // pos + abs(pos**vit))
    solver.solve(10)
