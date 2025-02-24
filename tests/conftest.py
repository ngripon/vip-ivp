import pytest

import vip_ivp as vip


@pytest.fixture(autouse=True)
def clear_solver_before_tests():
    vip.clear()
