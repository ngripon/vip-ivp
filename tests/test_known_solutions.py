import matplotlib.pyplot as plt
import pytest
import numpy as np

import main as vip

ABSOLUTE_TOLERANCE = 0.01


@pytest.fixture(autouse=True)
def clear_solver_before_tests():
    vip.clear()


def test_rc_circuit():
    q0_values = np.linspace(1, 10, 10)
    r_values = np.linspace(1, 10, 10)
    c_values = np.linspace(1, 10, 10)

    for q0 in q0_values:
        for R in r_values:
            for C in c_values:
                # Compute exact solution
                t = np.linspace(0, 100, 1001)
                exact_solution = q0 * np.exp(-t / (R * C))
                # Compute solver solution
                vip.clear()
                dq = vip.loop_node(0)
                q = vip.integrate(dq, q0)
                dq.loop_into(-q / (R * C))
                vip.solve(t[-1], t_eval=t)
                error_array = exact_solution - q.values
                assert all(error_array < ABSOLUTE_TOLERANCE)
