import pytest
import numpy as np

import vip_ivp as vip

ABSOLUTE_TOLERANCE = 0.01


def test_rc_circuit():
    # r * dq/dt + q/c = 0
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
                dq = vip.loop_node()
                q = vip.integrate(dq, q0)
                dq.loop_into(-q / (R * C))
                vip.solve(t[-1], t_eval=t)
                error_array = exact_solution - q.values
                assert all(error_array < ABSOLUTE_TOLERANCE)


def test_harmonic_equation():
    # y'' + 9 * y = 0
    # Compute exact solution
    x = np.linspace(0, 10, 1001)
    y_exact = np.cos(3 * x) + 2 / 3 * np.sin(3 * x)
    # Compute solver solution
    ddy = vip.loop_node()
    dy = vip.integrate(ddy, 2)
    y = vip.integrate(dy, 1)
    ddy.loop_into(-9 * y)
    vip.solve(x[-1], t_eval=x)
    error_array = y_exact - y.values
    assert all(error_array < ABSOLUTE_TOLERANCE)


def test_second_order_ode():
    # y'' + 4 * y' + 4 * y = 0
    # Compute exact solution
    x = np.linspace(0, 100, 1001)
    y_exact = (2 * x + 1) * np.exp(-2 * x)
    # Compute solver solution
    ddy = vip.loop_node()
    dy = vip.integrate(ddy, 0)
    y = vip.integrate(dy, 1)
    ddy.loop_into(-4 * dy - 4 * y)
    vip.solve(x[-1], t_eval=x)
    error_array = y_exact - y.values
    assert all(error_array < ABSOLUTE_TOLERANCE)
