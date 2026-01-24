import numpy as np

from src.vip_ivp.application_services.api import state, solve


def test_ode():
    y = state(1)
    dy = -2 * y
    y.derivative = dy

    solve(10)

    expected_fun = lambda t: np.exp(-2 * t)
    time_vector = np.linspace(0, 10, 10)
    expected_result = expected_fun(time_vector)
    output = y(time_vector)

    np.testing.assert_almost_equal(output, expected_result, 4)
