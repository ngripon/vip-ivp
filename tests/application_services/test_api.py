import numpy as np

from src.vip_ivp.application_services.api import state, solve


def test_ode():
    y = state(1)
    dy = -2 * y
    y.derivative = dy

    solve(10)

    expected_y_fun = lambda t: np.exp(-2 * t)
    time_vector = np.linspace(0, 10, 10)
    y_expected = expected_y_fun(time_vector)
    dy_expected=-2*y_expected
    y_output = y(time_vector)
    dy_output = dy(time_vector)

    np.testing.assert_almost_equal(y_output, y_expected, 3)
    np.testing.assert_almost_equal(dy_output, dy_expected, 3)
