import numpy as np

from src.vip_ivp.domain.system import get_integrated_variable, IVPSystem


def test_ode_result():
    y = get_integrated_variable(0)
    dy = -2 * y
    y0 = 1
    sut = IVPSystem((dy,), (y0,))

    sol_fun = sut.solve(10)

    expected_fun = lambda t: np.exp(-2 * t)
    time_vector = np.linspace(0, 10, 10)
    expected_result = expected_fun(time_vector)
    output = y(time_vector, sol_fun(time_vector))

    np.testing.assert_almost_equal(expected_result, output, 4)
