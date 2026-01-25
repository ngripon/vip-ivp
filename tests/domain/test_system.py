import numpy as np

from vip_ivp.domain.system import IVPSystem, create_system_output


def test_ode_result():
    y = create_system_output(0)
    dy = lambda t, y_: -2 * y(t, y_)
    y0 = 1
    sut = IVPSystem((dy,), (y0,))

    sol_fun = sut.solve(10)

    expected_fun = lambda t: np.exp(-2 * t)
    time_vector = np.linspace(0, 10, 10)
    expected_result = expected_fun(time_vector)
    output = y(time_vector, sol_fun(time_vector))

    np.testing.assert_almost_equal(expected_result, output, 4)


def test_empty_system_result():
    sut = IVPSystem((), ())

    sol_fun = sut.solve(10)

    assert sol_fun(0).shape == (0,)
    assert sol_fun(np.array([1, 2, 3])).shape == (0, 3)
