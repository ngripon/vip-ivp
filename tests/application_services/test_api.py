import numpy as np

from src.vip_ivp.application_services.api import *


def test_ode():
    y = state(1)
    dy = -2 * y
    y.derivative = dy

    solve(10)

    expected_y_fun = lambda t: np.exp(-2 * t)
    time_vector = np.linspace(0, 10, 10)
    y_expected = expected_y_fun(time_vector)
    dy_expected = -2 * y_expected
    y_output = y(time_vector)
    dy_output = dy(time_vector)

    np.testing.assert_almost_equal(y_output, y_expected, 3)
    np.testing.assert_almost_equal(dy_output, dy_expected, 3)


def test_empty_system():
    y = temporal(lambda t: np.sin(t))

    solve(10)

    np.testing.assert_array_equal(y.values, y(y.system.t_eval))


def test_conditional_variable():
    a = temporal(1)
    b = temporal(2)
    time = temporal(lambda t: t)
    sut = where(time < 5, a, b)

    solve(10)

    assert sut(0)==1
    assert sut(5)==2
    assert sut(10)==2

def test_function_wrapping():
    a=temporal(lambda t: t)
    np.random.seed(10)
    map_x = np.array([0,1,2,3,4,5])
    map_y = np.array([-1,2,5,-3,4,-8])
    sut = f(np.interp)(a, map_x, map_y)

    solve(10)

    plot(a,sut)

    for x_in,y_in in zip(map_x, map_y):
        assert sut(x_in)==y_in
    # def foo(x, i):
    #     if i>0:
    #         return 2*x
    #     return x
    #
    # i=temporal(lambda t: np.sin(t))
    # x=temporal(lambda t: t)
    #
    # sut=f(foo)(i,x)
    #
    # solve(10)
    # plot(sut)
