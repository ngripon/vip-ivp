import pytest

import vip_ivp as vip


def test_algebraic_loop():
    x = vip.loop_node()
    ix = vip.integrate(x, 0)
    x.loop_into(x + ix)

    with pytest.raises(RecursionError):
        vip.solve(10)


def test_set_loop_node_two_times():
    x = vip.loop_node()
    x.loop_into(6)
    with pytest.raises(Exception):
        x.loop_into(5)


def test_crossing_integration_bounds():
    a = vip.create_source(1)
    signal=vip.create_source(lambda t:6-t)
    ia = vip.integrate(a, 0, max=signal, min=-1)

    ia.to_plot("Integral")

    with pytest.raises(ValueError):
        vip.solve(10, time_step=1)
