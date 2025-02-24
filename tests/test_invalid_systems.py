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
