import pytest

import vip_ivp as vip


def test_algebraic_loop():
    x = vip.loop_node(1)
    ix = vip.integrate(x, 0)
    x.loop_into(x + ix)

    with pytest.raises(RecursionError):
        vip.solve(10)
