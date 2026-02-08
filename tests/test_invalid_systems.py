import pytest

from vip_ivp import *


def test_crossing_integration_bounds():
    a = temporal(1)
    signal = temporal(lambda t: 6 - t)
    ia = state(0, -1,signal)
    ia.der=a

    with pytest.raises(ValueError):
        try:
            solve(10)
            _ = ia.values
        except Exception as e:
            print(e)
            raise e

def test_if_statement():
    time = temporal(lambda t: t)
    with pytest.raises(ValueError):
        if time < 5:
            step = temporal(0)
        else:
            step = temporal(1)

        solve(10)

def test_x0_outside_bound():
    with pytest.raises(ValueError):
        x = state(0, 2, 10)
