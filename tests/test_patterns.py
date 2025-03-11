from dataclasses import dataclass

import numpy as np

import vip_ivp as vip


def test_collections_get_methods():
    @dataclass
    class Foo:
        bar: int
        lol: float

    l = vip.create_source([1, 2, 3, 4])
    d = vip.create_source({"a": 5, "b": 6})
    arr = vip.create_source(np.array([1, 2, 3, 4, 5]))
    obj = vip.create_source(Foo(1, 5.5))
    arr_slice = arr[2:]

    b = vip.integrate(l[0], 0)
    c = vip.integrate(d["b"], 0)
    d = vip.integrate(arr_slice[0], 0)
    e = vip.integrate(obj.lol, 0)

    vip.solve(10)
