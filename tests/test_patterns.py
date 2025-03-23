from dataclasses import dataclass

import numpy as np

import vip_ivp as vip


def test_collections_get_methods():
    @dataclass
    class Foo:
        bar: int
        lol: float

    l = vip.create_source([1, 2, 3, 4])
    di = vip.create_source({"a": 5, "b": 6})
    arr = vip.create_source(np.array([1, 2, 3, 4, 5]))
    obj = vip.create_source(Foo(1, 5.5))
    arr_slice = arr[2:]

    b = vip.integrate(l[0], 0)
    c = vip.integrate(di["b"], 0)
    d = vip.integrate(arr_slice[0], 0)
    e = vip.integrate(obj.lol, 0)

    vip.solve(10, time_step=1)

    print(l[0].values)
    print(di["b"].values)
    print(arr_slice[0].values)
    print(obj.lol.values)


def test_use_numpy_function():
    x = vip.create_source(5)
    a = vip.integrate(x, 0)

    np.random.seed(10)
    map_length = 100
    map_x = np.linspace(0, 100, map_length)
    map_y = np.random.rand(map_length)
    y = vip.f(np.interp)(a, map_x, map_y)
    vip.solve(10, time_step=0.01)

    error_array = y.values - np.interp(a.values, map_x, map_y)

    assert np.all(error_array == 0)

def test_multidimensional_integration():
    da=vip.create_source(np.array([5,4]))
    print(da)
    a=vip.integrate(da,[0,0])
    vip.solve(10)


def test_conditions():
    a = vip.create_source(lambda t: t)
    b = vip.create_source(5)
    c = vip.where(a < 5, a, b)
    vip.solve(10, time_step=0.1)
    assert np.array_equal(c.values, np.where(a.values < 5, a.values, b.values))
