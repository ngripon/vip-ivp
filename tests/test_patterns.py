import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

import vip_ivp as vip


def test_collections_get_methods():
    l = vip.create_source([1, 2, 3, 4])
    di = vip.create_source({"a": 5, "b": 6})
    arr = vip.create_source(np.array([1, 2, 3, 4, 5]))
    arr_slice = arr[2:]

    b = vip.integrate(l[0], 0)
    c = vip.integrate(di["b"], 0)
    d = vip.integrate(arr_slice[0], 0)

    vip.solve(10, time_step=1)

    print(l[0].values)
    print(di["b"].values)
    print(arr_slice[0].values)


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


def test_multidimensional_integration_source():
    da = vip.create_source(np.array([5, 4]))
    a = vip.integrate(da, np.array([1, 0]))
    # d = vip.integrate({"a": 5, "b": 4}, {"a": 1, "b": 0})
    vip.solve(10)

    print(da.solver.y)
    print(da.solver.t)

    a0_fun = lambda t: 5 * t + 1
    a1_fun = lambda t: 4 * t
    print(da(da.t,da.solver.t))
    assert np.allclose(a[0].values, a0_fun(a[0].t))
    assert np.allclose(a[1].values, a1_fun(a[1].t))
    # assert np.allclose(d["a"].values, a0_fun(a[0].t))
    # assert np.allclose(d["b"].values, a1_fun(a[1].t))

def test_multidimensional_integration_loop_node():
    ...



def test_conditions():
    a = vip.create_source(lambda t: t)
    b = vip.create_source(5)
    c = vip.where(a < 5, a, b)
    vip.solve(10, time_step=0.1)
    assert np.array_equal(c.values, np.where(a.values < 5, a.values, b.values))


def test_scenario_interpolation():
    scenario_dict = {"t": [0, 1, 2, 3, 4], "a": [1, 2, 3, 4, 5], "b": [0, 10, -10, 10, -10]}
    scenario_df = pd.DataFrame(scenario_dict)
    scenario_json = "tests/files/scenarii/scenario_basic.json"
    scenario_csv = "tests/files/scenarii/scenario_basic.csv"

    scenarii_inputs = [scenario_df, scenario_dict, scenario_json, scenario_csv]

    for scenario in scenarii_inputs:
        print(f"Test scenario: {scenario}")
        vip.new_system()
        variables = vip.create_scenario(scenario, "t", sep=";")

        vip.solve(4, time_step=0.5)

        a = variables["a"]
        b = variables["b"]

        assert a.values[0] == 1
        assert a.values[1] == 1.5
        assert b.values[0] == 0
        assert b.values[1] == 5
        assert b.values[3] == 0
