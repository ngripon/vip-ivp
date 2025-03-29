import os
from dataclasses import dataclass

import matplotlib
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
    arr = np.array([5, 4])
    arr_x0 = np.array([1, 0])
    lis = [5, 4]
    lis_x0 = [1, 0]
    dic = {"a": 5, "b": 4}
    dic_x0 = {"a": 1, "b": 0}

    # Integrate with sources
    da = vip.create_source(arr)
    a = vip.integrate(da, arr_x0)
    dd = vip.create_source(dic)
    d = vip.integrate(dd, dic_x0)

    # Integrate with python variables
    a2 = vip.integrate(arr, arr_x0)
    a3 = vip.integrate(lis, lis_x0)
    d2 = vip.integrate(dic, dic_x0)

    vip.solve(10, time_step=1)

    a0_fun = lambda t: 5 * t + 1
    a1_fun = lambda t: 4 * t

    # Get values
    print(a.values)
    print(d.values)

    # Evaluate integration from source
    assert np.allclose(a[0].values, a0_fun(a[0].t))
    assert np.allclose(a[1].values, a1_fun(a[1].t))
    assert np.allclose(d["a"].values, a0_fun(a[0].t))
    assert np.allclose(d["b"].values, a1_fun(a[1].t))
    # Evaluate integration from python variables
    assert np.allclose(a2[0].values, a0_fun(a[0].t))
    assert np.allclose(a2[1].values, a1_fun(a[1].t))
    assert np.allclose(a3[0].values, a0_fun(a[0].t))
    assert np.allclose(a3[1].values, a1_fun(a[1].t))
    assert np.allclose(d2["a"].values, a0_fun(a[0].t))
    assert np.allclose(d2["b"].values, a1_fun(a[1].t))


def test_multidimensional_integration_loop_node():
    dx = vip.loop_node(shape=2)
    x = vip.integrate(dx, [1, 2])
    dx.loop_into([-0.5 * x[0], -0.4 * x[1]])

    dx1 = vip.loop_node()
    x1 = vip.integrate(dx1, 1.0)
    dx1.loop_into(-0.5 * x1)

    dx2 = vip.loop_node()
    x2 = vip.integrate(dx2, 2.0)
    dx2.loop_into(-0.4 * x2)

    vip.solve(10, time_step=1)

    assert np.array_equal(x[0].values, x1.values)
    assert np.array_equal(x[1].values, x2.values)


def test_set_loop_node_multiple_times():
    source1 = vip.create_source(lambda t: t)
    source2 = vip.create_source(lambda t: 2 * t)
    loop = vip.loop_node()
    loop.loop_into(source1)
    loop.loop_into(source2, force=True)
    vip.solve(10, time_step=1)
    assert np.allclose(loop.values, np.linspace(0, 27, 10))
    assert loop.expression == "source1 + source2"


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


def test_plot_collections():
    arr = np.array([5, 4])
    arr_x0 = np.array([1, 0])
    dic = {"a": 2, "b": 3}
    dic_x0 = {"a": 1, "b": 0}
    arr2 = np.arange(6).reshape(2, 3)

    # Integrate with sources
    da = vip.create_source(arr)
    a = vip.integrate(da, arr_x0)
    dd = vip.create_source(dic)
    d = vip.integrate(dd, dic_x0)
    a2 = vip.create_source(arr2)

    matplotlib.use('Agg')
    a.to_plot("Array")
    a2.to_plot("2D array")
    d.to_plot("Dict")

    vip.solve(10, 0.1)


def test_array_comparisons_operators():
    f1 = lambda t: t
    f2 = lambda t: 2.5 * t

    a = vip.create_source([5, 5])
    b = vip.create_source([f1, f2])

    # Equality
    equ_arr = a == b
    equ1 = b[0] == a[0]
    equ2 = b[1] == a[1]

    # Inequality
    neq_arr = a != b
    neq1 = b[0] != a[0]
    neq2 = b[1] != a[1]

    # Less than
    lt_arr = b < a
    lt1 = b[0] < a[0]
    lt2 = b[1] < a[1]

    # Greater than
    gt_arr = b > a
    gt1 = b[0] > a[0]
    gt2 = b[1] > a[1]

    # Less than or equal
    le_arr = b <= a
    le1 = b[0] <= a[0]
    le2 = b[1] <= a[1]

    # Greater than or equal
    ge_arr = b >= a
    ge1 = b[0] >= a[0]
    ge2 = b[1] >= a[1]

    vip.solve(10, 1)

    # Assertions for equality
    assert np.array_equal(equ_arr[0].values, equ1.values)
    assert np.array_equal(equ_arr[1].values, equ2.values)

    # Assertions for inequality
    assert np.array_equal(neq_arr[0].values, neq1.values)
    assert np.array_equal(neq_arr[1].values, neq2.values)

    # Assertions for less than
    assert np.array_equal(lt_arr[0].values, lt1.values)
    assert np.array_equal(lt_arr[1].values, lt2.values)

    # Assertions for greater than
    assert np.array_equal(gt_arr[0].values, gt1.values)
    assert np.array_equal(gt_arr[1].values, gt2.values)

    # Assertions for less than or equal
    assert np.array_equal(le_arr[0].values, le1.values)
    assert np.array_equal(le_arr[1].values, le2.values)

    # Assertions for greater than or equal
    assert np.array_equal(ge_arr[0].values, ge1.values)
    assert np.array_equal(ge_arr[1].values, ge2.values)


def test_bounded_integration():
    a = vip.create_source(1)
    # Bounded by constant
    ia = vip.integrate(a, 0, max=5)

    ia.to_plot("Integral")

    vip.solve(10, time_step=0.1)

    assert ia.values[-1] == 5
