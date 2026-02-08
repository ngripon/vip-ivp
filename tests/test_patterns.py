import time

import matplotlib
import numpy as np
import pandas as pd

import vip_ivp as vip


def test_collections_get_methods():
    l = vip.temporal([1, 2, 3, 4])
    di = vip.temporal({"a": 5, "b": 6})
    arr = vip.temporal(np.array([1, 2, 3, 4, 5]))
    arr_slice = arr[2:]

    b = vip.state(0, derivative=l[0])
    c = vip.state(0, derivative=di["b"])
    d = vip.state(0, derivative=arr_slice[0])

    vip.solve(10)

    print(l[0].values)
    print(di["b"].values)
    print(arr_slice[0].values)

    # TODO: Add assert methods


def test_logical_operators():
    x_a = vip.temporal(lambda t: np.sin(t))
    x_b = vip.temporal(lambda t: np.sin(t + np.pi / 2))
    a = x_a >= 0
    b = x_b >= 0

    # and
    a_and_b = a & b
    b_and_a = b & a
    # or
    a_or_b = a | b
    b_or_a = b | a
    # xor
    a_xor_b = a ^ b
    b_xor_a = b ^ a
    # not
    not_a = ~a

    vip.solve(20)
    # Assertions
    # and
    assert np.array_equal(a_and_b.values, np.logical_and(a.values, b.values))
    assert np.array_equal(b_and_a.values, np.logical_and(a.values, b.values))
    # or
    assert np.array_equal(a_or_b.values, np.logical_or(a.values, b.values))
    assert np.array_equal(b_or_a.values, np.logical_or(a.values, b.values))
    # xor
    assert np.array_equal(a_xor_b.values, np.logical_xor(a.values, b.values))
    assert np.array_equal(b_xor_a.values, np.logical_xor(a.values, b.values))
    # not
    assert np.array_equal(not_a.values, np.logical_not(a.values))


def test_t_eval_over_time_step():
    t_eval = [0, 2, 2.5, 7]
    a = vip.temporal(1)

    vip.solve(10, t_eval=t_eval)

    assert np.array_equal(t_eval, a.t)


def test_solvers():
    methods = ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA']
    for method in methods:
        vip.new_system()
        n = vip.state(1)
        n.der = -0.5 * n

        start = time.time()
        vip.solve(10, method=method)
        print(f"{method}: {time.time() - start}s")
    # TODO: Add assert


def test_use_numpy_function():
    x = vip.temporal(5)
    a = vip.state(0, derivative=x)

    np.random.seed(10)
    map_length = 100
    map_x = np.linspace(0, 100, map_length)
    map_y = np.random.rand(map_length)
    y = vip.f(np.interp)(a, map_x, map_y)

    vip.solve(10)

    error_array = y.values - np.interp(a.values, map_x, map_y)

    assert np.all(error_array == 0)


def test_use_basic_function():
    def basic_function(x: float) -> int:
        if x > 0:
            return 1
        else:
            return -1

    input = vip.temporal(lambda t: np.cos(t))
    output = vip.f(basic_function)(input)

    vip.solve(10)

    # TODO: Add assert

# TODO: Check this
# def test_use_numpy_method():
#     array_source = vip.temporal([lambda t: t, lambda t: 2 * t, lambda t: 3 * t, lambda t: 4 * t])
#     reshaped_array = array_source.m(array_source.output_type.reshape)((2, 2))
#     square_array_source = vip.temporal([[lambda t: t, lambda t: 2 * t], [lambda t: 3 * t, lambda t: 4 * t]])
#     # reshaped_array.to_plot()
#     vip.solve(3, step_eval=1)
#
#     print(reshaped_array.values)
#     print(square_array_source.values)
#     # Bug explanation: When the TemporalVariable possess a numpy array that is computed from an operation, it does not
#     # manipulate the shape to have the time dimensions the last instead of the first.
#
#     assert np.testing.assert_array_equal(reshaped_array.values, square_array_source.values)


def test_conditions():
    a = vip.temporal(lambda t: t)
    b = vip.temporal(5)
    c = vip.where(a < 5, a, b)
    vip.solve(10)
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
        scenario_variable = vip.create_scenario(scenario, time_key="t", sep=";", interpolation_kind="linear")

        vip.solve(4, step_eval=0.5)

        a = scenario_variable["a"]
        b = scenario_variable["b"]

        assert a.values[0] == 1
        assert a.values[1] == 1.5
        assert b.values[0] == 0
        assert b.values[1] == 5
        assert b.values[3] == 0


def test_array_comparisons_operators():
    f1 = lambda t: t
    f2 = lambda t: 2.5 * t

    a = vip.temporal([5, 5])
    b = vip.temporal([f1, f2])

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

    vip.solve(10)

    # Assertions for equality
    equ_arr.values
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


def test_bounded_integration_by_constant():
    a = vip.temporal(1)
    ia_inc = vip.state(2, 2, 5, a)
    ia_dec = vip.state(5, 2, 5, -a)

    vip.solve(10)

    assert ia_inc.values[-1] == 5
    assert ia_inc.values[0] == 2
    assert ia_dec.values[0] == 5
    assert ia_dec.values[-1] == 2


def test_bounded_integration_by_variable():
    a = vip.temporal(1)
    signal = vip.temporal(lambda t: 6 - t)
    ia_inc = vip.state(0, upper_bound=signal, derivative=a)
    ia_dec = vip.state(0, lower_bound=-signal, derivative=-a)

    vip.solve(10, step_eval=1)

    assert ia_inc(3) == 3
    assert ia_inc.values[-1] == -4
    assert ia_dec(3) == -3
    assert ia_dec.values[-1] == 4


def test_terminate_event():
    a = vip.temporal(lambda t: t)

    crossing = a.crosses(6)
    vip.when(crossing & (a < 3), vip.terminate)

    vip.solve(10)

    assert a.t[-1] == 6

# TEMPORARILY DELETED. IT WILL BECOME RELEVANT AGAIN WITH STATEFUL VARIABLES

# def test_increment_timeout():
#     count = vip.temporal(0)
#
#     vip.set_timeout(count.action_set_to(count + 1), 2)
#
#     count.to_plot()
#     vip.solve(10, time_step=1)
#
#     assert count.values[0] == 0
#     assert count.values[2] == 1
#     assert count.values[-1] == 1
#
#
# def test_increment_interval():
#     count = vip.temporal(0)
#     vip.set_interval(count.action_set_to(count + 1), 2)
#
#     count.to_plot()
#     vip.solve(10, time_step=1)
#
#     print(count.values)
#     print(count.t)
#
#     assert count.values[0] == 0
#     assert count.values[2] == 1
#     assert count.values[4] == 2
#     assert count.values[6] == 3
#     assert count.values[8] == 4
#     assert count.values[10] == 5
