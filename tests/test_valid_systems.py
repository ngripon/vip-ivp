import pathlib
import runpy

import matplotlib
import numpy as np
import pytest

import vip_ivp as vip


# TODO: clarifies the purpose of these tests and splits into multiple files

def test_differentiate():
    n = vip.state(1)
    n.der = -0.5 * n
    d_n2 = n.compute_derivative(t=0.001)

    vip.solve(10)

    errors = n.der.values - d_n2.values
    assert all(errors[1:] < 0.01)


def test_float_crossing_event():
    a = vip.temporal(lambda t: t)

    crossing = a.crosses(5)
    vip.when(crossing, vip.terminate)

    vip.solve(10, step_eval=1)
    print(a.values)
    print(a.t)
    assert len(a.t) == 6
    assert a.values[-1] == 5


def test_boolean_crossing_event():
    a = vip.temporal(lambda t: t)
    cond = a >= 5

    crossing = cond.crosses(True)
    vip.when(crossing, vip.terminate)

    vip.solve(10, step_eval=1)
    print(cond.values)
    print(cond.t)
    assert len(a.t) == 6
    assert cond.values[-1] == True


def test_string_crossing_event():
    a = vip.temporal(lambda t: t)
    string = vip.where(a >= 5, "Aa", "Ba")

    crossing = string.crosses("Aa")
    vip.when(crossing, vip.terminate)

    vip.solve(10, step_eval=1)
    print(string.values)
    print(string.t)
    assert len(a.t) == 6
    assert string.values[-1] == "Aa"


# def test_eval_events_at_all_time_points():
#     # Parameters
#     GRAVITY = -9.81
#     v0 = 20
#     th0 = np.radians(45)
#     mu = 0.1  # Coefficient of air drag
#
#     # Compute initial condition
#     v0 = [v0 * np.cos(th0), v0 * np.sin(th0)]
#     x0 = [0, 0]
#
#     k = 0.7  # Bouncing coefficients
#     v_min = 0.01
#
#     # Create system
#     acceleration = vip.loop_node(2)
#     velocity = vip.integrate(acceleration, v0)
#     position = vip.integrate(velocity, x0)
#     v_norm = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
#     acceleration.loop_into([-mu * velocity[0] * v_norm,
#                             GRAVITY - mu * velocity[1] * v_norm])
#
#     stopped = abs(velocity[1]) < v_min
#
#     hit_ground = position[1].crosses(
#         0,
#         direction="falling"
#     )
#
#     velocity[1].reset_on(hit_ground, -k * velocity[1]),
#
#     vip.terminate_on(stopped)
#
#     # position.to_plot("Position")
#     stopped.to_plot("Stopping condition")
#
#     vip.solve(20, time_step=0.01)
#     # print(position.t)
#     assert np.count_nonzero(stopped.values) == 1


# def test_eval_events_at_all_time_points_with_trigger():
#     # Parameters
#     GRAVITY = -9.81
#     v0 = 20
#     th0 = np.radians(45)
#     mu = 0.1  # Coefficient of air drag
#
#     # Compute initial condition
#     v0 = [v0 * np.cos(th0), v0 * np.sin(th0)]
#     x0 = [0, 0]
#
#     k = 0.7  # Bouncing coefficients
#     v_min = 0.01
#
#     # Create system
#     acceleration = vip.loop_node(2)
#     velocity = vip.integrate(acceleration, v0)
#     position = vip.integrate(velocity, x0)
#     v_norm = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
#     acceleration.loop_into([-mu * velocity[0] * v_norm,
#                             GRAVITY - mu * velocity[1] * v_norm])
#
#     stopped = abs(velocity[1]) < v_min
#
#     hit_ground = position[1].crosses(
#         0,
#         direction="falling"
#     )
#
#     velocity[1].reset_on(hit_ground, -k * velocity[1]),
#
#     stop_trigger = stopped.crosses(True)
#     vip.terminate_on(stop_trigger)
#
#     # position.to_plot("Position")
#     stopped.to_plot("Stopping condition")
#     stop_trigger.to_plot()
#
#     vip.solve(20, time_step=0.01)
#     # print(position.t)
#     assert np.count_nonzero(stopped.values) == 1


# def test_multiple_events_at_the_same_instant():
#     a = vip.temporal(1)
#     ia = vip.integrate(a, 0)
#
#     inhibit = vip.integrate(0, 1)
#
#     t1 = vip.interval_trigger(2)
#     t2 = vip.timeout_trigger(6)
#     e1 = ia.reset_on(t1 & inhibit, 0)
#     e2 = inhibit.reset_on(t2, 0)
#
#     ia.to_plot()
#
#     vip.solve(10, time_step=0.01)
#
#     assert ia.values[-1] == 4


def test_demos():
    matplotlib.use("Agg")
    demo_dir = pathlib.Path(__file__).parent.parent / "demos"

    # Get all .py files in demo folder
    demo_scripts = list(demo_dir.glob("*.py"))
    demo_scripts = [path for path in demo_scripts if "explore" not in str(path)]
    print(demo_scripts)

    for script_path in demo_scripts:
        vip.new_system()
        print(script_path)
        try:
            runpy.run_path(str(script_path), run_name="__main__")
        except Exception as e:
            pytest.fail(f"Demo script {script_path.name} raised an exception: {e}")


def test_forgiving_temporal_functions():
    """
    Test if the temporal function can accept functions that do not support array inputs
    """

    def non_vec_fun(t):
        return max(1.0 - 0.005 * t, 0)

    a = vip.temporal(non_vec_fun)

    a.to_plot()
    vip.solve(10)


def test_forgiving_f():
    def non_vec_fun(x):
        return max(x, 1)

    # Test f
    a = vip.temporal(lambda t: t)
    b = vip.f(non_vec_fun)(a)

    a.to_plot()
    b.to_plot()

    vip.solve(10)





# def test_cascading_events():
#     # Parameters
#     initial_height = 1  # m
#     GRAVITY = -9.81
#     k = 0.7  # Bouncing coefficient
#     v_min = 0.01  # Minimum velocity need to bounce
#
#     # Create the system
#     acceleration = vip.temporal(GRAVITY)
#     velocity = vip.integrate(acceleration, x0=0)
#     height = vip.integrate(velocity, x0=initial_height)
#
#     count = vip.temporal(0)
#
#     # Create the bouncing event
#     bounce = vip.where(abs(velocity) > v_min, velocity.action_reset_to(-k * velocity), vip.action_terminate)
#     height.on_crossing(0, bounce, terminal=False, direction="falling")
#     velocity.on_crossing(0, count.action_set_to(count + 1), direction="rising")
#
#     # Add variables to plot
#     height.to_plot("Height (m)")
#     velocity.to_plot()
#     count.to_plot()
#
#     # Solve the system
#     vip.solve(20, time_step=0.001)
#
#     assert count.values[-1] == 18


# def test_stiff_ode():
#     dy = vip.loop_node(3)
#     # Robertson problem
#     y = vip.integrate(dy, [1, 0, 0])
#     dy1 = -0.04 * y[0] + 1e4 * y[1] * y[2]
#     dy2 = 0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2
#     dy3 = 3e7 * y[1] ** 2
#     dy.loop_into([dy1, dy2, dy3])
#
#     vip.solve(1e2, method="BDF")
#     print(y.values)
