import pathlib
import runpy
import sys
from typing import Sequence

import matplotlib
import numpy as np
import pytest

import vip_ivp as vip


def test_plot_constant():
    a = vip.temporal(1)
    a.to_plot()
    vip.solve(10)


def test_multiple_loop_into():
    d_n1 = vip.loop_node()
    n1 = vip.integrate(d_n1, 1)
    d_n1.loop_into(-0.3 * n1)
    d_n1.loop_into(-0.2 * n1, force=True)

    d_n2 = vip.loop_node()
    n2 = vip.integrate(d_n2, 1)
    d_n2.loop_into(-0.5 * n2)

    vip.solve(10)
    error_array = n2.values - n1.values
    assert all(error_array < 1e-10)


def test_pendulum():
    dd_th = vip.loop_node()
    d_th = vip.integrate(dd_th, 0)
    th = vip.integrate(d_th, np.pi / 2)
    dd_th.loop_into(-9.81 / 1 * np.sin(th))
    vip.solve(10, time_step=0.1)


def test_source():
    u = vip.temporal(lambda t: 5 * np.sin(5 * t))
    dd_th = vip.loop_node()
    d_th = vip.integrate(dd_th, 0)
    th = vip.integrate(d_th, np.pi / 2)
    dd_th.loop_into(u - 9.81 / 1 * np.sin(th))
    vip.solve(10)


def test_loop():
    acc = vip.loop_node()
    vit = vip.integrate(acc, 0)
    pos = vip.integrate(vit, 5)
    acc.loop_into(0.1 + 1 / 10 * (-1 * vit - 1 * pos) + 5)
    vip.solve(50)


def test_integrate_scalar():
    x = vip.integrate(5, 1)
    vip.solve(10, time_step=1)
    assert np.allclose(x.values, np.linspace(1, 51, 11))


def test_no_integration():
    a = vip.temporal(lambda t: t)
    b = 2 * a

    # a.to_plot('A')
    # b.to_plot('2*A')

    vip.solve(10)


def test_system_without_integration():
    # Without time step
    a = vip.temporal(lambda t: t)
    b = 2 * a
    vip.solve(10)
    assert np.array_equal(2 * a.values, b.values)

    # With time step
    vip.new_system()
    a = vip.temporal(lambda t: t)
    b = 2 * a
    vip.solve(10, time_step=0.1)
    assert np.array_equal(2 * a.values, b.values)


def test_plant_controller():
    def controller(error):
        ki = 1
        kp = 1
        i_err = vip.integrate(ki * error, x0=0)
        return i_err + kp * error

    def plant(x):
        m = 1
        k = 1
        c = 1
        v0 = 0
        x0 = 5
        acc = vip.loop_node()
        vit = vip.integrate(acc, v0)
        pos = vip.integrate(vit, x0)
        acc.loop_into(1 / m * (x - c * vit - k * pos + x))
        return pos

    target = 1
    error = vip.loop_node()
    x = controller(error)
    y = plant(x)
    error.loop_into(target - y)

    vip.solve(50)


def test_mass_spring_bond_graph():
    def inertia(forces: Sequence[vip.TemporalVar], mass: float):
        acc = sum(forces) / mass + 9.81
        vit = vip.integrate(acc, 0)
        return vit

    def spring(speed1, speed2, stiffness: float):
        x = vip.integrate(speed1 - speed2, 0)
        force2 = k * x
        force1 = -force2
        return force1, force2

    k = 1
    mass = 1
    speed1 = vip.loop_node()
    force1, force2 = spring(speed1, 0, k)
    vit = inertia((force1,), mass)
    speed1.loop_into(vit)

    vip.solve(50)


def test_differentiate():
    d_n = vip.loop_node()
    n = vip.integrate(d_n, 1)
    d_n.loop_into(-0.5 * n)
    d_n2 = n.derivative()

    d_n2.to_plot()
    d_n.to_plot()

    vip.solve(10, time_step=0.01)

    print(d_n.values)
    print(d_n2.values)

    errors = d_n.values - d_n2.values
    assert all(errors[1:] < 0.01)


def test_integrated_differentiation():
    step = vip.temporal(lambda t: 0 if t < 1 else 1)
    # Differentiate then integrate
    d_step_bad = step.derivative()
    step_bad = vip.integrate(d_step_bad, 0)

    # Integrate then differentiate
    i_step = vip.integrate(step, 0)
    step_ok = i_step.derivative()

    step.to_plot()
    step_ok.to_plot()
    # d_step_bad.to_plot()
    step_bad.to_plot()

    vip.solve(2, time_step=0.01)


def test_float_crossing_event():
    a = vip.temporal(lambda t: t)

    crossing = a.cross_trigger(5)
    vip.terminate_on(crossing)

    vip.solve(10, time_step=1)
    print(a.values)
    print(a.t)
    assert len(a.t) == 6
    assert a.values[-1] == 5


def test_boolean_crossing_event():
    a = vip.temporal(lambda t: t)
    cond = a >= 5

    crossing = cond.cross_trigger(True)
    vip.terminate_on(crossing)

    vip.solve(10, time_step=1)
    print(cond.values)
    print(cond.t)
    assert len(a.t) == 6
    assert cond.values[-1] == True


def test_string_crossing_event():
    a = vip.temporal(lambda t: t)
    string = vip.where(a >= 5, "Aa", "Ba")

    crossing = string.cross_trigger("Aa")
    vip.terminate_on(crossing)

    vip.solve(10, time_step=1)
    print(string.values)
    print(string.t)
    assert len(a.t) == 6
    assert string.values[-1] == "Aa"


def test_bouncing_projectile_motion():
    # Parameters
    GRAVITY = -9.81
    v0 = 20
    th0 = np.radians(45)
    mu = 0.1  # Coefficient of air drag

    # Compute initial condition
    v0 = [v0 * np.cos(th0), v0 * np.sin(th0)]
    x0 = [0, 0]

    k = 0.7  # Bouncing coefficients
    v_min = 2

    # Create system
    acceleration = vip.loop_node(2)
    velocity = vip.integrate(acceleration, v0)
    position = vip.integrate(velocity, x0)
    v_norm = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
    acceleration.loop_into([-mu * velocity[0] * v_norm,
                            GRAVITY - mu * velocity[1] * v_norm])

    hit_ground = position[1].cross_trigger(0, "falling")
    bounce = velocity[1].reset_on(hit_ground & (abs(velocity[1]) > v_min), -k * velocity[1])
    vip.terminate_on(hit_ground & (abs(velocity[1]) <= v_min))

    position.to_plot("Position")
    velocity[1].to_plot()

    vip.solve(20, time_step=0.2, verbose=True)
    print(position.t)


def test_eval_events_at_all_time_points():
    # Parameters
    GRAVITY = -9.81
    v0 = 20
    th0 = np.radians(45)
    mu = 0.1  # Coefficient of air drag

    # Compute initial condition
    v0 = [v0 * np.cos(th0), v0 * np.sin(th0)]
    x0 = [0, 0]

    k = 0.7  # Bouncing coefficients
    v_min = 0.01

    # Create system
    acceleration = vip.loop_node(2)
    velocity = vip.integrate(acceleration, v0)
    position = vip.integrate(velocity, x0)
    v_norm = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
    acceleration.loop_into([-mu * velocity[0] * v_norm,
                            GRAVITY - mu * velocity[1] * v_norm])

    stopped = abs(velocity[1]) < v_min

    hit_ground=position[1].cross_trigger(
        0,
        direction="falling"
    )

    velocity[1].reset_on(hit_ground,-k * velocity[1]),

    vip.terminate_on(stopped)

    # position.to_plot("Position")
    stopped.to_plot("Stopping condition")

    vip.solve(20, time_step=0.01)
    # print(position.t)
    assert np.count_nonzero(stopped.values) == 1

def test_eval_events_at_all_time_points_with_trigger():
    # Parameters
    GRAVITY = -9.81
    v0 = 20
    th0 = np.radians(45)
    mu = 0.1  # Coefficient of air drag

    # Compute initial condition
    v0 = [v0 * np.cos(th0), v0 * np.sin(th0)]
    x0 = [0, 0]

    k = 0.7  # Bouncing coefficients
    v_min = 0.01

    # Create system
    acceleration = vip.loop_node(2)
    velocity = vip.integrate(acceleration, v0)
    position = vip.integrate(velocity, x0)
    v_norm = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
    acceleration.loop_into([-mu * velocity[0] * v_norm,
                            GRAVITY - mu * velocity[1] * v_norm])

    stopped = abs(velocity[1]) < v_min

    hit_ground=position[1].cross_trigger(
        0,
        direction="falling"
    )

    velocity[1].reset_on(hit_ground,-k * velocity[1]),


    stop_trigger=stopped.cross_trigger(True)
    vip.terminate_on(stop_trigger)

    # position.to_plot("Position")
    stopped.to_plot("Stopping condition")
    stop_trigger.to_plot()

    vip.solve(20, time_step=0.01)
    # print(position.t)
    assert np.count_nonzero(stopped.values) == 1


def test_multiple_events_at_the_same_instant():
    a = vip.temporal(1)
    ia = vip.integrate(a, 0)

    e1 = vip.set_interval(ia.action_reset_to(0), 2)
    e2 = vip.set_timeout(e1.action_disable, 6)

    ia.to_plot()

    vip.solve(10, time_step=0.01, include_events_times=False)

    assert e1.deletion_time == 6


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


def test_loads_of_recursion():
    a = vip.loop_node()
    b = a.delayed(10)
    a.loop_into(b + 1)

    a.to_plot()
    b.to_plot()

    vip.solve(1000, plot=False, time_step=0.05)

    print(a.values)


def test_big_delay():
    a = vip.loop_node()
    b = a.delayed(100)
    a.loop_into(b + 1)

    a.to_plot()
    b.to_plot()

    vip.solve(1000, time_step=0.05)

    print(a.values)


def test_cascading_events():
    # Parameters
    initial_height = 1  # m
    GRAVITY = -9.81
    k = 0.7  # Bouncing coefficient
    v_min = 0.01  # Minimum velocity need to bounce

    # Create the system
    acceleration = vip.temporal(GRAVITY)
    velocity = vip.integrate(acceleration, x0=0)
    height = vip.integrate(velocity, x0=initial_height)

    count = vip.temporal(0)

    # Create the bouncing event
    bounce = vip.where(abs(velocity) > v_min, velocity.action_reset_to(-k * velocity), vip.action_terminate)
    height.on_crossing(0, bounce, terminal=False, direction="falling")
    velocity.on_crossing(0, count.action_set_to(count + 1), direction="rising")

    # Add variables to plot
    height.to_plot("Height (m)")
    velocity.to_plot()
    count.to_plot()

    # Solve the system
    vip.solve(20, time_step=0.001)

    assert count.values[-1] == 18


def test_stiff_ode():
    dy = vip.loop_node(3)
    # Robertson problem
    y = vip.integrate(dy, [1, 0, 0])
    dy1 = -0.04 * y[0] + 1e4 * y[1] * y[2]
    dy2 = 0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2
    dy3 = 3e7 * y[1] ** 2
    dy.loop_into([dy1, dy2, dy3])

    vip.solve(1e2, method="BDF")
    print(y.values)
