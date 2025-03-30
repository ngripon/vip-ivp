from typing import Sequence

import numpy as np

import vip_ivp as vip


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
    u = vip.create_source(lambda t: 5 * np.sin(5 * t))
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
    a = vip.create_source(lambda t: t)
    b = 2 * a

    a.to_plot('A')
    b.to_plot('2*A')

    vip.solve(10)


def test_system_without_integration():
    # Without time step
    a = vip.create_source(lambda t: t)
    b = 2 * a
    vip.solve(10)
    assert np.array_equal(2 * a.values, b.values)

    # With time step
    vip.new_system()
    a = vip.create_source(lambda t: t)
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
    d_n2 = vip.differentiate(n)
    vip.solve(10, time_step=0.001)

    errors = d_n.values - d_n2.values
    assert all(errors[1:] < 0.001)


def test_float_crossing_event():
    a = vip.create_source(lambda t: t)

    a.on_crossing(5, terminal=True)

    vip.solve(10, time_step=1)
    print(a.values)
    print(a.t)
    assert len(a.t) == 6
    assert a.values[-1] == 5


def test_boolean_crossing_event():
    a = vip.create_source(lambda t: t)
    cond = a >= 5

    cond.on_crossing(True, terminal=True)

    vip.solve(10, time_step=1)
    print(cond.values)
    print(cond.t)
    assert len(a.t) == 6
    assert cond.values[-1] == True


def test_string_crossing_event():
    a = vip.create_source(lambda t: t)
    string = vip.where(a >= 5, "A", "B")

    string.on_crossing("A", terminal=True)

    vip.solve(10, time_step=1)
    print(string.values)
    print(string.t)
    assert len(a.t) == 6
    assert string.values[-1] == "A"


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
    v_min = 0.01

    # Create system
    acceleration = vip.loop_node(2)
    velocity = vip.integrate(acceleration, v0)
    position = vip.integrate(velocity, x0)
    v_norm = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
    acceleration.loop_into([-mu * velocity[0] * v_norm,
                            GRAVITY - mu * velocity[1] * v_norm])

    def bounce(t, y):
        if abs(velocity[1](t, y)) > v_min:
            velocity[1].set_value(-k * velocity[1])(t, y)
        else:
            vip.terminate(t, y)

    position[1].on_crossing(
        0,
        bounce,
        terminal=False, direction="falling"
    )

    position.to_plot("Position")

    vip.solve(20, time_step=0.2)
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

    position[1].on_crossing(
        0,
        velocity[1].set_value(-k * velocity[1]),
        terminal=False, direction="falling"
    )

    stopped.on_crossing(
        True,
        terminal=True
    )
    #
    # position.to_plot("Position")
    # stopped.to_plot("Stopping condition")

    vip.solve(20, time_step=0.01)
    # print(position.t)
    assert np.count_nonzero(stopped.values)==1
