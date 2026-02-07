import numpy as np

import vip_ivp as vip

ABSOLUTE_TOLERANCE = 0.01


def test_t_end():
    a = vip.temporal(lambda t: t)
    vip.solve(10)
    assert a.t[-1] == 10


def test_rc_circuit():
    # r * dq/dt + q/c = 0
    q0_values = np.linspace(1, 10, 5)
    r_values = np.linspace(1, 10, 5)
    c_values = np.linspace(1, 10, 5)
    t_end = 100

    for q0 in q0_values:
        for R in r_values:
            for C in c_values:
                # Compute exact solution
                t = np.linspace(0, t_end, 1001)
                exact_solution = q0 * np.exp(-t / (R * C))
                # Compute solver solution
                vip.new_system()
                q = vip.state(q0)
                q.der = -q / (R * C)
                vip.solve(t_end, t_eval=t)
                error_array = exact_solution - q.values
                assert all(error_array < ABSOLUTE_TOLERANCE)


def test_harmonic_equation():
    # y'' + 9 * y = 0
    # Compute exact solution
    x_end = 10
    x = np.linspace(0, x_end, 11)
    y_exact = np.cos(3 * x) + 2 / 3 * np.sin(3 * x)
    # Compute solver solution
    y, dy = vip.n_order_state(1, 2)
    dy.der = -9 * y
    vip.solve(x_end, t_eval=x)
    error_array = y_exact - y.values
    assert all(error_array < ABSOLUTE_TOLERANCE)


def test_second_order_ode():
    # y'' + 4 * y' + 4 * y = 0
    # Compute exact solution
    x_end = 100
    x = np.linspace(0, x_end, 1001)
    y_exact = (2 * x + 1) * np.exp(-2 * x)
    # Compute solver solution
    y, dy = vip.n_order_state(1, 0)
    dy.der = -4 * dy - 4 * y

    vip.solve(x_end, t_eval=x)
    error_array = y_exact - y.values

    assert all(error_array < ABSOLUTE_TOLERANCE)


def test_bouncing_ball():
    # Arrange
    # Build true solution
    gravity = -9.81
    h = 10
    k = 0.5
    v_min = 0.1

    time_step = 0.01

    def dy(t, v0):
        return gravity * t + v0

    def y(t, v0, h0):
        return 0.5 * gravity * t ** 2 + v0 * t + h0

    def t_ground(v0, h0):
        return (-v0 - np.sqrt(v0 ** 2 - 2 * gravity * h0)) / gravity

    t = np.arange(0, 10 + time_step / 2, time_step)
    solution = np.zeros_like(t)
    t0 = 0
    v0 = 0
    current_h = h

    while True:
        t_g = t0 + t_ground(v0, current_h)
        current_sol = y(t - t0, v0, current_h)
        solution[(t0 <= t) & (t < t_g)] = current_sol[(t0 <= t) & (t < t_g)]
        v0 = dy(t_g - t0, v0)
        print(f"{t_g=} {v0=}")
        if abs(v0) < v_min:
            solution = solution[t <= t_g]
            t = t[t <= t_g]
            break
        v0 = -k * v0
        t0 = t_g
        current_h = 0

    # Create SUT
    acc = vip.temporal(gravity)
    velocity = vip.integrate(acc, 0)
    h = vip.integrate(velocity, h)

    hit_ground = h.crosses(0, "falling")
    velocity.reset_on(hit_ground, -k * velocity)
    vip.terminate_on(hit_ground & (0 > velocity) & (velocity >= -v_min))

    vip.solve(10, time_step=time_step, plot=False, include_crossing_times=False)

    # plt.plot(t, solution)
    # plt.plot(h.t, h.values)
    # plt.plot(velocity.t, velocity.values)
    # plt.plot(hit_ground.t, hit_ground.values)
    # plt.hlines([-v_min, v_min], 0, 5)
    # plt.grid()
    # plt.show()
    # Asser
    assert np.allclose(h.values, solution)
