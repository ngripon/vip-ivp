import numpy as np

import vip_ivp as vip
from scipy.integrate import solve_ivp

from vip_ivp.application_services.api import new_system, state, solve


def rc_circuit_vip(q0=1, r=1, c=1):
    vip.new_system()
    dq = vip.loop_node()
    q = vip.integrate(dq, q0)
    dq.loop_into(-q / (r * c))
    vip.solve(10, time_step=0.001)
    return q.values


def rc_circuit_scipy(q0=1, r=1, c=1):
    # r * dq/dt + q/c = 0
    t_eval = np.linspace(0, 10, 10001)

    def dy(t, y):
        return -y[0] / (c * r)

    sol = solve_ivp(dy, [0, 10], [q0], t_eval=t_eval)
    return sol.y[0]

def test_circuit_new_api(q0=1, r=1, c=1):
    new_system()
    q=state(q0)
    q.der= -q / (r * c)
    solve(10)
    return q.values


def stiff_ode_vip():
    vip.new_system()
    dy=vip.loop_node(3)
    # Robertson problem
    y=vip.integrate(dy,[1,0,0])
    dy1 = -0.04 * y[0] + 1e4 * y[1] * y[2]
    dy2 = 0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2
    dy3 = 3e7 * y[1] ** 2
    dy.loop_into([dy1,dy2,dy3])

    vip.solve(1e3, method="BDF")
    return y.values


def stiff_ode_scipy():
    def robertson(t, y):
        dy1 = -0.04 * y[0] + 1e4 * y[1] * y[2]
        dy2 = 0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2
        dy3 = 3e7 * y[1] ** 2
        return [dy1, dy2, dy3]

    y0 = [1, 0, 0]
    t_span = (0, 1e3)  # Long time span
    t_eval=np.linspace(0,t_span[1],10001)

    # Use a stiff solver like Radau or BDF
    sol = solve_ivp(robertson, t_span, y0, method='BDF', t_eval=t_eval)
    return sol.y


def test_differential_equation_equality():
    assert np.allclose(rc_circuit_vip(), rc_circuit_scipy())


def test_differential_equation_vip(benchmark):
    result = benchmark(rc_circuit_vip)


def test_differential_equation_scipy(benchmark):
    result = benchmark(rc_circuit_scipy)

def test_differential_equation_new_api(benchmark):
    result = benchmark(rc_circuit_scipy)

def test_stiff_ode_equality():
    assert np.allclose(stiff_ode_vip(), stiff_ode_scipy())

def test_stiff_ode_vip(benchmark):
    result = benchmark(stiff_ode_vip)


def test_stiff_ode_scipy(benchmark):
    result = benchmark(stiff_ode_scipy)
