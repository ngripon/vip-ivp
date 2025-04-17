import numpy as np

import vip_ivp as vip
from scipy.integrate import solve_ivp


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


def test_differential_equation_equality():
    assert np.allclose(rc_circuit_vip(), rc_circuit_scipy())


def test_differential_equation_vip(benchmark):
    result = benchmark(rc_circuit_vip)


def test_differential_equation_scipy(benchmark):
    result = benchmark(rc_circuit_scipy)
