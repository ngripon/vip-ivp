from typing import Sequence

import numpy as np
import pytest

import vip_ivp as vip


def test_operator_overloading():
    acc = vip.loop_node()
    vit = vip.integrate(acc, 0)
    pos = vip.integrate(vit, 0)
    acc.loop_into(-pos * vit - pos / vit % vit // pos + abs(pos ** vit))

    acc(0, [1, 1])
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
    error_array=n2.values-n1.values
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
    vip.solve(10)


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
