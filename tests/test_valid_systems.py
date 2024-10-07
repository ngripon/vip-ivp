
import matplotlib.pyplot as plt
import numpy as np
from numpy.random.mtrand import Sequence

import main as vip



def test_operator_overloading():
    vip.clear()
    acc = vip.loop_node(0)
    vit = vip.integrate(acc, 0)
    pos = vip.integrate(vit, 0)
    acc.loop_into(-pos * vit - pos / vit % vit // pos + abs(pos ** vit))

    acc(0,[1,1])
    vip.solve(10)


def test_pendulum():
    vip.clear()
    dd_th = vip.loop_node(0)
    d_th = vip.integrate(dd_th, 0)
    th = vip.integrate(d_th, np.pi / 2)
    dd_th.loop_into(-9.81 / 1 * np.sin(th))

    vip.solve(10, time_step=0.1)
    plt.plot(th.t, th.values)
    plt.show()


def test_source():
    vip.clear()
    u = vip.create_source(lambda t: 5 * np.sin(5 * t))
    dd_th = vip.loop_node(u)
    d_th = vip.integrate(dd_th, 0)
    th = vip.integrate(d_th, np.pi / 2)
    dd_th.loop_into(-9.81 / 1 * np.sin(th))
    vip.solve(10)


def test_loop():
    vip.clear()
    acc = vip.loop_node(0.1)
    vit = vip.integrate(acc, 0)
    pos = vip.integrate(vit, 5)
    acc.loop_into(1 / 10 * (-1 * vit - 1 * pos))
    acc.loop_into(5)
    vip.solve(50)


def test_integrate_scalar():
    vip.clear()
    x = vip.integrate(5, 1)
    vip.solve(10)


def test_plant_controller():
    vip.clear()
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
        acc = vip.loop_node(1 / m * x)
        vit = vip.integrate(acc, v0)
        pos = vip.integrate(vit, x0)
        acc.loop_into(1 / m * (-c * vit - k * pos + x))
        return pos

    target = 1
    error = vip.loop_node(target)
    x = controller(error)
    y = plant(x)
    error.loop_into(-y)

    vip.solve(50)


def test_mass_spring_bond_graph():
    vip.clear()
    def inertia(forces: Sequence[vip.TemporalVar], mass: float):
        acc = np.sum(forces) / mass + 9.81
        vit = vip.integrate(acc, 0)
        return vit

    def spring(speed1, speed2, stiffness: float):
        x = vip.integrate(speed1 - speed2, 0)
        force2 = k * x
        force1 = -force2
        return force1, force2

    k = 1
    mass = 1
    speed1 = vip.loop_node(0)
    force1, force2 = spring(speed1, 0, k)
    vit = inertia((force1,), mass)
    speed1.loop_into(vit)

    vip.solve(50)
