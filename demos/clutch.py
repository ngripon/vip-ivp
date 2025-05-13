import numpy as np
import pandas as pd
import vip_ivp as vip

# Parameters
ie, iv = 1, 5  # Moments of inertia
be, bv = 2, 1  # damping rates
mu_k, mu_s = 1, 1.5  # kinetic and static coefficients of friction
r1, r2 = 1, 1  # radii of plate friction surfaces

# r = (r2 ** 3 - r1 ** 3) / (r2 ** 2 - r1 ** 2)
# Computed quantities
r = 1

locked_w = 1
unlocked_we = vip.temporal(1)
unlocked_wv = vip.temporal(1)


def unlocked_system(t_in, tf_max_k):
    # Create differential variables
    d_we = vip.loop_node()
    we = vip.integrate(d_we, locked_w)
    d_wv = vip.loop_node()
    wv = vip.integrate(d_wv, locked_w)
    # Set values
    t_cl = np.sign(we - wv) * tf_max_k
    d_we.loop_into((t_in - be * we - t_cl) / ie)
    d_wv.loop_into((t_cl - wv * bv) / iv)
    return we, wv


def locked_system(t_in):
    dw = vip.loop_node()
    w = vip.integrate(dw, unlocked_we)
    dw.loop_into((t_in - (be + bv) * w) / (ie + iv))
    return w


def friction_model(f_n):
    tf_max_s = 2 / 3 * r * f_n * mu_s
    tf_max_k = 2 / 3 * r * f_n * mu_k
    return tf_max_k, tf_max_s


def friction_mode_logic(t_in, tf_max_s):
    t_f = compute_friction_torque(t_in)
    torque_condition = np.abs(t_f) <= tf_max_s
    velocities_match = unlocked_we.crosses(unlocked_wv)
    return velocities_match & torque_condition


def compute_friction_torque(t_in):
    t_f = unlocked_we * bv + iv / (ie + iv) * (t_in - unlocked_we * (bv + be))
    return t_f


fn_map = np.array([
    [0, 0],
    [1, 0.8],
    [2, 1.6],
    [3, 1.6],
    [4, 1.6],
    [5, 1.6],
    [6, 0.8],
    [7, 0],
    [8, 0],
    [9, 0],
    [10, 0]
])

tn_map = np.array([
    [1, 2],
    [2, 2],
    [3, 2],
    [4, 2],
    [5, 2],
    [6, 0],
    [7, 0],
    [8, 0],
    [9, 0],
    [10, 0]
])


fn=vip.create_scenario(pd.DataFrame(fn_map),0)[1]
tn=vip.create_scenario(pd.DataFrame(tn_map),0)[1]


fn.to_plot()
tn.to_plot()

vip.solve(10)
