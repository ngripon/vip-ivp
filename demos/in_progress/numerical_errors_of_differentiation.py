"""
This example show how the use of .derivative() can introduce uncontrolled numerical error.
"""

import vip_ivp as vip

step = vip.temporal(lambda t: 0 if t < 1 else 1)
# Integrate then differentiate → Just a slight delay
i_step = vip.state(0, derivative=step)
step_ok = i_step.compute_derivative(0.001)

# Differentiate then integrate → Huge error !!!
d_step_bad = step.compute_derivative(0.001)
step_bad = vip.state( 0, derivative=d_step_bad)

vip.solve(2)

vip.plot(step, step_ok, step_bad)
