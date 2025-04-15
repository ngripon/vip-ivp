"""
This example show how the use of .derivative() can introduce uncontrolled numerical error.
"""

import vip_ivp as vip

step = vip.create_source(lambda t: 0 if t < 1 else 1)
# Integrate then differentiate → Just a slight delay
i_step = vip.integrate(step, 0)
step_ok = i_step.derivative()

# Differentiate then integrate → Huge error !!!
d_step_bad = step.derivative()
step_bad = vip.integrate(d_step_bad, 0)

step.to_plot()
step_ok.to_plot()
step_bad.to_plot()

vip.solve(2, time_step=0.01)

print(step_bad.values)
