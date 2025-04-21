---
sidebar_position: 7
---

# Post-Processing

Once the system is solved, Temporal Variables gains numerical values. `vip-ivp` provide some utilities for getting the results ready for post-processing.

## Accessing values of variables

**The numerical data of a solved Temporal Variable is a NumPy array of the evolution of its value with time**.

To get the array of the evolution of values, use the `.values` property of `TemporalVar` instance.  
To get the time vector associated with those values, use the `.t` property.

```python
import matplotlib.pyplot as plt
import vip_ivp as vip

# Exponential decay : dN/dt = - λ * N
d_n = vip.loop_node()
n = vip.integrate(d_n, 1)
d_n.loop_into(-0.5 * n)
vip.solve(10, time_step=0.001)

# highlight-start
# Manual plotting
plt.plot(n.t, n.values)
plt.xlabel("Time (s)")
plt.ylabel("Evolution (-)")
plt.show()
# highlight-end
```

## Export to DataFrame
To export 

## Export to CSV file
