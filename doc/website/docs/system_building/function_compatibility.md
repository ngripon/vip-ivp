---
sidebar_position: 6
---

# Applying Functions

## Wrapping Functions with `vip.f()`

`vip-ivp` provides the `vip.f(fun)` function. It creates a wrapper that makes any function compatible with Temporal Variables by:

- Accepting `TemporalVar` inputs
- Returning a `TemporalVar` result


## When Wrapping is not needed

A system is composed of `TemporalVar` instances. Because it is a custom class that do not possess values before the solving, many functions are not compatible out-of-the-box with `TemporalVar` inputs.

When a function is not compatible, it must be wrapped by using `vip.f()`.

Functions are compatible out-of-the-box with `TemporalVar` in only 2 cases:

1. The function is a NumPy `ufunc` ([List of `ufuncs`](https://numpy.org/doc/stable/reference/ufuncs.html))
2. The function accepts and return `TemporalVar` instances. This case happens mostly for the custom function you will build to architecture your systems.

For all the other functions, you will need wrapping.



## Examples

### Example 1: Apply a empirical map with `np.interp()`

Let's create a map for the voltage of battery in function of its state of charge (SoC):

```python
import numpy as np

# Data map
voltage_per_soc = {0.0: 3.0, 0.2: 3.4, 0.5: 3.6, 0.8: 3.8, 1.0: 4.2}
# Create input
soc = vip.create_source(lambda t: np.maximum(1.0 - 0.05 * t, 0))
# Create output with np.interp
voltage = vip.f(np.interp)(soc, list(voltage_per_soc.keys()), list(voltage_per_soc.values()))

voltage.to_plot()
soc.to_plot()

vip.solve(30, time_step=0.01)
```
![Battery map results](../images/battery_map.png)

### Example 2: Use a trained model with PyTorch