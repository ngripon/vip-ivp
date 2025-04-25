---
sidebar_position: 6
---

# Solving

After creating your Temporal Variables, you can simulate your system using:

```python
vip.solve(t_end)
```

This will integrate the system from time `t = 0` to `t = t_end`.

## Choosing Time Points

The `vip.solve()` function supports different ways to define the time points of the simulation:

- **Fixed time step** (default):  
  Specify the `time_step` argument to produce outputs at regular intervals.
- **Automatic time points**:  
  Leave `time_step=None` to output only the time points that the solver chooses adaptively.
- **Custom time points**:  
  Provide a list of time values using the `t_eval` argument.

If your system includes events, you can disable automatic insertion of event time points by setting:

```python
include_events_times=False
```

## Choosing a Solving Method

You can choose among several integration methods provided by SciPy by setting the `method` argument:

| Method                 | When to Use                                             | Notes                                                                                         |
| ---------------------- | ------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| **`"RK45"`** (default) | General-purpose for smooth, non-stiff problems          | Fast, accurate, and a good default choice.                                                    |
| **`"RK23"`**           | Same as RK45, but with lower-order accuracy             | Takes more steps, useful if you need dense sampling.                                          |
| **`"DOP853"`**         | For **very smooth** problems requiring high precision   | Higher-order method. Slower but more accurate for some problems.                              |
| **`"Radau"`**          | For **stiff** problems                                  | Implicit solver, stable for stiff systems but slower.                                         |
| **`"BDF"`**            | Also for **stiff** systems, especially long simulations | Multi-step implicit solver. Efficient for large or slowly evolving systems.                   |
| **`"LSODA"`**          | Unsure whether the system is stiff or not?              | Automatically switches between stiff and non-stiff solvers. Great for mixed or unknown cases. |

👉 See [SciPy’s `solve_ivp()` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) for technical details on each method.

## Controlling Precision

Solvers approximate the true solution. You can control the precision using the following parameters:

- `rtol`: Relative tolerance (default: `1e-3`)
- `atol`: Absolute tolerance (default: `1e-6`)

The solver ensures the local error stays below:

```
atol + rtol * abs(y)
```

You can also set the `max_step` argument to limit how far the solver can go in one integration step, which can improve precision.

:::note
**Difference between `max_step` and `time_step`:**

- `max_step` controls the maximum internal step size used by the solver.
- `time_step` defines how often the solution is output. It does not modify how the integration method works. However, events are evaluated at each time step, so a smaller time step may detect higher frequency events.

:::

## Other Options

- `plot=False`:  
  Disables the automatic plotting of variables marked with `.to_plot()`.

- `verbose=True`:  
  Prints useful information about the solver, performance, and triggered events.
