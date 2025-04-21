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
To convert a set of variables to a pandas DataFrame, use `vip.export_to_df(*variables)`:

```python
# Create the system
acceleration = vip.temporal(9.81)
velocity = vip.integrate(acceleration, x0=0)
height = vip.integrate(velocity, x0=10)

vip.solve(10)

# highlight-start
# Create a dataframe
dataframe = vip.export_to_df(velocity, height)
print(dataframe)
# highlight-end
```

The console prints:
```
     Time (s)  velocity     height
0         0.0     0.000   10.00000
1         0.1     0.981   10.04905
2         0.2     1.962   10.19620
3         0.3     2.943   10.44145
4         0.4     3.924   10.78480
..        ...       ...        ...
96        9.6    94.176  462.04480
97        9.7    95.157  471.51145
98        9.8    96.138  481.07620
99        9.9    97.119  490.73905
100      10.0    98.100  500.50000

[101 rows x 3 columns]
```

## Export to a file

Temporal Variables results can be directed exported to a CSV or JSON file using `vip.export_file(filename, variable_list, file_format)`.

### CSV-file

Using the previous example:
```python
vip.solve(10, time_step=1)  # Use a smaller time step

# highlight-next-line
vip.export_file("result.csv", (velocity, height), "csv")
```

Here is the content of the CSV:
```csv title="result.csv"
Time (s),velocity,height
0.0,0.0,10.0
1.0,9.810000000000008,14.904999999999992
2.0,19.620000000000005,29.620000000000026
3.0,29.430000000000003,54.14500000000009
4.0,39.24000000000001,88.48000000000016
5.0,49.05000000000001,132.62500000000017
6.0,58.86000000000003,186.58000000000018
7.0,68.67000000000003,250.3450000000001
8.0,78.48000000000005,323.9200000000001
9.0,88.29000000000008,407.3050000000002
10.0,98.10000000000008,500.5000000000001
```

### JSON-file

```python
vip.export_file("result.json", (velocity, height), "json")
```

JSON content:

```json title="result.json"
[
  {
    "Time (s)": 0.0,
    "velocity": 0.0,
    "height": 10.0
  },
  {
    "Time (s)": 1.0,
    "velocity": 9.81,
    "height": 14.905
  },
  {
    "Time (s)": 2.0,
    "velocity": 19.62,
    "height": 29.62
  },
  {
    "Time (s)": 3.0,
    "velocity": 29.43,
    "height": 54.145
  },
  {
    "Time (s)": 4.0,
    "velocity": 39.24,
    "height": 88.48
  },
  {
    "Time (s)": 5.0,
    "velocity": 49.05,
    "height": 132.625
  },
  {
    "Time (s)": 6.0,
    "velocity": 58.86,
    "height": 186.58
  },
  {
    "Time (s)": 7.0,
    "velocity": 68.67,
    "height": 250.345
  },
  {
    "Time (s)": 8.0,
    "velocity": 78.48,
    "height": 323.92
  },
  {
    "Time (s)": 9.0,
    "velocity": 88.29,
    "height": 407.305
  },
  {
    "Time (s)": 10.0,
    "velocity": 98.1,
    "height": 500.5
  }
]
```