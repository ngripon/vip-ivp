---
sidebar_position: 8
---

# Post-Processing

Once the system is solved, Temporal Variables contain numerical data. `vip-ivp` provides several utilities to access and export this data for analysis and visualization.

## Accessing the Values of Variables

**After solving, a `TemporalVar` instance has two key properties:**

- `.values`: a NumPy array containing the evolution of the variable's value over time
- `.t`: a NumPy array of the corresponding time values

```python
import matplotlib.pyplot as plt
import vip_ivp as vip

# Exponential decay: dN/dt = -λ * N
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

To convert one or more `TemporalVar` instances into a pandas `DataFrame`, use:

```python
vip.export_to_df(*variables)
```

Example:

```python
# Create the system
acceleration = vip.temporal(-9.81)
velocity = vip.integrate(acceleration, x0=0)
height = vip.integrate(velocity, x0=10)

vip.solve(10, time_step=1)

# highlight-start
# Convert to a pandas DataFrame
dataframe = vip.export_to_df(velocity, height)
print(dataframe)
# highlight-end
```

Console output:
```
    Time (s)  velocity   height
0        0.0      0.00   10.000
1        1.0     -9.81    5.095
2        2.0    -19.62   -9.620
...
10      10.0    -98.10 -480.500
```

## Export to a File

You can export the results to a CSV or JSON file using:

```python
vip.export_file(filename, variable_list, file_format)
```

- `filename`: path to the output file
- `variable_list`: a tuple of `TemporalVar` instances
- `file_format`: either `"csv"` or `"json"`

### Export to CSV

```python
vip.export_file("result.csv", (velocity, height), "csv")
```

**Content of `result.csv`:**

```csv title="result.csv"
Time (s),velocity,height
0.0,0.0,10.0
1.0,-9.81,5.09
2.0,-19.62,-9.62
...
10.0,-98.10,-480.50
```

### Export to JSON

```python
vip.export_file("result.json", (velocity, height), "json")
```

**Content of `result.json`:**

```json title="result.json"
[
  {
    "Time (s)": 0.0,
    "velocity": 0.0,
    "height": 10.0
  },
  {
    "Time (s)": 1.0,
    "velocity": -9.81,
    "height": 5.095
  },
  {
    "Time (s)": 2.0,
    "velocity": -19.62,
    "height": -9.62
  },
...
  {
    "Time (s)": 10.0,
    "velocity": -98.1,
    "height": -480.5
  }
]

```
