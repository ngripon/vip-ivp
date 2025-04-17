---
sidebar_position: 2
---

# Source Variables

To create Temporal Variables from standard Python values, use the `temporal()` function.

```python
source_variable = vip.temporal(5)
```

## Scalar values

Temporal Variables can be created from all native Python scalar types:

```python
int_variable = vip.temporal(1)
float_variable = vip.temporal(3.14)
bool_variable = vip.temporal(True)
string_variable = vip.temporal("Hello world!")
```

:::note
While `str` is technically a sequence type in Python, it behaves like a scalar in `vip-ivp` operations and transformations.
:::

## Temporal functions

Temporal Variables can also be created from a temporal function. Input a single argument function into `temporal()`:

```python
import numpy as np

def sinus(t):
    return np.sin(t)

temporal_variable = vip.temporal(sinus)
lambda_variable = vip.temporal(lambda t: 2 * t)
```

:::warning
The temporal functions must handle both scalar floats and NumPy arrays as its t argument. For example:

```python
# Good implementation that works with both scalars and arrays
def good_function(t):
    return np.sin(2 * np.pi * t)

# Problematic implementation that only works with scalars
def bad_function(t):
    if t > 0.5:  # This would fail for array inputs
        return 1.0
    else:
        return 0.0
```

:::

## Collections

### NumPy arrays

A Temporal Variable can contain a NumPy ndarray. Lists are automatically converted to NumPy arrays when passed to `temporal()`.

```python
import numpy as np

arr = np.arange(6).reshape(2, 3)
list_input = [[0, 1, 2], [3, 4, 5]]
list_of_mixed = [lambda t: t, "Hello", 5, True]

array_variable = vip.temporal(arr)
array_variable_too = vip.temporal(list_input) # Automatically converted to np.array
array_mixed = vip.temporal(list_of_mixed)
```

:::warning
Nested list inputs must represent a valid rectangular matrix.

The following example triggers a ValueError:

```python
list_input = [[0, 1], [2]]

# Error: The requested array has an inhomogeneous shape after 1 dimensions.
array_variable = vip.temporal(list_input)
```

:::

### Dictionaries

A Temporal Variable can also contain a Python `dict`:

```python
d = {
    "fun": lambda t: t,
    "number": 49.3,
    "text": "Hello world!"
}

dict_variable = vip.temporal(d)
```

## Scenario tables

A Scenario table represents time-varying values as a mapping between time points and corresponding values A Temporal Variable can be created from a scenario in various formats: `dict`, `pandas.DataFrame`, CSV-file or JSON-file.

Values between specified time points are interpolated according to the specified `interpolation_kind`. Supported interpolation types include "linear", "nearest", "nearest-up", "zero", "slinear", "quadratic", "cubic", "previous", or "next". The default is "linear".

```python
scenario_variable = vip.create_scenario(scenario, time_key="t", interpolation_kind="linear")
```

:::note
When the simulation time exceeds the maximum time value in the scenario, the scenario variable stays at the last defined value.
:::

### From a dictionary

```python
scenario_dict = {"t": [0, 1, 2, 3, 4], "a": [1, 2, 3, 4, 5], "b": [0, 10, -10, 10, -10]}
scenario_variable = vip.create_scenario(scenario_dict, time_key="t")
```

### From a DataFrame

```python
scenario_dict = {"t": [0, 1, 2, 3, 4], "a": [1, 2, 3, 4, 5], "b": [0, 10, -10, 10, -10]}
scenario_df = pd.DataFrame(scenario_dict)
scenario_variable = vip.create_scenario(scenario_df, time_key="t")
```

### From a CSV file

When importing from a CSV file, specify the separator with the `sep` argument.

```csv title="scenario.csv"
t;a;b
0;1;0
1;2;10
2;3;-10
3;4;10
4;5;-10
```

```python
scenario_variable = vip.create_scenario("scenario.csv", time_key="t", sep=";")
```

### From a JSON file

```json title="scenario.json"
{
  "t": [0, 1, 2, 3, 4],
  "a": [1, 2, 3, 4, 5],
  "b": [0, 10, -10, 10, -10]
}
```

```python
scenario_variable = vip.create_scenario("scenario.json", time_key="t")
```
