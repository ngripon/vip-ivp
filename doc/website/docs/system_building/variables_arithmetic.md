---
sidebar_position: 2.5
---

# Arithmetic and Manipulation

Temporal Variables in `vip-ivp` are designed to behave like regular numerical variables. This allows users to define systems naturally and expressively. They support a wide range of operations out of the box.

## Arithmetic Support

The `TemporalVar` class implements all standard arithmetic operations:

- Addition: `x + y`
- Subtraction: `x - y`
- Multiplication: `x * y`
- Division: `x / y`
- Floor Division: `x // y`
- Remainder (modulo) : `x % y`
- Power: `x ** 2`
- Negation: `-x`

These operations return new `TemporalVar` instances that represent the result of the operation over time.

```python
z = x + 2 * y
```

:::tip
They also work with Temporal Variables containing NumPy arrays !
:::
:::warning
However, arithmetic does not work with Temporal Variables containing a dictionary.
:::

## Comparison Operator Support

Temporal variables also support comparison operators:

- Equal: `x == y`
- Not equal: `x != y`
- Greater than: `x > y`
- Less than: `x < y`
- Greater than or equal to: `x >= y`
- Less than or equal to: `x <= y`

These operations do not return booleans; instead, they return new TemporalVar instances representing the condition's value over time.

```python
condition = x > threshold
```

## Item Getting

If a Temporal Variable contains a collection, its elements can be accessed using standard indexing. This returns a new `TemporalVar` instance that represents the value of the selected element over time.

```python
arr_child = arr[0]
dict_child = d["a"]
```

## NumPy UFunc Support

`vip-ivp` supports NumPy universal functions (ufuncs). You can use NumPy functions like `np.sin`, `np.exp`, `np.abs`, etc., directly on `TemporalVar` instances.

```python
z = np.sin(x) + np.exp(y)
```

These expressions return new `TemporalVar` instances.

[List of available NumPy ufuncs](https://numpy.org/doc/stable/reference/ufuncs.html)
