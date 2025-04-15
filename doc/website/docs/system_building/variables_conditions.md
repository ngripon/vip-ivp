---
sidebar_position: 5
---

# Conditional Statements

## How to create a conditional statement

Avoid using Python `if` statements to define `TemporalVar` instances. Instead, use `vip.where(condition, a, b)`.

The expression  `variable = vip.where(condition, a, b)` means that the value of `variable` is `a` at times when the condition is `True`, and `b` otherwise.

## Why `if` doesn't work

`TemporalVar` instances don't behave correctly with regular `if` statements. Using `if` will evaluate the entire `TemporalVar` object as a boolean, which ignores its temporal nature and always returns `True`.

To illustrate the issue, let's say we want to create the following step function:

$$
f(t) =
\begin{cases}
0 & \text{if } t < 5 \\
1 & \text{if } t \geq 5
\end{cases}
$$

### ❌ Don't do this

:::danger
This code does **not** work:

```python
time = vip.create_source(lambda t: t)
if time < 5:
    step = vip.create_source(0)
else:
    step = vip.create_source(1)

vip.solve(10, time_step=1)

print(step.values)
```

It prints:

```
[0 0 0 0 0 0 0 0 0 0 0]
```

The variable stays at `0` for the entire simulation, silently failing.  
This happens because only one execution path runs in an `if` block, and `vip.create_source(1)` is never called.

You need a way to define both outcomes at once, not conditionally.
:::

### ✅ Do this instead

:::tip
Use `vip.where()` to define the conditional behavior in a single expression:

```python
time = vip.create_source(lambda t: t)
step = vip.where(time < 5, 0, 1)

step.to_plot()

vip.solve(10, time_step=0.01)
```

![Step plot](../images/step.png)

:::
