---
sidebar_position: 4
---

# Loop Nodes

## How to use

Circularity is an inevitable component of dynamical systems. To model those loops while keeping a causal approach, `vip-ivp` proposes the Loop Node.

The Loop Node is a placeholder variable that can be instantiated, used to create other variables, then set afterward when the necessary variables have been created.

A Loop Node must always be used in the same way:

1. Create the highest order derivative with a Loop Node by calling `vip.loop_node()`.
2. Create the input variables of the Loop Node by using the created loop node.
3. Set the Loop Node value when all the needed are available with the `.loop_into(value)` method of the `LoopNode` instance.

Here is a minimal example with the exponential decay ODE $\frac{dN}{dt} = -\lambda N$ :

```python
lambda_value = 0.5
# 1. Create the highest order derivative with a loop node
d_n = vip.loop_node()
# 2. Create the necessary variables by using the created loop node
n = vip.integrate(d_n, 1)
# 3. Set the loop node value when all the needed variables are available.
d_n.loop_into(-lambda_value * n)

n.to_plot()

vip.solve(10, time_step=0.001)
```
![Exponential decay](../images/exponential_decay.png)

## Loop Nodes containing a NumPy array

Loop Nodes possesses a shape that must be initialized. When calling `vip.loop_node()` without arguments, it creates a scalar Temporal Variable. To instantiate a Loop Node containing a Numpy array, fill the `shape` argument of `vip.loop_node()`:

```python
# Multidimensional exponential decay:
import numpy as np

# Array of lambda coefficients
lambdas = np.linspace(0, 1, 6).reshape(2, 3)
# Building the system
d_n = vip.loop_node(shape=(2, 3))
n = vip.integrate(d_n, np.zeros((2, 3)))
# Loop into an array of shape (2, 3)
d_n.loop_into(n * lambdas)

n.to_plot()

vip.solve(10, time_step=0.001)
```
![Multidimensional exponential decay](../images/exponential_decay_multidim.png)

Loop Nodes containing an array are especially useful for physical quantities modelled by vectors, for example a position in a 2D or 3D space.

### Example: Projectile motion with air drag

```python
import numpy as np

# Parameters
GRAVITY = -9.81
v0 = 20
th0 = np.radians(45)
mu = 0.1  # Coefficient of air drag

# Compute initial condition
v0 = [v0 * np.cos(th0), v0 * np.sin(th0)]
x0 = [0, 0]

# Create system
acceleration = vip.loop_node(2)
velocity = vip.integrate(acceleration, v0)
position = vip.integrate(velocity, x0)
v_norm = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
acceleration.loop_into([-mu * velocity[0] * v_norm,
                        GRAVITY - mu * velocity[1] * v_norm])

vip.solve(10, time_step=0.1)
```

## Securities

Loop Nodes are easy to misuse, hence various securities are implemented to prevent common errors.

### Forcing the user to set the Loop Node value

It is unfortunately way to easy to forget to set the value of a Loop Node. To prevent this common error, an exception is thrown when `vip.solve()` is called if a Loop Node has not been set.

To disable this security, create the Loop Node with argument `strict=False`:

```python
d_n = vip.loop_node(strict=False)
```

The Loop Node will act as a Temporal Variables whose value is `0` or `np.zeros(shape)`.

### Preventing the user from setting the Loop Node multiple times

If the Loop Node has already been set and its `loop_into()` method is called a second time, an exception will be thrown.

To disable this security, call `loop_into()` with argument `force=True`:

```python
d_n = vip.loop_node()
n = vip.integrate(d_n, 1)
d_n.loop_into(-0.2 * n)
d_n.loop_into(-0.3 * n, force=True)
```

The value of the second `.loop_into()` will be added.

### Algebraic loops

Algebraic loops occurs when a variable is its own input. In Python, it causes a `RecursionError`.

Algebraic loops should not exist. If there is a loop, there must be an integration or a delay in it for it not to be an algebraic loop.

Here is an example of an algebraic loop:

```python
d_n = vip.loop_node()
a = 2 * d_n
d_n.loop_into(a)
```

There is an algebraic loop because `d_n = 2 * d_n`, which will cause an infinite recursion.
