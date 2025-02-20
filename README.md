# vip-ivp

Solve ODEs using the flow of the script, without having to build the system of equations.

## Minimal example

```python
import vip_ivp as vip

# Exponential decay : dN/dt = - λ * N
d_n = vip.loop_node()
n = vip.integrate(d_n, 1)
d_n.loop_into(-0.5 * n)

# Choose which variables to plot
n.to_plot("Quantity")
d_n.to_plot("Derivative")

# Solve the system. The plot will automatically show.
vip.solve(10, time_step=0.001)
```

## Motivation

The traditional way to solve an Initial Value Problem (IVP) is to determine the function $y'=f(t,y(t))$ of the system
and pass it into a solver, typically by using `scipy.integrate.solve_ivp()`.

However, this approach is error-prone for solving complex systems, as the function $f$ and the vector $y$ become huge.
That's why the industry relies on tools like Simulink to solve IVP with lots of variables.

The goal of this package is to bring some good abstractions from graphical IVP solvers to a script approach:

- Decoupling the solver and the system.
- Building the system like a flow.
- Representing differential equations as loops.
- Architecting the system with a functional-oriented paradigm.

## Demo: Mass-spring-damper model

```python
import vip_ivp as vip

# System parameters
m = 300.0  # Mass (kg)
c = 1500  # Damping coefficient (N.s/m)
k = 25000  # Spring stiffness (N/m)
displacement_x0 = 0.2  # Initial value of displacement (m)

# Create simulation
# System equation is : m * acc + c * vel + k * disp = 0 <=> acc = - 1 / m * (c * vel + k * disp)
# We do not have access to velocity and displacement at this stage, so we create a loop node.
acceleration = vip.loop_node()
velocity = vip.integrate(acceleration, 0)
displacement = vip.integrate(velocity, displacement_x0)
# Now we can set the acceleration
acceleration.loop_into(-(c * velocity + k * displacement) / m)

# Choose results to plot
displacement.to_plot("Displacement (m)")
velocity.to_plot("Velocity (m/s)")

# Solve the system
t_simulation = 10  # s
time_step = 0.001
vip.solve(t_simulation, time_step=time_step)
```

## Features

### Integrate

Integrate a temporal variable starting from an initial condition.

```python
integrated_var = vip.integrate(source, x0=0)
```

### Handle integration loops

Create loop nodes to handle feedback loops in the system.

```python
loop = vip.loop_node(input_value=0)
loop.loop_into(integrated_var)
```

Loop nodes are essential to solve ODEs in a "sequential" way.

To solve an ODE, follow those steps:

1. Create a loop node for the most derived variable :

```python 
ddy = vip.loop_node()
```

2. Create the other variables by integration :

```python
dy = vip.integrate(ddy, dy0)
y = vip.integrate(dy, y0)
```

3. Loop into the equation (In this example : $4 \frac{d^2y}{dt^2} + 3 \frac{dy}{dt} + 2y = 5$) :

```python
ddy.loop_into(5 - 1 / 4 * (3 * dy + 2 * y))
```

### Create sources

Create source signals from temporal functions or scalar values.

```python
source = vip.create_source(lambda t: 2 * t)
```

### Solve the system of equations

Solve the system until a specified end time.

```python
vip.solve(t_end=10,
          method="RK45",
          time_step=None,
          t_eval=None,
          plot=True,
          **options)
```

For `**options`, see
the [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).

### Explore results

Explore the function over given bounds and solve the system.

```python
vip.explore(lambda x: x ** 2, t_end=10, bounds=(0, 1))
```

## Advanced features

### Save and plot intermediary results

Save and plot variables for later analysis.

Its only use-case is when the variable may be lost due to context, typically for variables that are created inside
functions.

```python
def foo():
    variable = vip.create_source(5)
    variable.save("bar")
    variable.to_plot("Variable name")


foo()
bar = vip.get_var("bar")
vip.solve(10)  # 'variable' will be plotted, even if it was declared in a function.

```

### Create a new system

Initialize a new system.

If you want to simulate multiple systems in the same script, use this function. Otherwise, the
previous systems will be solved again with the new one, which will be slower.

```python
vip.new_system()
```

## Limitations

- Temporal variables can only access their values at time $t$.
- Therefore, there is no function to compute derivatives.
