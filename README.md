# vip-ivp

Solve ODEs naturally within the script's flow—no need to manually construct a system of equations.

## Minimal example

```python
import vip_ivp as vip

# Exponential decay: dN/dt = - λ * N
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

The traditional way to solve an **Initial Value Problem (IVP)** is to define the function  $y'=f(t,y(t))$  and pass it
into a solver, such as `scipy.integrate.solve_ivp()`.

However, this approach becomes **cumbersome and error-prone** for complex systems, as both $f$ and $y$ grow in size and
complexity. This is why industries rely on tools like **Simulink**, which provide a more intuitive, graphical approach
for handling large IVP systems.

This package brings **key abstractions from graphical IVP solvers** into a scripting environment, enabling a **more
natural and modular** way to define differential equations:

- **Decouples** the solver from the system definition.
- **Builds the system incrementally**, following the script’s natural flow.
- **Represents differential equations as loops**, improving clarity.
- **Encourages a functional programming paradigm** for better system architecture.
- **Seamlessly integrates with the Python ecosystem**, working alongside libraries like NumPy and SciPy.

## Demo: Mass-spring-damper model

```python
import vip_ivp as vip

# System parameters
m = 300.0  # Mass (kg)
c = 1500  # Damping coefficient (N.s/m)
k = 25000  # Spring stiffness (N/m)
displacement_x0 = 0.2  # Initial value of displacement (m)

# Create simulation
# System equation is: m * acc + c * vel + k * disp = 0 <=> acc = - 1 / m * (c * vel + k * disp)
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

To solve an ODE, follow these steps:

1. Create a loop node for the highest-order derivative of the equation:

```python
ddy = vip.loop_node()
```

2. Create lower-order derivatives by integration

```python
dy = vip.integrate(ddy, dy0)
y = vip.integrate(dy, y0)
```

3. Loop into the equation (In this example: $4 \frac{d^2y}{dt^2} + 3 \frac{dy}{dt} + 2y = 5$):

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

### Plot results

Plot variables with `.to_plot(variable_name : str)` method.

The plot is automatically created when the system is solved.

```python
def foo():
    variable = vip.create_source(5)
    variable.to_plot("Variable name")


foo()
vip.solve(10)  # 'variable' will be plotted, even if it was declared in a function.
```

### Explore results

Generate an interactive plot from a given function. The plot includes sliders, allowing users to adjust input values dynamically.

This feature requires the [sliderplot](https://github.com/ngripon/sliderplot) package:

`pip install sliderplot`

```python
def mass_spring_damper_system(m=1, c=1, k=1, x0=0.2):
    acceleration = vip.loop_node()
    velocity = vip.integrate(acceleration, 0)
    displacement = vip.integrate(velocity, x0)
    acceleration.loop_into(-(c * velocity + k * displacement) / m)
    return displacement


t_simulation = 50  # seconds
time_step = 0.001  # seconds
vip.explore(mass_spring_damper_system, t_simulation, time_step=time_step, title="Mass-Spring-Damper mechanism")
```

## Advanced features

### Save intermediary results

Save variables for later analysis.

Its only use-case is when the variable may be lost due to context, typically for variables that are created inside
functions.

```python
def foo():
    variable = vip.create_source(5)
    variable.save("bar")


foo()
bar = vip.get_var("bar")
vip.solve(10)  # 'variable' will be plotted, even if it was declared in a function.
```

### Create a new system

Initialize a new system.

If you want to simulate multiple systems in the same script, use this function. Otherwise, the previous systems will be
solved again with the new one, which will be slower.

```python
vip.new_system()
```

## Limitations

- Temporal variables can only access their values at time $t$.
- Therefore, there is no function to compute derivatives.
