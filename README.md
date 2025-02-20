# vip-ivp

Solve ODEs using the flow of the script, without having to build the system of equations.

## Minimal example

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
integrated_var = integrate(source, x0=0)
```

### Handle integration loops

Create loop nodes to handle feedback loops in the system.

```python
loop = loop_node(input_value=0)
loop.loop_into(integrated_var)
```

### Create sources

Create source signals from temporal functions or scalar values.

```python
source = create_source(lambda t: 2 * t)
```

### Solve the system of equations

Solve the system until a specified end time.

```python
solve(t_end=10)
```

### Explore results

Explore the function over given bounds and solve the system.

```python
explore(lambda x: x ** 2, t_end=10, bounds=(0, 1))
```

### Save and plot intermediary results

Save and plot variables for later analysis.

```python
integrated_var.save("integrated_var")
plot()
```

### Create a new system

Initialize a new solver system.

```python
new_system()
```

## Limitations

- There is no function to compute derivatives.
