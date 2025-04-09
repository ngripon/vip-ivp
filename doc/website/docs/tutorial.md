---
sidebar_position: 3
---

# Tutorial

In this 5-minute tutorial, we will solve the equation of a mass-spring-damper system using `vip-ivp`. Then, we'll demonstrate the event system by counting the number of oscillations.

## Problem Statement

We consider a classic second-order mechanical system: a **mass** attached to a **spring** and a **damper**. The motion of the mass is governed by the differential equation:

$$
m \ddot{x} + c \dot{x} + kx = 0
$$

Where:

- $x$ is the displacement,
- $m$ is the mass,
- $c$ is the damping coefficient,
- $k$ is the spring constant.

We want to:

1. Simulate the motion of the system over time.
2. Count how many times the mass crosses the equilibrium point (i.e., $x = 0$).

## Step-by-step

### 1. Define the parameters

```python
import vip_ivp as vip

m = 1.0     # Mass (kg)
k = 10.0    # Spring constant (N/m)
c = 0.5     # Damping coefficient (N·s/m)

x0 = 1.0    # Initial position (m)
v0 = 0.0    # Initial velocity (m/s)
```

### 2. Build the system

Ordinary Differential Equations are inherently circular, as higher-order derivatives depend on variables that are themselves computed by integrating those derivatives. To manage this circular dependency, `vip-ivp` introduces the `loop_node()` function. A Loop Node acts as a placeholder for a variable whose definition will be completed later

To solve an ODE, follow these steps:

1. **Create a loop node** for the highest-order derivative of the equation. In our case: the acceleration $\ddot{x}$.
2. **Integrate to obtain lower-order derivatives**. In our case: the velocity $\dot{x}$ and displacement $x$.
3. **Loop into the equation**. In our case, $\ddot{x} =-\frac{1}{m}( c \dot{x} + kx)$.

```python
a = vip.loop_node()  # Acceleration
v = vip.integrate(a, v0)  # Velocity
x = vip.integrate(v, x0)  # Displacement
a.loop_into(-(c * v + k * x) / m) # Set acceleration value
```

### 3. Add an event

```python
# Create a variable to count the number of oscillations
count = vip.create_source(0)
# Create event that triggers when x crosses 0 (from negative to positive)
x.on_crossing(0, count.action_set_to(count + 1), direction="rising")
```

### 4. Plot and solve

```python
# Choose results to plot
x.to_plot("Displacement (m)")
count.to_plot("Oscillations count")
# Solve the system
vip.solve(10, time_step=0.01)
```

### 5. Post-processing

Phase diagram

## Complete example
