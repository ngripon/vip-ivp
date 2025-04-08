---
sidebar_position: 1
---

# Overview



**vip-ivp** is a Python package that allows you to **create and simulate dynamical systems** with an elegant, script-based approach. It is a free, code-driven alternative to Simulink—each block becomes a line of code, enabling you to build systems rapidly.


## Quick example

Here is a simple example of a bouncing ball simulation with event handling:

```python title="bouncing_ball.py"
import vip_ivp as vip

# Parameters
initial_height = 1  # m
GRAVITY = -9.81
k = 0.7  # Bouncing coefficient
v_min = 0.01  # Minimum velocity need to bounce

# Create the system
acceleration = vip.create_source(GRAVITY)
velocity = vip.integrate(acceleration, x0=0)
height = vip.integrate(velocity, x0=initial_height)

# Define bounce action: reverse velocity if it's large enough, else stop
bounce = vip.where(abs(velocity) > v_min, velocity.set_value(-k * velocity), vip.terminate)
# Define the event that triggers bounce when height crosses 0 downward (falling)
height.on_crossing(0, bounce, terminal=False, direction="falling")

# Add variables to the plot
height.to_plot("Height (m)")

# Solve the system
vip.solve(20, time_step=0.001)
```

The following plot opens upon solving:

![Boucing ball plot](./images/bouncing_ball.png)

## Features

- **Awesome Developer Experience**: Create dynamic systems in just a few lines of code using a familiar arithmetic syntax.
- **Python Integration**: Seamlessly use it with your favorite Python libraries and tools, like SciPy, PyTorch, Plotly, and more.
- **Hybrid Solver**: Handle both differential equations and events.
- **Causal Approach**: Easy to understand and debug, thanks to a clear direction of data flow.
- **Free and Open Source**: Open for academic, research, or commercial use at no cost.

## Comparison with other tools

### MATLAB/Simulink

Simulink is powerful, but expensive — and its toolbox system makes it even pricier, harder to share, and more restrictive. Building models with blocks is often slow and awkward, especially for simple math formulas that would take just one line in `vip-ivp`. The graphical format also makes it hard to use good development practices like version control, unit testing, or documentation. 

With `vip-ivp`, everything is done in clean, readable Python code — fast to write, easy to share, and simple to maintain.

### SciPy

SciPy provides `scipy.integrate.solve_ivp`, which is very effective for solving systems of ODEs defined by a single function `dy/dt = f(y, t)`. The problem is that this function is cumbersome to write even for moderately complex systems. 

SciPy’s approach is low-level and not well suited for building modular, structured models involving many interconnected components.  `vip-ivp` fills this gap by allowing users to build modular systems incrementally — more like how you'd structure a block diagram, but in code. It’s ideal for building systems that are too structured or too complex to express cleanly in a monolithic function.

### Modelica

Modelica defines systems using equations, which is elegant from a mathematical standpoint. However, this acausal approach can make debugging challenging—if the solver fails to assemble the system, pinpointing the problem can be difficult.

In contrast, `vip-ivp` uses a causal approach where the direction and update rules for each variable are explicit. While this approach is less abstract than Modelica’s, it’s easier to follow, simpler to debug, and more reliable in practice.


### SimuPy

SimuPy is a Python package that enables users to build system models using interconnections of blocks, similar to Simulink. It supports symbolic modeling and numerical integration.

While SimuPy offers a graph-based abstraction, it requires users to define dynamics using state-space systems, which can be less intuitive when modeling systems with internal logic or event-driven behavior. `vip-ivp`, on the other hand, is designed to feel more natural to Python users, allowing you to define states, events, and signals directly without the need to work with state-space formalisms unless desired.

