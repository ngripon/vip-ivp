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

- **Awesome Developer Experience**: Create dynamic systems in just a few lines of code using the familiar Python syntax.
- **Python Integration**: Seamlessly use it with your favorite Python libraries and tools, like SciPy, PyTorch, Plotly, Streamlit and more.
- **Hybrid Solver**: Handle both differential equations and events.
- **Causal Approach**: Easy to understand and debug, thanks to a clear direction of data flow.
- **Free and Open Source**: Open for academic, research, or commercial use at no cost.

## Comparison with other tools

### MATLAB/Simulink

Simulink is one of the most widely used tools for modeling and simulating dynamic systems in industry, research, and education. Its visual interface and integration with MATLAB make it a strong choice for many engineering workflows.

However, for developers and researchers who value code clarity, open tools, and modern software practices, Simulink can introduce a few challenges:

- Frequent breaking changes: Simulink updates may introduce compatibility issues from one yearly version to another, requiring extra effort to maintain long-term models.
- Closed ecosystem: Limits flexibility compared to Python’s vast open-source ecosystem.
- Toolbox licensing model: Many features are locked behind paid toolboxes. This can make collaboration difficult if others don’t have the same toolboxes installed.
- Slow model building: Building models block-by-block is effective for visualization, but often inefficient for expressing simple mathematical relationships that would take just one line in `vip-ivp`.
- Limited support for modern software development:
  - Not easy to read or navigate in complex systems
  - Hard to unit test components
  - Difficult to version control using Git
  - Limited support for inline documentation
  - No workflow for AI-assisted development

`vip-ivp` offers a code-first alternative built for modern developers and scientists:

- Fully **open-source** and built in Python
- Seamless integration with libraries like NumPy, SciPy, and PyTorch
- Easy to read, test, share, and version control
- Fast to prototype, extend, and maintain

If you're looking for a lightweight, developer-friendly way to simulate dynamic systems — without leaving your Python environment — `vip-ivp` might be exactly what you need.

### SciPy

SciPy provides `scipy.integrate.solve_ivp`, which is very effective for solving systems of ODEs defined by a single function `dy/dt = f(y, t)`. The problem is that this function is cumbersome to write even for moderately complex systems.

SciPy’s approach is low-level and not well suited for building modular, structured models involving many interconnected components. `vip-ivp` fills this gap by allowing users to build modular systems incrementally — more like how you'd structure a block diagram, but in code. It’s ideal for building systems that are too structured or too complex to express cleanly in a monolithic function.

### Modelica

Modelica defines systems using equations, which is elegant from a mathematical standpoint. However, this acausal approach can make debugging challenging—if the solver fails to assemble the system, pinpointing the problem can be difficult.

In contrast, `vip-ivp` uses a causal approach where the direction and update rules for each variable are explicit. While this approach is less abstract than Modelica’s, it’s easier to follow, simpler to debug, and more reliable in practice.

### SimuPy

SimuPy is a Python package that enables users to build system models using interconnections of blocks, similar to Simulink. It supports symbolic modeling and numerical integration.

While SimuPy offers a graph-based abstraction, it requires users to define dynamics using state-space systems, which can be less intuitive when modeling systems with internal logic or event-driven behavior. `vip-ivp`, on the other hand, is designed to feel more natural to Python users, allowing you to define states, events, and signals directly without the need to work with state-space formalisms unless desired.
