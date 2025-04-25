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
acceleration = vip.temporal(GRAVITY)
velocity = vip.integrate(acceleration, x0=0)
height = vip.integrate(velocity, x0=initial_height)

# Define bounce action: reverse velocity if it's large enough, else stop
bounce = vip.where(abs(velocity) > v_min, velocity.action_reset_to(-k * velocity), vip.action_terminate)
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

- **Awesome Developer Experience**: Create dynamic systems of continuous variables in just a few lines of code using a Pythonic syntax.
- **Python Integration**: Seamlessly use it with your favorite Python libraries and tools, like SciPy, PyTorch, Plotly, Streamlit and more.
- **Hybrid Solver**: Handle both differential equations and events.
- **Causal Approach**: Easy to understand and debug, thanks to a clear direction of data flow.
- **Free and Open Source**: Open for academic, research, or commercial use at no cost.

## Comparison with other tools

### MATLAB/Simulink

Simulink is one of the most widely used tools for modeling and simulating dynamic systems in industry, research, and education. Its visual interface, huge number of features and integration with MATLAB make it a strong choice for many engineering workflows.

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

`vip-ivp` is **not a full replacement for Simulink**, but it offers a powerful alternative to its **continuous-time modeling and simulation features**.  
`vip-ivp` offers a code-first alternative built for modern developers and scientists. It brings everything people love about Python to system simulation: easy to read, test, version, and extend — and no license fees.

If you're looking for a lightweight, developer-friendly way to simulate continuous-time dynamic systems — without leaving your Python environment — `vip-ivp` might be exactly what you need.

### SciPy

SciPy provides `scipy.integrate.solve_ivp`, which is very effective for solving systems of ODEs defined by a single function `dy/dt = f(y, t)`. The problem is that this function is cumbersome to write even for moderately complex systems.

SciPy’s approach is low-level and not well suited for building modular, structured models involving many interconnected components. `vip-ivp` fills this gap by allowing users to build modular systems incrementally — more like how you'd structure a block diagram, but in code. It’s ideal for building systems that are too structured or too complex to express cleanly in a monolithic function.

### Modelica

Modelica defines systems using equations, which is elegant from a mathematical standpoint. However, this acausal approach can make debugging challenging—if the solver fails to assemble the system, pinpointing the problem can be difficult.

In contrast, `vip-ivp` uses a causal approach where the direction and update rules for each variable are explicit. While this approach is less abstract than Modelica’s, it’s easier to follow, simpler to debug, and more reliable in practice.

## Roadmap

`vip-ivp` is still in its early stages, and several major features are planned to greatly expand its capabilities:

- **Discrete-time variables** with user-defined sampling rates  
  Enable hybrid modeling by mixing continuous and discrete-time variables.

- **Stateful logic and state machines**  
  Model systems with internal state and mode transitions using intuitive constructs.

- **Interactive user interface** for simulation analysis  
  Explore the behavior of your entire system through a visual interface: inspect variables, trace dependencies, and debug more easily.

- **Multiphysics modeling with Bond Graphs**  
  Introduce components grounded in the Bond Graph formalism for unified modeling of mechanical, electrical, hydraulic, and thermal systems.
