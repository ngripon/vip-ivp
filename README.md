# vip-ivp

A modular approach to solve complex dynamical systems with a Pythonic style.

It is a
lightweight [alternative to Simulink and Modelica](#for-those-looking-for-an-alternative-to-simulink-or-modelica) that
integrates with your favorites Python libraries.

**Full documentation:** [vip-ivp.org](https://vip-ivp.org)

## Features

- **Incremental modeling** – Build complex systems step by step, using clean and modular code.
- **Python-native** – Fully compatible with NumPy, SciPy, PyTorch, OpenCV, and more.
- **Event system** – Define and respond to time-based or conditional events during simulation.

## Installation

```
pip install vip-ivp
```

## Minimal example: Exponential Decay

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

The traditional way to solve an **Initial Value Problem (IVP)** is to define the function  $y'=f(t,y(t))$  and pass it
into a solver, such as `scipy.integrate.solve_ivp()`.

However, this approach becomes **cumbersome and error-prone** for complex systems, as both $f$ and $y$ grow in size and
complexity. This is why industries rely on tools like **Simulink** and **OpenModelica**, which provide a more intuitive,
graphical approach for handling large IVP systems.

This package brings **key abstractions from graphical IVP solvers** into a scripting environment, enabling a **more
natural and modular** way to define differential equations

## Roadmap

- **Discrete temporal variables** – Currently, all variables are continuous. Support for discrete updates is planned.
- **User interface for analysis** – While scripting is ideal for building models, a graphical UI will help with
  debugging and exploring results.
- **State machine support** – Integrate state-based logic for modeling hybrid and conditional systems.
- **Multiphysics via bond graphs** – Introduce energy-based modeling for complex physical systems using a Bond Graph
  formalism.

## For Those Looking for an Alternative to Simulink or Modelica

### ✅ What `vip-ivp` can do

- Model and solve **initial value problems (IVPs)** with modular, composable Python code.
- Define **conditional events** and time-based triggers within your models.
- Build systems incrementally, keeping code clean and easy to understand.
- Integrate smoothly with popular Python libraries like NumPy, SciPy, PyTorch, and Matplotlib.
- Visualize and analyze results directly from your code.

### 🕒 What `vip-ivp` cannot do yet

- Support **discrete-time variables** or hybrid discrete/continuous systems.
- Provide built-in **state machines** or transition logic.
- Offer native support for **multiphysics modeling**.

These features are planned for future releases.

### 🚫 What `vip-ivp` will not do

- Generate code for embedded or production systems because Python has inherent limitations in that area.
- Shift to a GUI-first workflow — we think that scripting is the best way to create a system.

## License

MIT