---
sidebar_position: 1
---

# Concepts

In this section, we will cover the main concepts that allow to understand the functionalities of `vip-ivp`.

## Initial Value Problem (IVP)

`vip-ivp` is a tool designed for solving Initial Value Problems (IVPs). You may already be familiar with other tools that deal with IVPs, such as Simulink, Amesim, Modelica, or Dymola.

An IVP is a type of **differential equation** where the solution involves **variables that evolve over time**, **starting from a set of known initial conditions**. These problems are common in various fields of physics and engineering, such as mechanics, electronics, and chemical reactions..

## Temporal Variables

In `vip-ivp`, **Temporal Variables** represent quantities that evolve over time. They are instances of the `TemporalVar` class. **To build a system, you create Temporal Variables**. It is that simple.

Here’s the behavior of Temporal Variables in `vip-ivp`:

1. **Before the system is solved**, Temporal Variables have no value. Attempting to access their value will result in an error.
2. **After the system is solved**, their values over time can be computed and accessed.

Temporal Variables can be created in many ways, including from constants, temporal functions, arithmetic expressions, or integration.

## Solver

The solver in `vip-ivp` is responsible for computing the evolution of variables over time using numerical integration schemes. It ensures that the computed solution meets the required precision.

One of the strengths of `vip-ivp` is the flexibility in how systems are defined. Unlike other solvers such as SciPy's `solve_ivp()`, which impose a strict structure on how systems must be defined, `vip-ivp` allows users to define systems in a more modular and flexible way.

:::warning
An inherent limitation of IVP solvers is that **differentiation is not supported in a "clean" way**. Since `vip-ivp` uses an integration scheme to solve the system, differentiation is not possible without relying on past values.

This implies the following considerations for system design:

- **`vip.integrate()`** should always be preferred over **`vip.differentiate()`**, as integration guarantees the precision of the solution, while differentiation does not.
- **`vip.differentiate()`** may be used for specific cases (e.g., a PID controller), but in those cases, a small time step is recommended to maintain accuracy.
  :::
