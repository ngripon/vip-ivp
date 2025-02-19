# vip-ivp

Solve ODE using the normal flow of the script, without having to build the system of equations.

## Minimal example

## Motivation

The traditional way to solve an Initial Value Problem (IVP) is to determine the function $y'=f(t,y(t))$ of the system
and pass it into a solver, typically by using `scipy.integrate.solve_ivp()`.

However, this approach is error-prone for solving complex systems, as the function $f$ and the vector $y$ become huge.
That's why the industry rely on tools like Simulink to solve IVP with lots of variables.

The goal of this package is to bring some good abstractions from graphical IVP solvers to a script approach:

- Decoupling the solver and the system.
- Building the system like a flow.
- Representing differential equations as loops.
- Architecture the system with a functional oriented paradigm.

## Features

### Integrate

### Handle integration loops

### Create sources

### Solve the system of equation

### Explore results

### Save and plot intermediary results

### Create a new system

## Limitations

- There is no function to compute derivatives. 
