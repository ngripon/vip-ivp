# Design choices documentation for developers

## Initial Value Problem

The package `vip-ivp` is aimed at solving IVP for complex systems.

### Mathematical definition

The mathematical definition of an IVP is an ODE with initial conditions at $T=0$. To solve an IVP, we need the following
elements :

1. A function $\frac{dy}{dt}(t,y)$
2. A vector $x_0$ of initial conditions for $y$
3. A integration schema to get $y$ values over time

### What the solver do

**The purpose of the solver is to compute $t_s$ and $y_s$**, which corresponds to the time and state vectors of the solved
system.

An integration scheme takes a point $t_i$ and $y_i$ and computes $t_{i+1}$ and $y_{i+i}$. For the first point, $t_0=0$
and $y_0=x_0$.

The integration is prolonged until $t_{end}$ or a terminal event has been reached.

The solver also manages events. An event is a condition that triggers an action. For an IVP system, conditions are
crossing of values. The solver evaluates the exact time at which the crossing happens, and execute the action at this
time.

### How the package helps

The package allow to build the system iteratively, operation by operation. The benefits are mainly gained from how the
integration is handled. When `vip.integrate(var, x0)` is called:

- An element is added into the vector $y$
- The function in `var`, which corresponds to the added element, is added in $\frac{dy}{dt}(t,y)$
- `x0` is added to the vector of initial conditions $x_0$

By tracking these information, the package has all it needs to solve the IVP when the user call `vip.solve()`.

## System definition: a Functional Programming approach

### A system is composed of functions

Each system variable's value depends on time $t$ and system state $y$. It is therefore a function of state and
time : $v=f(t,y)$.

A system is built by composing functions without evaluating them. Each time a variable is transformed, its output must
be another function of $t$ and $y$. For example, $a+b=g(t,y)=f_a(t,y)+f_b(t,y)$.

### Variables are lazy

Variables are function that may be evaluated only at those occasions:

- During the solving, when the solver evaluates $\frac{dy}{dt}(t_i,y_i)$ to compute an integration step.
- After the solving, when the vectors $t_s$ and $y_s$ are available. By using those vectors, we get the temporal
  evolution fo the variable values.

**Before the solving, the variables can not be evaluated.**

### ODEs are circular

A particularity of ODEs is their circularity. Take the exponential decay ODE : $\dot{n}=-\lambda.n$:

- To get $n$, you need to integrate $\dot{n}$
- To compute $\dot{n}$, you need $n$

There is a cycle because both $n$ and $\dot{n}$ use the other as its input.

### Possible solutions for defining a system



