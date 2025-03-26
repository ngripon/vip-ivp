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

**The purpose of the solver is to compute $t_s$ and $y_s$**, which corresponds to the time and state vectors of the
solved
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

#### Graph approach

Simulink, the main IVP solver on the market, use a graph approach to solve these problems :

- The system is defined in a graph data structure, which allows to easily define cycles.
- Each node of this graph contains a function. The function of the node is predefined, the user has to choose from a
  library of available nodes called *blocks*.
- When the simulation is launched, a compilation step is executed to build the equation from the graph.
- Once the system is build, the integration scheme is applied.

This approach is well adapted to a visual programming interface. However, defining a graph in a script is more
cumbersome, because the data representation of a graph does not reflect its flow. A functional programming approach is
more adapted.

#### Functional Programming approach

Functional programming languages like Haskell possess much of the properties we need: all variables are functions and
lazy. All the math and freedom of code is available. The flow of the system become the flow of the script.

However, there is no tool specific to Functional Programming to handle the circularity of ODEs. To create loops, we will
use a trick with Object-Oriented Programming : create an empty variable that can be integrated, then set later.

### Motivation of choosing the functional programming approach

The functional programming approach has been chosen for the following reasons:

- **Simple**: It preserves the natural flow of the script in the definition of variables. The boilerplate is minimal.
- **Open**: It allows any function to be used. Therefore, the whole Python ecosystem is compatible with this approach.

## Main Abstractions

### Solver

The responsibilities of the solver are:

- To build the $\frac{dy}{dt}(t,y)$ function and $x_0$ vector by tracking which variables are integrated.
- To manage events
- To solve the system.
- To propose any function relative to the whole system.

### TemporalVar

The `TemporalVar` class is the type of every variable composing the system.

TemporalVar implements all Python arithmetic magic methods. Therefore, algebra behaves as expected.

#### Handling collections

TemporalVar is a recursive class that contains a function $f(t,y)$ OR a collection of itself (dict or ndarray). 

This structure offers the best of the both world for the user : it acts exactly as a collection of `TemporalVar` (useful for
linear algebra with ndarrays) and allow to call specific methods on the whole collection (for example the `values`
property method that get the collection of results).

### LoopNode

The `LoopNode(TemporalVar)` class allows to define a loop. It is an empty variable that can be integrated, then set
later.

## Package structure

The package is structured as follows:

- `base.py` : Contains the base abstractions: `Solver`, `TemporalVar` and `LoopNode`. These abstractions should offer
  the most freedom possible.
- `api.py` : Contains functions to build the system that are user-friendly and opinionated.
- `__init__.py`: Control the interface of the package. Export only the functions of the `api.py` file.
- `utils.py`: Utility functions

## User abstractions
- Source: transform scalar, temporal functions and collections of them into a `TemporalVar` objects.
- Scenario: Transform a map into an interpolated temporal function and put it into a `TemporalVar`.
- `where`: Allow if-else conditions within a `TemporalVar`.
- `f()`: Transform a function into another function that is compatible with `TemporalVar` inputs. 

