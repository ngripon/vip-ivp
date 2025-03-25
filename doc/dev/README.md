# Design choices documentation for developers

## Initial Value Problem

The package `vip-ivp` is aimed at solving IVP for complex systems.

### Mathematical definition

The mathematical definition of an IVP is an ODE with initial conditions at $T=0$. To solve an IVP, we need the following
elements :

1. A function $\frac{dy}{dt}(t,y)$
2. A vector $x_0$ of initial conditions for $y$
3. A integration schema to get $y$ values over time

### How the package helps

The package allow to build the system iteratively, operation by operation. The benefits are mainly gained from how the
integration is handled. When `vip.integrate(var, x0)` is called:

- An element is added into the vector $y$
- The function in `var`, which corresponds to the added element, is added in $\frac{dy}{dt}(t,y)$
- `x0` is added to the vector of initial conditions $x_0$

By tracking these information, the package has all it needs to solve the IVP when the user call `vip.solve()`.