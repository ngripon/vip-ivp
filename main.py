import time
from collections.abc import Sequence
from inspect import signature
from numbers import Number
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from variable_exploration import explore


class Solver:
    def __init__(self):
        self.dim = 0
        self.vars = []
        self.feed_vars = []
        self.initialized_vars = []
        self.t = None
        self.y = None
        self.solved = False

    def integrate(self, input_value: "TemporalVar", x0: Number) -> "TemporalVar":
        if isinstance(input_value, TemporalVar):
            self.feed_vars.append(input_value)
            integrated_variable = TemporalVar(self, lambda t, y, idx=self.dim: y[idx], x0)
            self.dim += 1
            return integrated_variable
        else:
            raise Exception("Input value must be a TemporalVar instance created from the solver class. "
                            "Please check the documentation.")
        
    def loop_node(self, input_value)->"LoopNode":
        return LoopNode(self, input_value)

    # def create_derivatives(self, x0: Sequence[Number]) -> list:
    #     try:
    #         n = len(x0)
    #     except TypeError:
    #         n = 1
    #         x0 = (x0,)
    #     var_list = []
    #     var = TemporalVar(self, lambda t, y, idx=self.dim: y[idx])
    #     var._set_init(x0[0])
    #     var_list.append(var)
    #     for i in range(n):
    #         if i != n - 1:
    #             var = FeedVar(self, lambda t, y, idx=self.dim + i + 1: y[idx])
    #             var._set_init(x0[1 + i])
    #         else:
    #             var = FeedVar(self)
    #         var_list.append(var)
    #     self.dim += n
    #     return var_list

    def create_source(self, fun: Callable) -> "TemporalVar":
        """
        Create a source signal from a temporal function.
        :param fun: function f(t)
        :return: Solver variable
        """
        return TemporalVar(self, lambda t, y: fun(t))

    def solve(self, t_end: Number):
        # Apply checks before attempting to solve
        self._check_feed_init()
        x0 = [x.init for x in self.initialized_vars]
        # Reinit values
        [var._reset() for var in self.vars]
        start = time.time()
        try:
            res = solve_ivp(self._dy, (0, t_end), x0)
        except RecursionError:
            raise RecursionError("An algebraic loop has been detected in the system. "
                                 "Please check in the set_value() methods if a variable use itself for computing "
                                 "its value.")
        print(f"Performance = {time.time() - start}")
        self.t = res.t
        self.y = res.y
        self.solved = True
        return res

    def explore(self, f: Callable, t_end: Number, bounds=()):
        params = signature(f).parameters

        def wrapper(*args, **kwargs):
            self._clear()
            var = f(*args, **kwargs)
            self.solve(t_end)
            return var.t, var.values

        explore(wrapper, params, bounds)

    def _clear(self):
        """
        Clear stored information.
        """
        self.__init__()

    def _dy(self, t, y):
        return [var(t, y) for var in self.feed_vars]

    def _check_feed_init(self):
        uninitialized_vars = [var for var in self.feed_vars if var.function is None]
        if uninitialized_vars:
            raise ValueError(f"The following variables have not been set a value: {uninitialized_vars}. "
                             f"Call the set_value() method of each of these variables.")


class TemporalVar:
    def __init__(self, solver: Solver, fun: Callable = None, x0=None):
        self.solver = solver
        self.init = None
        self.function = fun
        self.initialized = False
        self._values = None

        self.solver.vars.append(self)
        if x0:
            self._set_init(x0)

    @property
    def values(self):
        if not self.solver.solved:
            raise Exception("The differential system has not been solved. "
                            "Call the solve() method before inquiring the variable values.")
        if self._values is None:
            self._values = self.function(self.solver.t, self.solver.y)
        return self._values

    @property
    def t(self):
        if not self.solver.solved:
            raise Exception("The differential system has not been solved. "
                            "Call the solve() method before inquiring the time variable.")
        return self.solver.t

    def apply_function(self, f: Callable) -> "TemporalVar":
        return TemporalVar(self.solver, lambda t, y: f(self(t, y)))

    def _reset(self):
        self._values = None

    def _set_init(self, x0: Number):
        self.init = x0
        self.initialized = True
        self.solver.initialized_vars.append(self)

    def __call__(self, t, y):
        return self.function(t, y)

    def __add__(self, other) -> "TemporalVar":
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: self(t, y) + other(t, y))
        else:
            return TemporalVar(self.solver, lambda t, y: other + self(t, y))

    def __radd__(self, other) -> "TemporalVar":
        return self.__add__(other)

    def __sub__(self, other) -> "TemporalVar":
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: self(t, y) - other(t, y))
        else:
            return TemporalVar(self.solver, lambda t, y: self(t, y) - other)

    def __rsub__(self, other) -> "TemporalVar":
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: other(t, y) - self(t, y))
        else:
            return TemporalVar(self.solver, lambda t, y: other - self(t, y))

    def __mul__(self, other) -> "TemporalVar":
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: self(t, y) * other(t, y))
        else:
            return TemporalVar(self.solver, lambda t, y: other * self(t, y))

    def __rmul__(self, other) -> "TemporalVar":
        return self.__mul__(other)

    def __truediv__(self, other) -> "TemporalVar":
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: self(t, y) / other(t, y))
        else:
            return TemporalVar(self.solver, lambda t, y: self(t, y) / other)

    def __rtruediv__(self, other) -> "TemporalVar":
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: other(t, y) / self(t, y))
        else:
            return TemporalVar(self.solver, lambda t, y: other / self(t, y))

    def __floordiv__(self, other) -> "TemporalVar":
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: self(t, y) // other(t, y))
        else:
            return TemporalVar(self.solver, lambda t, y: self(t, y) // other)

    def __rfloordiv__(self, other) -> "TemporalVar":
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: other(t, y) // self(t, y))
        else:
            return TemporalVar(self.solver, lambda t, y: other // self(t, y))

    def __mod__(self, other) -> "TemporalVar":
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: self(t, y) % other(t, y))
        else:
            return TemporalVar(self.solver, lambda t, y: self(t, y) % other)

    def __rmod__(self, other) -> "TemporalVar":
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: other(t, y) % self(t, y))
        else:
            return TemporalVar(self.solver, lambda t, y: other % self(t, y))

    def __pow__(self, other) -> "TemporalVar":
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: self(t, y) ** other(t, y))
        else:
            return TemporalVar(self.solver, lambda t, y: self(t, y) ** other)

    def __rpow__(self, other) -> "TemporalVar":
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: other(t, y) ** self(t, y))
        else:
            return TemporalVar(self.solver, lambda t, y: other ** self(t, y))

    def __pos__(self) -> "TemporalVar":
        return self

    def __neg__(self) -> "TemporalVar":
        return TemporalVar(self.solver, lambda t, y: - self(t, y))

    def __abs__(self) -> "TemporalVar":
        return TemporalVar(self.solver, lambda t, y: abs(self(t, y)))

    def __array_ufunc__(self, ufunc, method, *inputs) -> "TemporalVar":
        if method == "__call__":
            if len(inputs) == 1:
                return TemporalVar(self.solver, lambda t, y: ufunc(inputs[0](t, y)))
            elif len(inputs) == 2:
                return TemporalVar(self.solver, lambda t, y: ufunc(inputs[0](t, y), inputs[1]))
            else:
                return NotImplemented
        else:
            return NotImplemented

    def __repr__(self):
        if self.solver.solved:
            return f"{self.values}"
        for key, value in globals().items():
            if value is self:
                return str(key)
        else:
            return "Variable name has not been found in globals"


def compose(fun: Callable, var: TemporalVar) -> TemporalVar:
    return var.apply_function(fun)


class LoopNode(TemporalVar):
    def __init__(self, solver: Solver, input_value):
        super().__init__(solver)
        if isinstance(input_value, TemporalVar):
            self.function = input_value.function
        else:
            self.function = lambda t, y: input_value

    def loop_into(self, added_signal: TemporalVar):
        self.function = lambda t, y: self(t, y) + added_signal(t, y)


# class FeedVar(TemporalVar):
#     def __init__(self, solver: Solver, fun: Callable = None):
#         super().__init__(solver, fun)
#         self.solver.feed_vars.append(self)
#
#     def set_value(self, value):
#         if isinstance(value, TemporalVar):
#             if value.function is not None:
#                 self.function = value.function
#             else:
#                 raise RecursionError("There is an algebraic loop with this variable.")
#         else:
#             self.function = lambda t, y: value


if __name__ == '__main__':
    solver = Solver()

    m = 1
    k = 1
    c = 1
    v0 = 0
    x0 = 5
    x = 1

    acc = solver.loop_node(1 / m * x)
    vit = solver.integrate(acc, v0)
    pos = solver.integrate(vit, x0)
    # acc.loop_into(1 / m * (-c * vit - k * pos + x))
    solver.solve(50)

    plt.plot(pos.t, pos.values)
    plt.show()

    # def f(k=2, c=3, m=5, x0=1, v0=1):
    #     pos, vit, acc = solver.create_derivatives((x0, v0))
    #     acc.set_value(1 / m * (-c * vit - k * pos))
    #     return pos
    #
    #
    # t_final = 50
    # solver.explore(f, t_final, bounds=((-10, 10), (-10, 10), (0, 10)))
