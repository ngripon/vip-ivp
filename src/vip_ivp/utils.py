import functools
import operator
import time
import warnings

from numbers import Number
from typing import Callable, Union

import numpy as np
from scipy.integrate import solve_ivp
from sliderplot import sliderplot


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
        self.feed_vars.append(input_value)
        integrated_variable = TemporalVar(self, lambda t, y, idx=self.dim: y[idx], x0)
        self.dim += 1
        return integrated_variable

    def loop_node(self, input_value=0) -> "LoopNode":
        return LoopNode(self, input_value)

    def create_source(self, value: Union[Callable, Number]) -> "TemporalVar":
        """
        Create a source signal from a temporal function.
        :param value: function f(t) or scalar
        :return: Solver variable
        """
        if callable(value):
            return TemporalVar(self, lambda t, y: value(t))
        else:
            return TemporalVar(self, lambda t, y: value)

    def solve(self, t_end: Number, method='RK45', time_step=None, t_eval=None, **options) -> None:
        """
        Solve the equations of the dynamical system through an integration scheme.
        :param t_end: Time at which the integration stops
        :param method: Integration method to use. For more information, check https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html.
        :param t_eval: Times at which to store the computed solution, must be sorted and lie within t_span. If None (default), use points selected by the solver.
        :param options: Please check https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html.
        :return:
        """
        # Apply checks before attempting to solve
        x0 = [x.init for x in self.initialized_vars]
        # Reinit values
        [var._reset() for var in self.vars]
        start = time.time()
        # Set t_eval
        if time_step is not None:
            if t_eval is not None:
                warnings.warn("The value of t_eval has been overridden because time_step parameter is not None.")
            t_eval = np.arange(0, t_end, time_step)
        try:
            res = solve_ivp(self._dy, (0, t_end), x0, method=method, t_eval=t_eval, **options)
        except RecursionError:
            raise RecursionError("An algebraic loop has been detected in the system. "
                                 "Please check in the set_value() methods if a variable use itself for computing "
                                 "its value.")
        print(f"Performance = {time.time() - start}")
        self.t = res.t
        self.y = res.y
        self.solved = True

    def explore(self, f: Callable, t_end: Number, bounds=()):
        def wrapper(*args, **kwargs):
            self.clear()
            outputs = f(*args, **kwargs)
            self.solve(t_end)
            transformed_outputs = self.unwrap_leaves(outputs)
            return transformed_outputs

        functools.update_wrapper(wrapper, f)
        sliderplot(wrapper, bounds)

    def clear(self):
        """
        Clear stored information.
        """
        self.__init__()

    def _dy(self, t, y):
        return [var(t, y) if callable(var) else var for var in self.feed_vars]

    def unwrap_leaves(self, outputs):
        """
        Transform all TemporalVar in an iterable into (x.t, x.values) pairs
        :param outputs:
        :return:
        """
        if isinstance(outputs, TemporalVar):
            return outputs.t, outputs.values
        else:
            return list(map(self.unwrap_leaves, (el for el in outputs)))


class TemporalVar:
    def __init__(self, solver: Solver, fun: Callable = None, x0=None):
        self.solver = solver
        self.init = None
        if isinstance(fun, Callable):
            self.function = fun
        else:
            self.function = lambda t, y: fun
        self._values = None

        self.solver.vars.append(self)
        if x0 is not None:
            self._set_init(x0)

    @property
    def values(self) -> np.ndarray:
        if not self.solver.solved:
            raise Exception("The differential system has not been solved. "
                            "Call the solve() method before inquiring the variable values.")
        if self._values is None:
            self._values = self(self.solver.t, self.solver.y)
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
                if callable(inputs[0]):
                    return TemporalVar(self.solver, lambda t, y: ufunc(inputs[0](t, y)))
                else:
                    return TemporalVar(self.solver, lambda t, y: ufunc(inputs[0]))
            elif len(inputs) == 2:
                # Bad coding...
                if callable(inputs[0]) and callable(inputs[1]):
                    return TemporalVar(self.solver, lambda t, y: ufunc(inputs[0](t, y), inputs[1](t, y)))
                elif callable(inputs[0]) and not callable(inputs[1]):
                    return TemporalVar(self.solver, lambda t, y: ufunc(inputs[0](t, y), inputs[1]))
                elif not callable(inputs[0]) and callable(inputs[1]):
                    return TemporalVar(self.solver, lambda t, y: ufunc(inputs[0], inputs[1](t, y)))
                else:
                    return TemporalVar(self.solver, lambda t, y: ufunc(inputs[0], inputs[1]))

            else:
                return NotImplemented
        else:
            return NotImplemented

    def __array__(self):
        return self.values

    def __repr__(self):
        if self.solver.solved:
            return f"{self.values}"
        else:
            return "Please call solve to get the values."


def compose(fun: Callable, var: TemporalVar) -> TemporalVar:
    return var.apply_function(fun)


class LoopNode(TemporalVar):
    def __init__(self, solver: Solver, input_value):
        self._nested_functions = []
        super().__init__(solver, input_value)

    def loop_into(self, added_value: Union[TemporalVar, Number], operator_fun: Callable = operator.add):
        index = len(self._nested_functions) - 1
        if isinstance(added_value, TemporalVar):
            new_fun = lambda t, y, i=index: operator_fun(added_value(t, y), self._nested_functions[i](t, y))
        else:
            new_fun = lambda t, y, i=index: operator_fun(self._nested_functions[i](t, y), added_value)
        self._nested_functions.append(new_fun)

    @property
    def function(self):
        return self._nested_functions[-1]

    @function.setter
    def function(self, value):
        self._nested_functions.append(value)

    def __call__(self, t, y):
        return self.function(t, y)