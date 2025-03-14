import functools
import time
import warnings
import inspect

from numbers import Number
from typing import Callable, Union, TypeVar, Generic

import matplotlib.pyplot as plt
import numpy as np
from sliderplot import sliderplot

from scipy.integrate._ivp.bdf import BDF
from scipy.integrate._ivp.radau import Radau
from scipy.integrate._ivp.rk import RK23, RK45, DOP853
from scipy.integrate._ivp.lsoda import LSODA
from scipy.optimize import OptimizeResult
from scipy.integrate._ivp.common import OdeSolution
from scipy.integrate._ivp.base import OdeSolver

T = TypeVar('T')

METHODS = {'RK23': RK23,
           'RK45': RK45,
           'DOP853': DOP853,
           'Radau': Radau,
           'BDF': BDF,
           'LSODA': LSODA}

MESSAGES = {0: "The solver successfully reached the end of the integration interval.",
            1: "A termination event occurred."}


class Solver:
    def __init__(self):
        self.dim = 0
        self.vars = []
        self.feed_vars = []
        self.initialized_vars = []
        self.t = None
        self.y = None
        self.solved = False
        self.saved_vars = {}
        self.vars_to_plot = {}

    def integrate(self, input_value: "TemporalVar", x0: float) -> "TemporalVar":
        """
        Integrate the input value starting from the initial condition x0.

        :param input_value: The value to be integrated.
        :param x0: The initial condition for the integration.
        :return: The integrated TemporalVar.
        """
        self.feed_vars.append(input_value)
        integrated_variable = TemporalVar(
            self, lambda t, y, idx=self.dim: y[idx], x0)
        self.dim += 1
        return integrated_variable

    def loop_node(self) -> "LoopNode":
        """
        Create a loop node. A loop node is a temporal variable that dissociate its declaration and value setting.
        This mechanism allows it to take as input variable that are integrated from itself, thus allowing to solve ODEs.

        :return: The created LoopNode.
        """
        return LoopNode(self)

    def create_source(self, value: Union[Callable, Number]) -> "TemporalVar":
        """
        Create a source signal from a temporal function or a scalar value.

        :param value: A function f(t) or a scalar value.
        :return: The created TemporalVar.
        """
        if callable(value):
            return TemporalVar(self, lambda t, y: value(t))
        else:
            return TemporalVar(self, lambda t, y: value if np.isscalar(t) else np.full_like(t, value))

    def solve(
            self,
            t_end: Number,
            method="RK45",
            time_step=None,
            t_eval=None,
            plot: bool = True,
            **options,
    ) -> None:
        """
        Solve the equations of the dynamical system through an integration scheme.

        :param t_end: Time at which the integration stops.
        :param method: Integration method to use. Default is 'RK45'.
        :param time_step: Time step for the integration. If None, use points selected by the solver.
        :param t_eval: Times at which to store the computed solution. If None, use points selected by the solver.
        :param plot: Plot the variables that called the "to_plot()" method.
        :param options: Additional options for the solver. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html.
        """
        # Apply checks before attempting to solve
        x0 = [x.init for x in self.initialized_vars]
        # Reinit values
        [var._reset() for var in self.vars]
        start = time.time()
        # Set t_eval
        if time_step is not None:
            if t_eval is not None:
                warnings.warn(
                    "The value of t_eval has been overridden because time_step parameter is not None."
                )
            t_eval = np.arange(0, t_end, time_step)
        try:
            res = self.solve_ivp(
                self._dy, (0, t_end), x0, method=method, t_eval=t_eval, **options
            )
            if not res.success:
                raise Exception(res.message)
        except RecursionError:
            raise RecursionError(
                "An algebraic loop has been detected in the system. "
                "Please check in the set_value() methods if a variable use itself for computing "
                "its value."
            )
        print(f"Performance = {time.time() - start}")
        self.solved = True
        if plot:
            self.plot()

    def plot(self):
        """
        Plot the variables that have been marked for plotting.
        """
        if not self.vars_to_plot:
            return
        # Plot data
        for variable_name, var in self.vars_to_plot.items():
            plt.plot(var.t, var.values, label=variable_name)
        # Label and axis
        plt.title("Simulation results")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.xlim(0, var.t[-1])
        plt.grid()
        plt.tight_layout()
        plt.show()

    def explore(self, f: Callable, t_end: Number, bounds=(), time_step: float = None, title: str = ""):
        """
        Explore the function f over the given bounds and solve the system until t_end.
        This function needs the sliderplot package.

        :param title: Title of the plot
        :param time_step: Time step of the simulation
        :param f: The function to explore.
        :param t_end: Time at which the integration stops.
        :param bounds: Bounds for the exploration.
        """

        def wrapper(*args, **kwargs):
            self.clear()
            outputs = f(*args, **kwargs)
            self.solve(t_end, time_step=time_step)
            transformed_outputs = self.unwrap_leaves(outputs)
            return transformed_outputs

        functools.update_wrapper(wrapper, f)
        sliderplot(wrapper, bounds, page_title="vip-ivp",
                   titles=[title], axes_labels=(("Time (s)", ""),))

    def clear(self):
        """
        Clear stored information.
        """
        self.__init__()

    def _dy(self, t, y):
        return [var(t, y) if callable(var) else var for var in self.feed_vars]

    def unwrap_leaves(self, outputs):
        """
        Transform all TemporalVar in an iterable into (x.t, x.values) pairs.

        :param outputs: The outputs to transform.
        :return: The transformed outputs.
        """
        if isinstance(outputs, TemporalVar):
            return outputs.t, outputs.values
        else:
            return list(map(self.unwrap_leaves, (el for el in outputs)))

    def solve_ivp(self, fun, t_span, y0, method='RK45', t_eval=None, dense_output=False,
                  events=None, vectorized=False, args=None, **options):
        if method not in METHODS and not (
                inspect.isclass(method) and issubclass(method, OdeSolver)):
            raise ValueError(
                f"`method` must be one of {METHODS} or OdeSolver class.")

        t0, tf = map(float, t_span)

        if args is not None:
            # Wrap the user's fun (and jac, if given) in lambdas to hide the
            # additional parameters.  Pass in the original fun as a keyword
            # argument to keep it in the scope of the lambda.
            try:
                _ = [*(args)]
            except TypeError as exp:
                suggestion_tuple = (
                    "Supplied 'args' cannot be unpacked. Please supply `args`"
                    f" as a tuple (e.g. `args=({args},)`)"
                )
                raise TypeError(suggestion_tuple) from exp

            def fun(t, x, fun=fun):
                return fun(t, x, *args)

            jac = options.get('jac')
            if callable(jac):
                options['jac'] = lambda t, x: jac(t, x, *args)

        if t_eval is not None:
            t_eval = np.asarray(t_eval)
            if t_eval.ndim != 1:
                raise ValueError("`t_eval` must be 1-dimensional.")

            if np.any(t_eval < min(t0, tf)) or np.any(t_eval > max(t0, tf)):
                raise ValueError("Values in `t_eval` are not within `t_span`.")

            d = np.diff(t_eval)
            if tf > t0 and np.any(d <= 0) or tf < t0 and np.any(d >= 0):
                raise ValueError("Values in `t_eval` are not properly sorted.")

            if tf > t0:
                t_eval_i = 0
            else:
                # Make order of t_eval decreasing to use np.searchsorted.
                t_eval = t_eval[::-1]
                # This will be an upper bound for slices.
                t_eval_i = t_eval.shape[0]

        if method in METHODS:
            method = METHODS[method]

        if t_eval is None:
            self.t = [t0]
            self.y = [y0]
        elif t_eval is not None and dense_output:
            self.t = []
            ti = [t0]
            self.y = []
        else:
            self.t = []
            self.y = []

        solver = method(fun, t0, y0, tf, vectorized=vectorized, **options)

        interpolants = []

        status = None
        while status is None:
            message = solver.step()

            if solver.status == 'finished':
                status = 0
            elif solver.status == 'failed':
                status = -1
                break

            t_old = solver.t_old
            t = solver.t
            y = solver.y

            if dense_output:
                sol = solver.dense_output()
                interpolants.append(sol)
            else:
                sol = None

            if t_eval is None:
                self.t.append(t)
                self.y.append(y)
            else:
                # The value in t_eval equal to t will be included.
                if solver.direction > 0:
                    t_eval_i_new = np.searchsorted(t_eval, t, side='right')
                    t_eval_step = t_eval[t_eval_i:t_eval_i_new]
                else:
                    t_eval_i_new = np.searchsorted(t_eval, t, side='left')
                    # It has to be done with two slice operations, because
                    # you can't slice to 0th element inclusive using backward
                    # slicing.
                    t_eval_step = t_eval[t_eval_i_new:t_eval_i][::-1]

                if t_eval_step.size > 0:
                    if sol is None:
                        sol = solver.dense_output()
                    self.t.extend(t_eval_step)
                    self.y.extend(np.vstack(sol(t_eval_step)).T)
                    t_eval_i = t_eval_i_new

            if t_eval is not None and dense_output:
                ti.append(t)

        message = MESSAGES.get(status, message)
        if self.t:
            self.t = np.array(self.t)
            self.y = np.vstack(self.y).T

        if dense_output:
            if t_eval is None:
                sol = OdeSolution(
                    self.t, interpolants, alt_segment=True if method in [
                        BDF, LSODA] else False
                )
            else:
                sol = OdeSolution(
                    ti, interpolants, alt_segment=True if method in [
                        BDF, LSODA] else False
                )
        else:
            sol = None

        return OdeResult(t=self.t, y=self.y, sol=sol,
                         nfev=solver.nfev, njev=solver.njev, nlu=solver.nlu,
                         status=status, message=message, success=status >= 0)


class TemporalVar(Generic[T]):
    def __init__(self, solver: Solver, fun: Callable[[Union[float, np.ndarray], np.ndarray], T] = None, x0: Union[float, np.ndarray] = None):
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
            raise Exception(
                "The differential system has not been solved. "
                "Call the solve() method before inquiring the variable values."
            )
        if self._values is None:
            self._values = self(self.solver.t, self.solver.y)
        return self._values

    @property
    def t(self) -> np.ndarray:
        if not self.solver.solved:
            raise Exception(
                "The differential system has not been solved. "
                "Call the solve() method before inquiring the time variable."
            )
        return self.solver.t

    def apply_function(self, f: Callable[[T], T]) -> "TemporalVar[T]":
        """
        Apply a function to the TemporalVar.

        :param f: The function to apply.
        :return: The new TemporalVar with the applied function.
        """
        return TemporalVar(self.solver, lambda t, y: f(self(t, y)))

    def save(self, name: str) -> None:
        """
        Save the temporal variable with a name.

        :param name: Key to retrieve the variable.
        """
        if name in self.solver.saved_vars:
            warnings.warn(
                f"A variable with name {name} already exists. Its value has been overridden."
            )
        self.solver.saved_vars[name] = self

    def to_plot(self, name: str) -> None:
        """
        Add the variable to the plotted data on solve.

        :param name: Name of the variable in the legend of the plot.
        """
        self.solver.vars_to_plot[name] = self

    def delay(self, n_steps: int, initial_value: T = 0) -> "TemporalVar[T]":
        if n_steps < 1:
            raise Exception("Delay accept only a positive step.")

        def previous_value(t, y):
            if np.isscalar(t):
                if len(self.solver.t) >= n_steps:
                    previous_t = self.solver.t[-n_steps]
                    previous_y = self.solver.y[-n_steps]

                    return self(previous_t, previous_y)
                else:
                    return initial_value
            else:
                delayed_t = shift_array(t, n_steps, 0)
                delayed_y = shift_array(y, n_steps, initial_value)
                return self(delayed_t, delayed_y)

        return TemporalVar(self.solver, previous_value)

    def _reset(self):
        self._values = None

    def _set_init(self, x0: Union[float, np.ndarray]):
        self.init = x0
        self.solver.initialized_vars.append(self)

    def __call__(self, t: Union[float, np.ndarray], y: np.ndarray) -> T:
        return self.function(t, y)

    def __add__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: self(t, y) + other(t, y))
        else:
            return TemporalVar(self.solver, lambda t, y: other + self(t, y))

    def __radd__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        return self.__add__(other)

    def __sub__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: self(t, y) - other(t, y))
        else:
            return TemporalVar(self.solver, lambda t, y: self(t, y) - other)

    def __rsub__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: other(t, y) - self(t, y))
        else:
            return TemporalVar(self.solver, lambda t, y: other - self(t, y))

    def __mul__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: self(t, y) * other(t, y))
        else:
            return TemporalVar(self.solver, lambda t, y: other * self(t, y))

    def __rmul__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        return self.__mul__(other)

    def __truediv__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: self(t, y) / other(t, y))
        else:
            return TemporalVar(self.solver, lambda t, y: self(t, y) / other)

    def __rtruediv__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: other(t, y) / self(t, y))
        else:
            return TemporalVar(self.solver, lambda t, y: other / self(t, y))

    def __floordiv__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: self(t, y) // other(t, y))
        else:
            return TemporalVar(self.solver, lambda t, y: self(t, y) // other)

    def __rfloordiv__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: other(t, y) // self(t, y))
        else:
            return TemporalVar(self.solver, lambda t, y: other // self(t, y))

    def __mod__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: self(t, y) % other(t, y))
        else:
            return TemporalVar(self.solver, lambda t, y: self(t, y) % other)

    def __rmod__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: other(t, y) % self(t, y))
        else:
            return TemporalVar(self.solver, lambda t, y: other % self(t, y))

    def __pow__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: self(t, y) ** other(t, y))
        else:
            return TemporalVar(self.solver, lambda t, y: self(t, y) ** other)

    def __rpow__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: other(t, y) ** self(t, y))
        else:
            return TemporalVar(self.solver, lambda t, y: other ** self(t, y))

    def __pos__(self) -> "TemporalVar[T]":
        return self

    def __neg__(self) -> "TemporalVar[T]":
        return TemporalVar(self.solver, lambda t, y: -self(t, y))

    def __abs__(self) -> "TemporalVar[T]":
        return TemporalVar(self.solver, lambda t, y: abs(self(t, y)))

    def __eq__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[bool]":
        return TemporalVar(
            self.solver,
            lambda t, y: self(t, y) == (
                other(t, y) if isinstance(other, TemporalVar) else other)
        )

    def __ne__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[bool]":
        return TemporalVar(
            self.solver,
            lambda t, y: self(t, y) != (
                other(t, y) if isinstance(other, TemporalVar) else other)
        )

    def __lt__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[bool]":
        return TemporalVar(
            self.solver,
            lambda t, y: self(t, y) < (
                other(t, y) if isinstance(other, TemporalVar) else other)
        )

    def __le__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[bool]":
        return TemporalVar(
            self.solver,
            lambda t, y: self(t, y) <= (
                other(t, y) if isinstance(other, TemporalVar) else other)
        )

    def __gt__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[bool]":
        return TemporalVar(
            self.solver,
            lambda t, y: self(t, y) > (
                other(t, y) if isinstance(other, TemporalVar) else other)
        )

    def __ge__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[bool]":
        return TemporalVar(
            self.solver,
            lambda t, y: self(t, y) >= (
                other(t, y) if isinstance(other, TemporalVar) else other)
        )

    def __getitem__(self, item):
        return TemporalVar(self.solver, lambda t, y: self(t, y)[item])

    def __getattr__(self, item):
        return TemporalVar(self.solver, lambda t, y: getattr(self(t, y), item))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> "TemporalVar":
        if method == "__call__":
            return TemporalVar(
                self.solver,
                lambda t, y: ufunc(
                    *[inp(t, y) if isinstance(inp, TemporalVar)
                      else inp for inp in inputs],
                    **{
                        key: (value(t, y) if isinstance(
                            value, TemporalVar) else value)
                        for key, value in kwargs.items()
                    }
                ),
            )

        return NotImplemented

    def __array__(self) -> np.ndarray:
        return self.values

    def __repr__(self) -> str:
        if self.solver.solved:
            return f"{self.values}"
        else:
            return "Please call solve to get the values."


def compose(fun: Callable[[T], T], var: TemporalVar[T]) -> TemporalVar[T]:
    """
    Compose a function with a TemporalVar.

    :param fun: The function to compose.
    :param var: The TemporalVar to compose with.
    :return: The new TemporalVar with the composed function.
    """
    return var.apply_function(fun)


class LoopNode(TemporalVar[float]):
    def __init__(self, solver: Solver):
        self._nested_functions = []
        super().__init__(solver, lambda t, y: 0)
        self._is_set = False

    def loop_into(
            self,
            value: Union[TemporalVar[T], T],
            force: bool = False
    ):
        """
        Set the input value of the loop node.

        :param force: Add the value to the loop node even if it has already been set.
        :param value: The value to add, can be a TemporalVar or a number.
        """
        if self._is_set and not force:
            raise Exception(
                "This Loop Node has already been set. If you want to add another value, use argument 'force = True'."
            )
        index = len(self._nested_functions) - 1
        if isinstance(value, TemporalVar):
            def new_fun(t, y, i=index): return value(
                t, y) + self._nested_functions[i](t, y)
        else:
            def new_fun(t, y, i=index): return self._nested_functions[i](
                t, y) + value
        self._nested_functions.append(new_fun)
        self._is_set = True

    @property
    def function(self) -> Callable[[Union[float, np.ndarray], np.ndarray], T]:
        return self._nested_functions[-1]

    @function.setter
    def function(self, value: Callable[[Union[float, np.ndarray], np.ndarray], T]):
        self._nested_functions.append(value)

    def __call__(self, t: Union[float, np.ndarray], y: np.ndarray) -> T:
        return self.function(t, y)


class OdeResult(OptimizeResult):
    pass


def shift_array(arr: np.ndarray, n: int, fill_value: float = 0):
    shifted = np.roll(arr, n, axis=-1)  # Shift the array
    if n > 0:
        shifted[..., :n] = fill_value  # Fill first n elements
    elif n < 0:
        shifted[..., n:] = fill_value  # Fill last n elements
    return shifted
