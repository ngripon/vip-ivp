import functools
import time
import warnings
from collections import abc
from copy import copy
from typing import overload, Literal, TypeAlias, Type, ParamSpec, List, Iterable
from numbers import Number
from pathlib import Path
from typing import Callable, Union, TypeVar, Generic

import matplotlib.pyplot as plt
import numpy as np
from sliderplot import sliderplot
import pandas as pd
from scipy.interpolate import interp1d

from .solver_utils import *
from .utils import add_necessary_brackets, convert_to_string

T = TypeVar("T")
EventAction: TypeAlias = Callable[[float, np.ndarray], None]


class Solver:
    def __init__(self):
        self.dim = 0
        self.vars: List[TemporalVar] = []
        # All the scalars variables that are an input of an integrate function
        self.feed_vars: List[TemporalVar] = []
        # All the scalars variables that are an output of an integrate function
        self.integrated_vars: List[IntegratedVar] = []

        self.events = []
        self.t = []
        self.y = None
        self.solved = False
        self.saved_vars = {}
        self.named_vars = {}
        self.vars_to_plot = {}
        self.status = None

    def integrate(self, input_value: "TemporalVar[T]", x0: T, minimum: Union[T, "TemporalVar[T]"] = None,
                  maximum: Union[T, "TemporalVar[T]"] = None) -> "IntegratedVar[T]":
        """
        Integrate the input value starting from the initial condition x0.

        :param input_value: The value to be integrated.
        :param x0: The initial condition for the integration.
        :return: The integrated TemporalVar.
        """
        if isinstance(input_value, (dict, list, np.ndarray)):
            input_value = TemporalVar(self, input_value)
        integrated_structure = self._get_integrated_structure(input_value, x0, minimum, maximum)
        integrated_variable = IntegratedVar(
            self,
            integrated_structure,
            f"#INTEGRATE {get_expression(input_value)}",
            x0
        )
        return integrated_variable

    def _get_integrated_structure(self, data, x0, minimum, maximum):
        if isinstance(data, TemporalVar):
            if isinstance(data.function, np.ndarray):
                if not isinstance(maximum, np.ndarray):
                    maximum = np.full(data.function.shape, maximum)
                if not isinstance(minimum, np.ndarray):
                    minimum = np.full(data.function.shape, minimum)
                return [
                    self._get_integrated_structure(data[idx], np.array(x0)[idx], minimum[idx], maximum[idx])
                    for idx in np.ndindex(data.function.shape)
                ]

            elif isinstance(data.function, dict):
                if not isinstance(maximum, dict):
                    maximum = {key: maximum for key in data.function.keys()}
                if not isinstance(minimum, dict):
                    minimum = {key: minimum for key in data.function.keys()}
                return {
                    key: self._get_integrated_structure(value, x0[key], minimum[key], maximum[key])
                    for key, value in data.function.items()
                }

        return self._add_integration_variable(data, x0, minimum, maximum)

    def _add_integration_variable(self, var: Union["TemporalVar[T]", T], x0: T, minimum: T,
                                  maximum: T) -> "IntegratedVar[T]":
        self.feed_vars.append(var)
        # Manage min and max
        if maximum is None:
            maximum = np.inf
        if minimum is None:
            minimum = -np.inf

        # Add integration value
        integrated_variable = IntegratedVar(
            self,
            lambda t, y, idx=self.dim: y[idx],
            f"#INTEGRATE {get_expression(var)}",
            x0,
            minimum,
            maximum,
            self.dim,
        )
        self.integrated_vars.append(integrated_variable)
        self.dim += 1
        return integrated_variable

    def solve(
            self,
            t_end: float,
            method="RK45",
            time_step=None,
            t_eval=None,
            include_events_times: bool = True,
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
        self._get_remaining_named_variables()
        # Reinit values
        [var.reset() for var in self.vars]
        start = time.time()
        # Set t_eval
        if time_step is not None and t_eval is None:
            t_eval = np.arange(0, t_end + time_step, time_step)
        try:
            res = self._solve_ivp((0, t_end), self.x0, method=method, t_eval=t_eval,
                                  include_events_times=include_events_times, **options)
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

    @property
    def x0(self):
        return np.array([v.x0 for v in self.integrated_vars])

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
        plt.xlim(0, self.t[-1])
        plt.grid()
        plt.tight_layout()
        plt.show()

    def explore(
            self,
            f: Callable,
            t_end: Number,
            bounds=(),
            time_step: float = None,
            title: str = "",
    ):
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
        sliderplot(
            wrapper,
            bounds,
            page_title="vip-ivp",
            titles=[title],
            axes_labels=(("Time (s)", ""),),
        )

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

    def _get_remaining_named_variables(self):
        frame = inspect.currentframe().f_back
        while (frame.f_locals.get("self")
               and (isinstance(frame.f_locals.get("self"), TemporalVar)
                    or isinstance(frame.f_locals.get("self"), Solver))
               or Path(frame.f_code.co_filename).as_posix().endswith("vip_ivp/api.py")):
            frame = frame.f_back
        local_variables = frame.f_locals
        for key, value in local_variables.items():
            if isinstance(value, TemporalVar) and key not in self.named_vars:
                self.named_vars[key] = value

    def _solve_ivp(
            self,
            t_span,
            y0,
            method="RK45",
            t_eval=None,
            dense_output=False,
            vectorized=False,
            include_events_times=True,
            **options,
    ):
        if method not in METHODS and not (
                inspect.isclass(method) and issubclass(method, OdeSolver)
        ):
            raise ValueError(f"`method` must be one of {METHODS} or OdeSolver class.")

        t0, tf = map(float, t_span)

        y0 = self._bound_sol(t0, y0)

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

        solver = method(self._dy, t0, y0, tf, vectorized=vectorized, **options)
        if self.get_events(t0) is not None:
            [event.evaluate(t0, y0) for event in self.get_events(t0)]
            t_events = []
        else:
            t_events = None
            y_events = None

        interpolants = []

        self.status = None
        while self.status is None:
            message = solver.step()

            t_old = solver.t_old
            t = solver.t
            y = self._bound_sol(t, solver.y)

            events = self.get_events(t)

            if dense_output:
                sol = lambda t: self._bound_sol(t, solver.dense_output()(t))
                interpolants.append(sol)
            else:
                sol = None

            if events is not None:
                [event.evaluate(t, y) for event in events]
                active_events_indices = find_active_events(events)
                if active_events_indices.size > 0:
                    if sol is None:
                        sol = self._sol_wrapper(solver.dense_output())
                    for active_idx in active_events_indices:
                        events[active_idx].count += 1
                    active_events, roots = handle_events(sol, events, active_events_indices, t_old, t, t_eval)

                    # Get the first event, execute its action and relaunch the solver to begin at te.
                    e_idx = np.argmin(roots)
                    active_event: Event = events[e_idx]
                    te = roots[0]
                    ye = sol(te)
                    active_event.t_events.append(te)
                    active_event.y_events.append(ye)
                    active_event.execute_action(te, ye)
                    t = te
                    y = ye
                    events = self.get_events(te)
                    [event.evaluate(t, y) for event in events]
                    solver = method(self._dy, t, y, tf, vectorized=vectorized, **options)

                    if active_event.terminal:
                        self.status = 1

            if t_eval is None:
                self.t.append(t)
                self.y.append(y)
            else:
                # The value in t_eval equal to t will be included.
                if solver.direction > 0:
                    t_eval_i_new = np.searchsorted(t_eval, t, side="right")
                    t_eval_step = t_eval[t_eval_i:t_eval_i_new]
                else:
                    t_eval_i_new = np.searchsorted(t_eval, t, side="left")
                    # It has to be done with two slice operations, because
                    # you can't slice to 0th element inclusive using backward
                    # slicing.
                    t_eval_step = t_eval[t_eval_i_new:t_eval_i][::-1]

                if t_eval_step.size > 0:
                    if sol is None:
                        sol = self._sol_wrapper(solver.dense_output())
                    self.t.extend(t_eval_step)
                    if self.dim != 0:
                        self.y.extend(np.vstack(sol(t_eval_step)).T)
                    else:
                        self.y.extend([0] * len(t_eval_step))
                    t_eval_i = t_eval_i_new
                if events is not None and include_events_times:
                    if active_events_indices.size > 0 and self.status != 1:
                        self.t.append(te)
                        # When there is no integrated variable, self.y should be a list of zeros
                        self.y.append(ye if len(ye) else 0.0)

            if t_eval is not None and dense_output:
                ti.append(t)

            if solver.status == "finished":
                self.status = 0
            elif solver.status == "failed":
                self.status = -1
                break

        message = MESSAGES.get(self.status, message)
        if t_events is not None:
            t_events = [np.asarray(e.t_events) for e in events]
            y_events = [np.asarray(e.y_events) for e in events]

        if self.t:
            self.t = np.array(self.t)
            self.y = np.vstack(self.y).T

        if dense_output:
            if t_eval is None:
                sol = OdeSolution(
                    self.t,
                    interpolants,
                    alt_segment=True if method in [BDF, LSODA] else False,
                )
            else:
                sol = OdeSolution(
                    ti,
                    interpolants,
                    alt_segment=True if method in [BDF, LSODA] else False,
                )
        else:
            sol = None

        return OdeResult(
            t=self.t,
            y=self.y,
            t_events=t_events,
            y_events=y_events,
            sol=sol,
            nfev=solver.nfev,
            njev=solver.njev,
            nlu=solver.nlu,
            status=self.status,
            message=message,
            success=self.status >= 0,
        )

    def _bound_sol(self, t, y: np.ndarray):
        upper, lower = self._get_bounds(t, y)
        y_bounded_max = np.where(y < upper, y, upper)
        y_bounded = np.where(y_bounded_max > lower, y_bounded_max, lower)
        return y_bounded

    def _get_bounds(self, t, y):
        upper = []
        lower = []
        for var in self.integrated_vars:
            maximum = var.maximum(t, y)
            minimum = var.minimum(t, y)
            if np.any(minimum > maximum):
                raise ValueError(
                    f"Lower bound {minimum} is greater than upper bound {maximum} a time {t} s "
                    f"for variable {var.name or var.expression}."
                )
            upper.append(maximum)
            lower.append(minimum)
        upper = np.array(upper)
        lower = np.array(lower)
        return upper, lower

    def _sol_wrapper(self, sol):
        def output_fun(t: Union[float, np.ndarray]):
            return self._bound_sol(t, sol(t))

        return output_fun

    def get_events(self, t):
        event_list = [e for e in self.events if e.deletion_time is None or e.deletion_time < t]
        if not event_list:
            return None
        return event_list


class TemporalVar(Generic[T]):
    def __init__(
            self,
            solver: "Solver",
            fun: Union[
                Callable[[Union[float, np.ndarray], np.ndarray], T], np.ndarray, dict
            ] = None,
            expression: str = None,
            child_cls=None
    ):
        self.solver = solver
        self._output_type = None
        # Recursive building
        child_cls = child_cls or type(self)
        if callable(fun) and not isinstance(fun, child_cls):
            n_args = len(inspect.signature(fun).parameters)
            if n_args == 1:
                self.function = lambda t, y: fun(t)
            else:
                self.function = lambda t, y: fun(t, y)
        elif np.isscalar(fun):
            self.function = lambda t, y: fun if np.isscalar(t) else np.full(t.shape, fun)
            self._output_type = type(fun)
        elif isinstance(fun, (list, np.ndarray)):
            self._output_type = np.ndarray
            self.function = np.vectorize(lambda f: child_cls(solver, f))(
                np.array(fun)
            )
        elif isinstance(fun, TemporalVar):
            vars(self).update(vars(fun))
        elif isinstance(fun, dict):
            self._output_type = dict
            self.function = {key: child_cls(solver, val) for key, val in fun.items()}
        else:
            raise ValueError(f"Unsupported type: {type(fun)}.")

        self._values = None
        # Variable definition
        self._expression = convert_to_string(fun) if expression is None else expression
        self.name = None
        self._inputs: list[TemporalVar] = []

        self.solver.vars.append(self)

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
        return np.asarray(self.solver.t)

    @property
    def output_type(self):
        if self._output_type is None:
            self._output_type = type(self(0, self.solver.x0))
        return self._output_type

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
        if isinstance(self.function, np.ndarray):
            [
                self[idx].to_plot(f"{name}[{', '.join(str(i) for i in idx)}]")
                for idx in np.ndindex(self.function.shape)
            ]
            return
        elif isinstance(self.function, dict):
            [self[key].to_plot(f"{name}[{key}]") for key in self.function.keys()]
            return
        self.solver.vars_to_plot[name] = self

    @classmethod
    def from_scenario(
            cls,
            solver: "Solver",
            scenario_table: pd.DataFrame,
            time_key: str,
            interpolation_kind="linear",
    ) -> "TemporalVar":
        variables = {}
        for col in scenario_table.columns:
            if col == time_key:
                continue
            fun = interp1d(
                scenario_table[time_key],
                scenario_table[col],
                kind=interpolation_kind,
                bounds_error=False,
                fill_value=(scenario_table[col].iat[0], scenario_table[col].iat[-1]),
            )
            variables[col] = fun
        return cls(solver, variables)

    def on_crossing(self, value: T, action: "EventAction" = None,
                    direction: Literal["rising", "falling", "both"] = "both",
                    terminal: Union[bool, int] = False) -> "EventAction":
        if self.output_type in (bool, np.bool, str):
            crossed_variable = self == value
        elif issubclass(self.output_type, abc.Iterable):
            raise ValueError(
                "Can not apply crossing detection to a variable containing a collection of values because it is ambiguous."
            )
        else:
            crossed_variable = self - value
        event = Event(self.solver, crossed_variable, action, direction, terminal)
        return event.get_delete_from_simulation_action()

    def change_behavior(self, value: T) -> EventAction:
        def change_value(t):
            time = TemporalVar(self.solver, lambda t: t)
            new_value = TemporalVar(self.solver, value)
            new_var = where(self.solver, time < t, copy(self), new_value)
            self.function = new_var.function
            self._expression = new_var.expression

        return lambda t, y: change_value(t)

    def reset(self):
        self._values = None

    def __call__(self, t: Union[float, np.ndarray], y: np.ndarray) -> T:
        if isinstance(self.function, np.ndarray):
            if np.isscalar(t):
                return np.stack(np.frompyfunc(lambda f: f(t, y), 1, 1)(self.function))
            return np.stack(
                np.frompyfunc(lambda f: f(t, y), 1, 1)(self.function.ravel())
            ).reshape((*self.function.shape, *np.array(t).shape))
        elif isinstance(self.function, dict):
            return {key: val(t, y) for key, val in self.function.items()}
        return self.function(t, y)

    def __copy__(self):
        return TemporalVar(self.solver, self.function, self.expression)

    def __add__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{get_expression(self)} + {get_expression(other)}"
        if isinstance(self.function, np.ndarray):
            return TemporalVar(
                self.solver, self.function + other.function, expression=expression
            )
        if isinstance(other, TemporalVar):
            return TemporalVar(
                self.solver,
                lambda t, y: self(t, y) + other(t, y),
                expression=expression,
            )
        else:
            return TemporalVar(
                self.solver, lambda t, y: self(t, y) + other, expression=expression
            )

    def __radd__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{get_expression(other)} + {get_expression(self)}"
        if isinstance(self.function, np.ndarray):
            return TemporalVar(
                self.solver, other.function + self.function, expression=expression
            )
        if isinstance(other, TemporalVar):
            return TemporalVar(
                self.solver,
                lambda t, y: other(t, y) + self(t, y),
                expression=expression,
            )
        else:
            return TemporalVar(
                self.solver, lambda t, y: other + self(t, y), expression=expression
            )

    def __sub__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = (
            f"{get_expression(self)} - {add_necessary_brackets(get_expression(other))}"
        )
        if isinstance(self.function, np.ndarray):
            return TemporalVar(
                self.solver, self.function - other.function, expression=expression
            )
        if isinstance(other, TemporalVar):
            return TemporalVar(
                self.solver,
                lambda t, y: self(t, y) - other(t, y),
                expression=expression,
            )
        else:
            return TemporalVar(
                self.solver, lambda t, y: self(t, y) - other, expression=expression
            )

    def __rsub__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = (
            f"{get_expression(other)} - {add_necessary_brackets(get_expression(self))}"
        )
        if isinstance(self.function, np.ndarray):
            return TemporalVar(
                self.solver, other.function - self.function, expression=expression
            )
        if isinstance(other, TemporalVar):
            return TemporalVar(
                self.solver,
                lambda t, y: other(t, y) - self(t, y),
                expression=expression,
            )
        else:
            return TemporalVar(
                self.solver, lambda t, y: other - self(t, y), expression=expression
            )

    def __mul__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(self))} * {add_necessary_brackets(get_expression(other))}"
        if isinstance(self.function, np.ndarray):
            return TemporalVar(
                self.solver, self.function * other.function, expression=expression
            )
        if isinstance(other, TemporalVar):
            return TemporalVar(
                self.solver,
                lambda t, y: self(t, y) * other(t, y),
                expression=expression,
            )
        else:
            return TemporalVar(
                self.solver, lambda t, y: other * self(t, y), expression=expression
            )

    def __rmul__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(other))} * {add_necessary_brackets(get_expression(self))}"
        if isinstance(self.function, np.ndarray):
            return TemporalVar(
                self.solver, other.function * self.function, expression=expression
            )
        if isinstance(other, TemporalVar):
            return TemporalVar(
                self.solver,
                lambda t, y: other(t, y) * self(t, y),
                expression=expression,
            )
        else:
            return TemporalVar(
                self.solver, lambda t, y: other * self(t, y), expression=expression
            )

    def __truediv__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(self))} / {add_necessary_brackets(get_expression(other))}"
        if isinstance(self.function, np.ndarray):
            return TemporalVar(
                self.solver, self.function / other.function, expression=expression
            )
        if isinstance(other, TemporalVar):
            return TemporalVar(
                self.solver,
                lambda t, y: self(t, y) / other(t, y),
                expression=expression,
            )
        else:
            return TemporalVar(
                self.solver, lambda t, y: self(t, y) / other, expression=expression
            )

    def __rtruediv__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(other))} / {add_necessary_brackets(get_expression(self))}"
        if isinstance(self.function, np.ndarray):
            return TemporalVar(
                self.solver, other.function / self.function, expression=expression
            )
        if isinstance(other, TemporalVar):
            return TemporalVar(
                self.solver,
                lambda t, y: other(t, y) / self(t, y),
                expression=expression,
            )
        else:
            return TemporalVar(
                self.solver, lambda t, y: other / self(t, y), expression=expression
            )

    def __floordiv__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(self))} // {add_necessary_brackets(get_expression(other))}"
        if isinstance(self.function, np.ndarray):
            return TemporalVar(
                self.solver, self.function // other.function, expression=expression
            )
        if isinstance(other, TemporalVar):
            return TemporalVar(
                self.solver,
                lambda t, y: self(t, y) // other(t, y),
                expression=expression,
            )
        else:
            return TemporalVar(
                self.solver, lambda t, y: self(t, y) // other, expression=expression
            )

    def __rfloordiv__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(other))} // {add_necessary_brackets(get_expression(self))}"
        if isinstance(self.function, np.ndarray):
            return TemporalVar(
                self.solver, other.function // self.function, expression=expression
            )
        if isinstance(other, TemporalVar):
            return TemporalVar(
                self.solver,
                lambda t, y: other(t, y) // self(t, y),
                expression=expression,
            )
        else:
            return TemporalVar(
                self.solver, lambda t, y: other // self(t, y), expression=expression
            )

    def __mod__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(self))} % {add_necessary_brackets(get_expression(other))}"
        if isinstance(self.function, np.ndarray):
            return TemporalVar(
                self.solver, self.function % other.function, expression=expression
            )
        if isinstance(other, TemporalVar):
            return TemporalVar(
                self.solver,
                lambda t, y: self(t, y) % other(t, y),
                expression=expression,
            )
        else:
            return TemporalVar(
                self.solver, lambda t, y: self(t, y) % other, expression=expression
            )

    def __rmod__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(other))} % {add_necessary_brackets(get_expression(self))}"
        if isinstance(self.function, np.ndarray):
            return TemporalVar(
                self.solver, other.function % self.function, expression=expression
            )
        if isinstance(other, TemporalVar):
            return TemporalVar(
                self.solver,
                lambda t, y: other(t, y) % self(t, y),
                expression=expression,
            )
        else:
            return TemporalVar(
                self.solver, lambda t, y: other % self(t, y), expression=expression
            )

    def __pow__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(self))} ** {add_necessary_brackets(get_expression(other))}"
        if isinstance(self.function, np.ndarray):
            return TemporalVar(
                self.solver, self.function ** other.function, expression=expression
            )
        if isinstance(other, TemporalVar):
            return TemporalVar(
                self.solver,
                lambda t, y: self(t, y) ** other(t, y),
                expression=expression,
            )
        else:
            return TemporalVar(
                self.solver, lambda t, y: self(t, y) ** other, expression=expression
            )

    def __rpow__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(other))} ** {add_necessary_brackets(get_expression(self))}"
        if isinstance(self.function, np.ndarray):
            return TemporalVar(
                self.solver, other.function ** self.function, expression=expression
            )
        if isinstance(other, TemporalVar):
            return TemporalVar(
                self.solver,
                lambda t, y: other(t, y) ** self(t, y),
                expression=expression,
            )
        else:
            return TemporalVar(
                self.solver, lambda t, y: other ** self(t, y), expression=expression
            )

    def __pos__(self) -> "TemporalVar[T]":
        return self

    def __neg__(self) -> "TemporalVar[T]":
        expression = f"-{add_necessary_brackets(get_expression(self))}"
        return TemporalVar(self.solver, lambda t, y: -self(t, y), expression=expression)

    def __abs__(self) -> "TemporalVar[T]":
        expression = f"abs({get_expression(self)})"
        return TemporalVar(
            self.solver, lambda t, y: abs(self(t, y)), expression=expression
        )

    @overload
    def __eq__(
            self, other: "TemporalVar[np.ndarray[T]]"
    ) -> "TemporalVar[np.ndarray[bool]]":
        ...

    def __eq__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[bool]":
        expression = f"{add_necessary_brackets(get_expression(self))} == {add_necessary_brackets(get_expression(other))}"
        if isinstance(self.function, np.ndarray):
            fun_arr = np.vectorize(lambda f, o: f == o)(self.function, other.function)
            return TemporalVar(self.solver, fun_arr, expression=expression)
        return TemporalVar(
            self.solver,
            lambda t, y: self(t, y)
                         == (other(t, y) if isinstance(other, TemporalVar) else other),
            expression=expression,
        )

    @overload
    def __ne__(
            self, other: "TemporalVar[np.ndarray[T]]"
    ) -> "TemporalVar[np.ndarray[bool]]":
        ...

    def __ne__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[bool]":
        expression = f"{add_necessary_brackets(get_expression(self))} != {add_necessary_brackets(get_expression(other))}"
        if isinstance(self.function, np.ndarray):
            fun_arr = np.vectorize(lambda f, o: f != o)(self.function, other.function)
            return TemporalVar(self.solver, fun_arr, expression=expression)
        return TemporalVar(
            self.solver,
            lambda t, y: self(t, y)
                         != (other(t, y) if isinstance(other, TemporalVar) else other),
            expression=expression,
        )

    @overload
    def __lt__(
            self, other: "TemporalVar[np.ndarray[T]]"
    ) -> "TemporalVar[np.ndarray[bool]]":
        ...

    def __lt__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[bool]":
        expression = f"{add_necessary_brackets(get_expression(self))} < {add_necessary_brackets(get_expression(other))}"
        if isinstance(self.function, np.ndarray):
            fun_arr = np.vectorize(lambda f, o: f < o)(self.function, other.function)
            return TemporalVar(self.solver, fun_arr, expression=expression)
        return TemporalVar(
            self.solver,
            lambda t, y: self(t, y)
                         < (other(t, y) if isinstance(other, TemporalVar) else other),
            expression=expression,
        )

    @overload
    def __le__(
            self, other: "TemporalVar[np.ndarray[T]]"
    ) -> "TemporalVar[np.ndarray[bool]]":
        ...

    def __le__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[bool]":
        expression = f"{add_necessary_brackets(get_expression(self))} <= {add_necessary_brackets(get_expression(other))}"
        if isinstance(self.function, np.ndarray):
            fun_arr = np.vectorize(lambda f, o: f <= o)(self.function, other.function)
            return TemporalVar(self.solver, fun_arr, expression=expression)
        return TemporalVar(
            self.solver,
            lambda t, y: self(t, y)
                         <= (other(t, y) if isinstance(other, TemporalVar) else other),
            expression=expression,
        )

    @overload
    def __gt__(
            self, other: "TemporalVar[np.ndarray[T]]"
    ) -> "TemporalVar[np.ndarray[bool]]":
        ...

    def __gt__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[bool]":
        expression = f"{add_necessary_brackets(get_expression(self))} > {add_necessary_brackets(get_expression(other))}"
        if isinstance(self.function, np.ndarray):
            fun_arr = np.vectorize(lambda f, o: f > o)(self.function, other.function)
            return TemporalVar(self.solver, fun_arr, expression=expression)
        return TemporalVar(
            self.solver,
            lambda t, y: self(t, y)
                         > (other(t, y) if isinstance(other, TemporalVar) else other),
            expression=expression,
        )

    @overload
    def __ge__(
            self, other: "TemporalVar[np.ndarray[T]]"
    ) -> "TemporalVar[np.ndarray[bool]]":
        ...

    def __ge__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[bool]":
        expression = f"{add_necessary_brackets(get_expression(self))} >= {add_necessary_brackets(get_expression(other))}"
        if isinstance(self.function, np.ndarray):
            fun_arr = np.vectorize(lambda f, o: f >= o)(self.function, other.function)
            return TemporalVar(self.solver, fun_arr, expression=expression)
        return TemporalVar(
            self.solver,
            lambda t, y: self(t, y)
                         >= (other(t, y) if isinstance(other, TemporalVar) else other),
            expression=expression,
        )

    def __getitem__(self, item):
        expression = f"{add_necessary_brackets(get_expression(self))}[{item}]"
        # Ensure that childs of IntegratedVar and LoopNodes are of the same type.
        if isinstance(self.function[item], TemporalVar):
            item_cls = type(self.function[item])
        else:
            item_cls = TemporalVar
        variable: TemporalVar = item_cls(
            self.solver, self.function[item], expression
        )
        return variable

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> "TemporalVar":
        inputs_expr = [
            get_expression(inp) if isinstance(inp, TemporalVar) else str(inp)
            for inp in inputs
        ]
        kwargs_expr = [
            f"{key}={get_expression(value) if isinstance(value, TemporalVar) else str(value)}"
            for key, value in kwargs.items()
        ]
        expression = f"{ufunc.__name__}({', '.join(inputs_expr)}"
        if kwargs:
            expression += f", {', '.join(kwargs_expr)}"
        expression += ")"
        if method == "__call__":
            return TemporalVar(
                self.solver,
                lambda t, y: ufunc(
                    *[
                        inp(t, y) if isinstance(inp, TemporalVar) else inp
                        for inp in inputs
                    ],
                    **{
                        key: (value(t, y) if isinstance(value, TemporalVar) else value)
                        for key, value in kwargs.items()
                    },
                ),
                expression=expression,
            )

        return NotImplemented

    @property
    def expression(self):
        return self._expression

    def __repr__(self) -> str:
        if self.solver.solved:
            return f"{self.values}"
        else:
            return f"{self._expression}"


def convert_args_to_temporal_var(solver: Solver, arg_list: Iterable) -> List[TemporalVar]:
    def convert(arg):
        if not isinstance(arg, TemporalVar):
            arg = TemporalVar(solver, arg)
        return arg

    return [convert(a) for a in arg_list]


def where(solver, condition: TemporalVar[bool], a: Union[T, TemporalVar[T]], b: Union[T, TemporalVar[T]]) -> \
        TemporalVar[T]:
    condition, a, b = convert_args_to_temporal_var(solver, (condition, a, b))
    return TemporalVar(solver,
                       lambda t, y: (a(t, y) if condition(t, y) else b(t, y)) if np.isscalar(t) else np.where(
                           condition(t, y), a(t, y), b(t, y)),
                       expression=f"({get_expression(a)} if {get_expression(condition)} else {get_expression(b)})")


class LoopNode(TemporalVar[T]):
    def __init__(self, solver: "Solver", shape: Union[int, tuple[int, ...]] = None):
        if shape is not None:
            initial_value = np.zeros(shape)
        else:
            initial_value = 0
        self._input_var: TemporalVar = TemporalVar(solver, initial_value)
        super().__init__(solver, self._input_var, expression="", child_cls=TemporalVar)
        self._is_set = False

    def loop_into(self, value: Union[TemporalVar[T], T], force: bool = False):
        """
        Set the input value of the loop node.

        :param force: Add the value to the loop node even if it has already been set.
        :param value: The value to add, can be a TemporalVar or a number.
        """
        if self._is_set and not force:
            raise Exception(
                "This Loop Node has already been set. If you want to add another value, use argument 'force = True'."
            )
        if not isinstance(value, TemporalVar):
            value = TemporalVar(self.solver, value)
        if not self._is_set:
            self._input_var = value
        else:
            self._input_var += value
        self._is_set = True
        self._expression = get_expression(self._input_var)
        self.function = self._input_var.function

    def __call__(self, t: Union[float, np.ndarray], y: np.ndarray) -> T:
        return self._input_var(t, y)

    def __getitem__(self, item):
        expression = f"{add_necessary_brackets(get_expression(self))}[{item}]"
        variable: TemporalVar = TemporalVar(
            self.solver, lambda t, y: self(t, y)[item], expression
        )
        return variable


class IntegratedVar(TemporalVar[T]):
    def __init__(
            self,
            solver: "Solver",
            fun: Union[
                Callable[[Union[float, np.ndarray], np.ndarray], T], np.ndarray, dict
            ] = None,
            expression: str = None,
            x0: T = None,
            minimum: Union[TemporalVar[T], T] = -np.inf,
            maximum: Union[TemporalVar[T], T] = np.inf,
            y_idx: int = None
    ):
        self.x0 = x0
        self.maximum = convert_args_to_temporal_var(solver, (maximum,))[0]
        self.minimum = convert_args_to_temporal_var(solver, (minimum,))[0]

        self._y_idx = y_idx
        if isinstance(fun, IntegratedVar):
            self._y_idx = IntegratedVar.y_idx
        super().__init__(solver, fun, expression)

    @property
    def y_idx(self):
        if isinstance(self.function, np.ndarray):
            return np.vectorize(lambda v: v.y_idx)(self.function)
        elif isinstance(self.function, dict):
            return {key: value.y_idx for key, value in self.function.items()}
        elif self._y_idx is not None:
            return self._y_idx
        raise ValueError("The argument 'y_idx' should be set for IntegratedVar containing a single value.")

    def set_value(self, value: Union[TemporalVar[T], T]) -> EventAction:
        if not isinstance(value, TemporalVar):
            value = TemporalVar(self.solver, value)

        def action(t, y):
            def set_y0(idx, subvalue):
                if isinstance(idx, np.ndarray):
                    for arr_idx in np.ndindex(idx.shape):
                        y_idx = idx[arr_idx]
                        set_y0(y_idx, value[y_idx])
                elif isinstance(idx, dict):
                    for key, idx in idx.items():
                        y[idx] = value[key]
                        set_y0(idx, value[key])
                else:
                    y[idx] = subvalue(t, y)

            set_y0(self.y_idx, value)

        return action

    def change_behavior(self, value: T) -> EventAction:
        raise NotImplementedError(
            "This method is irrelevant for an integrated variable. "
            "If you want really want to change the behavior of an integrated variable, create a new variable by doing "
            "'new_var = 1*integrated_variable'."
        )


def get_expression(value) -> str:
    if isinstance(value, TemporalVar):
        frame = inspect.currentframe().f_back.f_back
        while (
                frame.f_locals.get("self")
                and (
                        isinstance(frame.f_locals.get("self"), TemporalVar)
                        or isinstance(frame.f_locals.get("self"), Solver)
                )
                or Path(frame.f_code.co_filename).as_posix().endswith("vip_ivp/api.py")
        ):
            frame = frame.f_back
        found_key = next(
            (key for key, dict_value in frame.f_locals.items() if dict_value is value),
            None,
        )
        if found_key is not None:
            value.name = found_key
            value.solver.named_vars[found_key] = value
            return value.name
        return value.expression
    else:
        return str(value)


class Event:
    DIRECTION_MAP = {"rising": 1, "falling": -1, "both": 0}

    def __init__(self, solver: Solver, fun, action: Union[EventAction, None],
                 direction: Literal["rising", "falling", "both"] = "both",
                 terminal: Union[bool, int] = False):
        self.solver = solver
        self.function: TemporalVar = convert_args_to_temporal_var(self.solver, [fun])[0]
        self.action = action
        self.terminal = terminal
        self.direction = self.DIRECTION_MAP[direction]

        self.count = 0
        # Compute max events
        message = ('The `terminal` attribute of each event '
                   'must be a boolean or positive integer.')
        if terminal is None or terminal == 0:
            self.max_events = np.inf
        elif int(terminal) == terminal and terminal > 0:
            self.max_events = terminal
        else:
            raise ValueError(message)

        self.value = None
        self.value_new = None

        self.t_events = []
        self.y_events = []

        self.deletion_time = None  # Time at which the event has been deleted from the simulation.

        self.solver.events.append(self)

    def __call__(self, t, y) -> float:
        return self.function(t, y)

    def evaluate(self, t, y):
        self.value = self.value_new
        self.value_new = self(t, y)

    def get_delete_from_simulation_action(self) -> EventAction:
        def delete_event(t, y):
            self.deletion_time = t

        return delete_event

    def execute_action(self, t, y):
        if self.action is not None:
            self.action(t, y)
