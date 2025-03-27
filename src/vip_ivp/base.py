import functools
import time
import warnings
from typing import Dict, Any, overload, Literal, TypeAlias
from numbers import Number
from pathlib import Path
from typing import Callable, Union, TypeVar, Generic

import matplotlib.pyplot as plt
from sliderplot import sliderplot
import pandas as pd
from scipy.interpolate import interp1d

from .solver_utils import *
from .utils import add_necessary_brackets, convert_to_string

T = TypeVar("T")
EventAction: TypeAlias = Callable[[], None]


class Solver:
    def __init__(self):
        self.dim = 0
        self.vars = []
        self.feed_vars = []
        self.x0 = []
        self.events = []
        self.t = None
        self.y = None
        self.solved = False
        self.saved_vars = {}
        self.named_vars = {}
        self.vars_to_plot = {}

    def integrate(self, input_value: "TemporalVar[T]", x0: T) -> "TemporalVar[T]":
        """
        Integrate the input value starting from the initial condition x0.

        :param input_value: The value to be integrated.
        :param x0: The initial condition for the integration.
        :return: The integrated TemporalVar.
        """
        if isinstance(input_value, (dict, list, np.ndarray)):
            input_value = TemporalVar(self, input_value)
        integrated_structure = self._get_integrated_structure(input_value, x0)
        integrated_variable = TemporalVar(
            self,
            integrated_structure,
            expression=f"#INTEGRATE {get_expression(input_value)}",
        )
        return integrated_variable

    def _get_integrated_structure(self, data, x0):
        if isinstance(data, TemporalVar):
            if isinstance(data.function, np.ndarray):
                return [
                    self._get_integrated_structure(data[idx], np.array(x0)[idx])
                    for idx in np.ndindex(data.function.shape)
                ]

            elif isinstance(data.function, dict):
                return {
                    key: self._get_integrated_structure(value, x0[key])
                    for key, value in data.function.items()
                }

        return self._add_integration_variable(data, x0)

    def _add_integration_variable(
            self, var: Union["TemporalVar[T]", T], x0: T
    ) -> "TemporalVar[T]":
        self.feed_vars.append(var)
        integrated_variable = TemporalVar(
            self,
            lambda t, y, idx=self.dim: y[idx],
            expression=f"#INTEGRATE {get_expression(var)}",
        )
        self.x0.append(x0)
        self.dim += 1
        return integrated_variable

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
        self._get_remaining_named_variables()
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
                self._dy, (0, t_end), self.x0, method=method, t_eval=t_eval, events=self.events, **options
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

    def solve_ivp(
            self,
            fun,
            t_span,
            y0,
            method="RK45",
            t_eval=None,
            dense_output=False,
            events=None,
            vectorized=False,
            args=None,
            **options,
    ):
        if method not in METHODS and not (
                inspect.isclass(method) and issubclass(method, OdeSolver)
        ):
            raise ValueError(f"`method` must be one of {METHODS} or OdeSolver class.")

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

            jac = options.get("jac")
            if callable(jac):
                options["jac"] = lambda t, x: jac(t, x, *args)

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
        if events is not None:
            events, max_events, event_dir = prepare_events(events)
            event_count = np.zeros(len(events))
            if args is not None:
                # Wrap user functions in lambdas to hide the additional parameters.
                # The original event function is passed as a keyword argument to the
                # lambda to keep the original function in scope (i.e., avoid the
                # late binding closure "gotcha").
                events = [lambda t, x, event=event: event(t, x, *args)
                          for event in events]
            g = [event(t0, y0) for event in events]
            t_events = [[] for _ in range(len(events))]
            y_events = [[] for _ in range(len(events))]
        else:
            t_events = None
            y_events = None

        interpolants = []

        status = None
        while status is None:
            message = solver.step()

            if solver.status == "finished":
                status = 0
            elif solver.status == "failed":
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

            if events is not None:
                g_new = [event(t, y) for event in events]
                active_events = find_active_events(g, g_new, event_dir)
                if active_events.size > 0:
                    if sol is None:
                        sol = solver.dense_output()

                    event_count[active_events] += 1
                    root_indices, roots, terminate = handle_events(
                        sol, events, active_events, event_count, max_events,
                        t_old, t)

                    for e, te in zip(root_indices, roots):
                        t_events[e].append(te)
                        y_events[e].append(sol(te))

                    if terminate:
                        status = 1
                        t = roots[-1]
                        y = sol(t)
                        t_eval=None

                g = g_new

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
                        sol = solver.dense_output()
                    self.t.extend(t_eval_step)
                    if self.dim != 0:
                        self.y.extend(np.vstack(sol(t_eval_step)).T)
                    else:
                        self.y.extend([0] * len(t_eval_step))
                    t_eval_i = t_eval_i_new

            if t_eval is not None and dense_output:
                ti.append(t)

        message = MESSAGES.get(status, message)
        if t_events is not None:
            t_events = np.array([np.asarray(te) for te in t_events])
            y_events = np.array([np.asarray(ye) for ye in y_events])

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
            status=status,
            message=message,
            success=status >= 0,
        )


class TemporalVar(Generic[T]):
    def __init__(
            self,
            solver: "Solver",
            fun: Union[
                Callable[[Union[float, np.ndarray], np.ndarray], T], np.ndarray, dict
            ] = None,
            expression: str = None,
    ):
        self.solver = solver
        if callable(fun) and not isinstance(fun, TemporalVar):
            n_args = len(inspect.signature(fun).parameters)
            if n_args == 1:
                self.function = lambda t, y: fun(t)
            else:
                self.function = lambda t, y: fun(t, y)
        elif np.isscalar(fun):
            self.function = lambda t, y: fun if np.isscalar(t) else np.full_like(t, fun)
        elif isinstance(fun, (list, np.ndarray)):
            self.function = np.vectorize(lambda f: TemporalVar(solver, f))(
                np.array(fun)
            )
        elif isinstance(fun, TemporalVar):
            self.function = fun.function
        elif isinstance(fun, dict):
            self.function = {key: TemporalVar(solver, val) for key, val in fun.items()}
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
        return self.solver.t

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
                    direction: Literal["rising", "falling", "both"] = "both", terminal: bool = False) -> "EventAction":
        event = Event(self.solver, self - value, action, direction, terminal)
        return event.get_delete_from_simulation_action()

    def _reset(self):
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
        variable: TemporalVar = TemporalVar(
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


class LoopNode(TemporalVar[T]):
    def __init__(self, solver: "Solver", shape: Union[int, tuple[int, ...]] = None):
        if shape is not None:
            initial_value = np.zeros(shape)
        else:
            initial_value = 0
        self._input_var: TemporalVar = TemporalVar(solver, initial_value)
        super().__init__(solver, self._input_var, expression="")
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

    def __init__(self, solver: Solver, fun: Callable, action: Union[EventAction, None],
                 direction: Literal["rising", "falling", "both"] = "both",
                 terminal: bool = False):
        self.solver = solver
        self.function = fun
        self.action = action
        self.terminal = terminal
        self.direction = self.DIRECTION_MAP[direction]

        self.solver.events.append(self)

    def __call__(self, t, y) -> float:
        return self.function(t, y)

    def get_delete_from_simulation_action(self) -> EventAction:
        return lambda: self.solver.events.remove(self)
