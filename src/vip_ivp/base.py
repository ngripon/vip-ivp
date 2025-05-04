import enum
import functools
import inspect
import operator
import time
import warnings
from collections import abc
from copy import copy
from typing import overload, Literal, Iterable, Dict, Tuple, List, Any
from pathlib import Path
from typing import Callable, Union, TypeVar, Generic

import numpy as np
from numpy.typing import NDArray
from typing_extensions import ParamSpec
from cachebox import LRUCache

from .solver_utils import *
from .utils import add_necessary_brackets, convert_to_string, operator_call, shift_array, vectorize_source

if TYPE_CHECKING:
    import pandas as pd

T = TypeVar("T")
P = ParamSpec("P")


class Solver:
    def __init__(self):
        self.dim = 0
        self.vars: List[TemporalVar] = []
        # All the scalars variables that are an input of an integrate function
        self.feed_vars: List[TemporalVar] = []
        # All the scalars variables that are an output of an integrate function
        self.integrated_vars: List[IntegratedVar] = []

        self.events: List[Event] = []
        self.cross_triggers: List[CrossTriggerVar] = []
        self.t_current: float = 0
        self.t = []
        self.y = None
        self.solved = False
        self.saved_vars: Dict[str, TemporalVar] = {}
        self.named_vars: Dict[str, TemporalVar] = {}
        self.vars_to_plot: Dict[str, TemporalVar] = {}
        self.status: Union[int, None] = None

        # Special solver TemporalVars
        self.time_variable: TemporalVar[float] = TemporalVar(self, lambda t: t, "Time")

    def integrate(self, input_value: "TemporalVar[T]", x0: T, minimum: Union[T, "TemporalVar[T]", None] = None,
                  maximum: Union[T, "TemporalVar[T]", None] = None) -> "IntegratedVar[T]":
        """
        Integrate the input value starting from the initial condition x0.

        :param input_value: The value to be integrated.
        :param x0: The initial condition for the integration.
        :return: The integrated TemporalVar.
        """
        if not isinstance(input_value, TemporalVar):
            input_value = TemporalVar(self, input_value)
        elif input_value.is_discrete:
            raise NotImplementedError("Discrete integration is not supported yet.")

        integrated_structure = self._get_integrated_structure(input_value, x0, minimum, maximum)
        integrated_variable = IntegratedVar(
            self,
            integrated_structure,
            f"#INTEGRATE {get_expression(input_value)}",
            x0
        )
        return integrated_variable

    def _get_integrated_structure(self, data, x0, minimum, maximum):
        if data.output_type is np.ndarray:
            if not isinstance(maximum, np.ndarray):
                maximum = np.full(data.shape, maximum)
            if not isinstance(minimum, np.ndarray):
                minimum = np.full(data.shape, minimum)
            result = np.empty(data.shape, dtype=object)
            for idx in np.ndindex(data.shape):
                result[idx] = self._get_integrated_structure(data[idx], np.array(x0)[idx], minimum[idx], maximum[idx])
            return result
        elif data.output_type is dict:
            if not isinstance(maximum, dict):
                maximum = {key: maximum for key in data.keys()}
            if not isinstance(minimum, dict):
                minimum = {key: minimum for key in data.keys()}

            return {
                key: self._get_integrated_structure(value, x0[key], minimum[key], maximum[key])
                for key, value in data.items()
            }

        return self._add_integration_variable(data, x0, minimum, maximum)

    def _add_integration_variable(self, var: Union["TemporalVar[T]", T], x0: T, minimum: Union["TemporalVar[T]", T],
                                  maximum: Union["TemporalVar[T]", T]) -> "IntegratedVar[T]":
        self.feed_vars.append(var)
        # Manage min and max
        if maximum is None:
            maximum = np.inf
        if minimum is None:
            minimum = -np.inf

        # Evaluate bounds at t = 0
        low_0 = minimum(0, self.x0) if isinstance(minimum, TemporalVar) else minimum
        up_0 = maximum(0, self.x0) if isinstance(maximum, TemporalVar) else maximum
        if not low_0 <= x0 <= up_0:
            raise ValueError(
                f"x0 = {x0} is outside the specified bounds [{low_0} ; {up_0}] at t=0. Please provide a value within these bounds.")

        # Add integration value
        integrated_variable = IntegratedVar(
            self,
            lambda t, y, idx=self.dim: y[idx] if y.ndim == 1 else y[..., idx],
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
            verbose: bool = False,
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
        # Check if LoopNodes has been set
        for loop_node in (var for var in self.vars if isinstance(var, LoopNode)):
            if not loop_node.is_valid():
                raise Exception(
                    f"The value has not been set in loop node {loop_node.name}. Please call the 'loop_into()' method.\n"
                    "If this is intentional, instantiate the LoopNode object with parameter 'strict = False' to "
                    "disable this exception.")
        # Reinit values
        [var.clear() for var in self.vars]
        [e.clear() for e in self.events]
        start = time.time()
        # Set t_eval
        if time_step is not None and t_eval is None:
            t_eval = np.arange(0, t_end + time_step / 2,
                               time_step)  # Add half a time step to get an array that stops on t_end

        res = self._solve_ivp((0, t_end), self.x0, method=method, t_eval=t_eval,
                              include_events_times=include_events_times, **options)
        if not res.success:
            raise Exception(res.message)

        self.solved = True
        if verbose:
            output_str = f"Solving time = {time.time() - start} s\n"
            if self.events:
                output_str += f"Number of triggered events = {sum(len(t) for t in res.t_events)}\n"
            print(output_str)
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

        import matplotlib.pyplot as plt
        # Plot data
        plt.figure("Results for VIP")
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
            t_end: float,
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
        from sliderplot import sliderplot

        def wrapper(*args, **kwargs):
            self.clear()
            outputs = f(*args, **kwargs)
            self.solve(t_end, time_step=time_step)
            transformed_outputs = self._unwrap_leaves(outputs)
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
        result_list = []
        for var in self.feed_vars:
            try:
                result = var(t, y)
                result_list.append(result)
            except RecursionError:
                raise RecursionError(
                    f"An algebraic loop has been detected when trying to compute the value of variable {var.name}.\n"
                    f"Make sure that a variable does not reference itself in `.loop_into()` methods."
                )
        return result_list

    def _unwrap_leaves(self, outputs):
        """
        Transform all TemporalVar in an iterable into (x.t, x.values) pairs.

        :param outputs: The outputs to transform.
        :return: The transformed outputs.
        """
        if isinstance(outputs, TemporalVar):
            return outputs.t, outputs.values
        else:
            return list(map(self._unwrap_leaves, (el for el in outputs)))

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
                value.name = key
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

        t_events = []
        y_events = []
        [e.evaluate(t0, y0) for e in self.get_events(t0)]

        interpolants = []

        self.status = None
        first_event = True
        while self.status is None:
            message = solver.step()

            if solver.status == "failed":
                self.status = -1
                break

            t_old = solver.t_old
            t = solver.t
            y = self._bound_sol(t, solver.y)

            events = self.get_events(t)

            if dense_output:
                sol = lambda t: self._bound_sol(t, solver.dense_output()(t))
                interpolants.append(sol)
            else:
                sol = None

            if events or self.cross_triggers:
                if sol is None:
                    sol = self._sol_wrapper(solver.dense_output())

                # Create list of time sample where to check triggers and events
                t_eval_i_new = np.searchsorted(t_eval, t, side="right")
                t_eval_step = t_eval[:t_eval_i_new]
                t_eval_step = t_eval_step[t_eval_step > t_old]
                t_eval_step = [*t_eval_step, t]

                for t_check in t_eval_step:
                    # Check triggers first
                    if self.cross_triggers:
                        ...

                    # Check events
                    if events:
                        ...
                    ...

                active_events_indices, te_upper, te_lower = find_active_events(events, sol, t_eval, t, t_old)
                if active_events_indices.size > 0:
                    for active_idx in active_events_indices:
                        events[active_idx].count += 1
                    active_events, roots = handle_events(sol, events, active_events_indices, te_lower, te_upper, t_eval)

                    # Get the first event
                    te = np.min(roots)
                    if te != t_old or first_event:  # Prevent endless loops
                        first_event = False  # Only enable the case where we found an event at t = 0s
                        ye = sol(te)
                        self.t_current = te
                        # Get all events that happens at te and execute their action
                        triggering_events = [active_events[i] for i in range(len(active_events)) if roots[i] == te]
                        triggered_events = []
                        for event in triggering_events:
                            event.t_events.append(te)
                            event.execute_action(te, ye)
                            triggered_events.append(event)
                        # Create a loop to check if other events has triggered
                        while True:
                            t_latest = t_eval[np.searchsorted(t_eval, te, side="left")]
                            g_latest = [e(t_latest, sol(t_latest)) for e in events]
                            g_current = [e(te, ye) for e in events]
                            direction = np.array([e.direction for e in events])
                            active_events_indices_cascade = find_active_events_in_step(g_latest, g_current, direction)
                            triggering_events = [self.events[i] for i in active_events_indices_cascade if
                                                 self.events[i] not in triggered_events]
                            if triggering_events:
                                for event in triggering_events:
                                    event.t_events.append(te)
                                    event.execute_action(te, ye)
                                    triggered_events.append(event)
                            else:
                                break

                        # Reset the solver and events evaluation to begin at te for the next step
                        t = te
                        y = ye
                        [e.evaluate(t, y) for e in self.get_events(t)]
                        for event in triggered_events:
                            event.g = 0
                        solver = method(self._dy, t, y, tf, vectorized=vectorized, **options)
                    else:
                        warnings.warn(
                            "Ignored event: The detected event computed time is the same as the previously solved "
                            "time.\n"
                            "A discontinuity in the variable you applied '.on_crossing()' to may cause the root "
                            "finding algorithm to fail. Consider applying '.on_crossing' to a more continuous variable."
                        )
                        active_events_indices = np.array([])
                        [e.evaluate(t, y) for e in self.get_events(t)]
                else:
                    [e.evaluate(t, y) for e in self.get_events(t)]
            self.t_current = t
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
                        self.y.extend(sol(t_eval_step))
                    else:
                        self.y.extend([0] * len(t_eval_step))
                    t_eval_i = t_eval_i_new
                # Add time events
                if events:
                    if active_events_indices.size > 0:
                        if self.t[-1] == te:
                            if self.dim != 0:
                                self.y[-1] = ye
                        elif include_events_times:
                            self.t.append(te)
                            # When there is no integrated variable, self.y should be a list of zeros
                            self.y.append(ye if len(ye) else 0.0)

            if t_eval is not None and dense_output:
                ti.append(t)

            if solver.status == "finished":
                self.status = 0

        message = MESSAGES.get(self.status, message)
        if self.events:
            t_events = [np.asarray(e.t_events) for e in events]
            # y_events = [np.asarray(e.y_events) for e in events]

        if self.t:
            self.t = np.array(self.t)
            self.y = np.array(self.y)

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

    def _bound_sol(self, t, y: NDArray):
        upper, lower = self._get_bounds(t, y)
        y_bounded_max = np.where(y <= upper, y, upper)
        y_bounded = np.where(y_bounded_max >= lower, y_bounded_max, lower)
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
        upper = np.moveaxis(np.array(upper), 0, -1)
        lower = np.moveaxis(np.array(lower), 0, -1)
        return upper, lower

    def _sol_wrapper(self, sol):
        def output_fun(t: Union[float, NDArray]):
            y = sol(t)
            if not np.isscalar(t):
                y = np.moveaxis(y, 0, -1)
            return self._bound_sol(t, y)

        return output_fun

    def get_events(self, t):
        event_list = [e for e in self.events]
        return event_list


class CallMode(enum.Enum):
    CALL_ARGS_FUN = 0
    CALL_FUN_RESULT = 1


class TemporalVar(Generic[T]):
    def __init__(
            self,
            solver: "Solver",
            source: Union[
                Callable[[Union[float, NDArray], NDArray], T],
                Callable[[Union[float, NDArray]], T],
                NDArray,
                Dict,
                float,
                Tuple
            ] = None,
            expression: str = None,
            child_cls=None,
            operator=None,
            call_mode: CallMode = CallMode.CALL_ARGS_FUN,
            is_discrete=False
    ):
        self.solver = solver
        self._output_type = None
        self._is_source = False
        self._call_mode = call_mode
        self.is_discrete = is_discrete
        # Recursive building
        self.operator = operator
        child_cls = child_cls or type(self)
        if self.operator is not None:
            self.source = source
        else:
            self._is_source = True
            if callable(source) and not isinstance(source, child_cls):
                n_args = len(inspect.signature(source).parameters)
                if n_args == 1:
                    self.source = lambda t, y: vectorize_source(source)(t)
                else:
                    self.source = lambda t, y: source(t, y)
            elif np.isscalar(source):
                self.source = source
                self._output_type = type(source)
            elif isinstance(source, (list, np.ndarray)):
                self._output_type = np.ndarray
                self.source = np.vectorize(lambda f: child_cls(solver, f))(
                    np.array(source)
                )
            elif isinstance(source, TemporalVar):
                vars(self).update(vars(source))
            elif isinstance(source, dict):
                self._output_type = dict
                self.source = {key: child_cls(solver, val) for key, val in source.items()}
            elif source is None:
                self.source = None
            else:
                raise ValueError(f"Unsupported type: {type(source)}.")

        self._values = None
        # Variable definition
        self._expression = convert_to_string(source) if expression is None else expression
        self.name = None

        # self.events: List[Event] = []

        self._cache = LRUCache(maxsize=4)

        self.solver.vars.append(self)

    def __call__(self, t: Union[float, NDArray], y: NDArray) -> T:
        # Handle dict in a recursive way
        if isinstance(self.source, dict):
            return {key: val(t, y) for key, val in self.source.items()}
        else:
            # Handle the termination leaves of the recursion
            if isinstance(t, np.ndarray):
                t_cache = tuple(t)
            else:
                t_cache = t
            if t_cache in self._cache:
                return self._cache[t_cache]
            elif isinstance(self.source, np.ndarray):
                if np.isscalar(t):
                    output = np.stack(np.frompyfunc(lambda f: f(t, y), 1, 1)(self.source))
                else:
                    output = np.stack(
                        np.frompyfunc(lambda f: f(t, y), 1, 1)(self.source.ravel())
                    ).reshape((*self.source.shape, *np.array(t).shape))
            elif self.operator is not None:
                if self.operator is operator_call and not np.isscalar(t):
                    output = np.array([self._resolve_operator(t[i], y[i]) for i in range(len(t))])
                    if output.ndim > 1:
                        output = np.moveaxis(output, 0, -1)
                else:
                    output = self._resolve_operator(t, y)
            else:
                if callable(self.source):
                    output = self.source(t, y)
                elif np.isscalar(t):
                    output = self.source
                else:
                    output = np.full(len(t), self.source)
            if self.solver.solved:
                self._cache[t_cache] = output
        return output

    def _resolve_operator(self, t, y):
        if self._call_mode == CallMode.CALL_ARGS_FUN:
            args = [x(t, y) if isinstance(x, TemporalVar) else x for x in self.source if
                    not isinstance(x, dict)]
            kwargs = {k: v for d in [x for x in self.source if isinstance(x, dict)] for k, v in d.items()}
            kwargs = {k: (x(t, y) if isinstance(x, TemporalVar) else x) for k, x in kwargs.items()}
            return self.operator(*args, **kwargs)
        elif self._call_mode == CallMode.CALL_FUN_RESULT:
            args = [x for x in self.source if not isinstance(x, dict)]
            kwargs = {k: v for d in [x for x in self.source if isinstance(x, dict)] for k, v in d.items()}
            return self.operator(*args, **kwargs)(t, y)
        else:
            raise ValueError(f"Unknown call mode: {self._call_mode}.")

    @property
    def values(self) -> NDArray:
        if not self.solver.solved:
            raise Exception(
                "The differential system has not been solved. "
                "Call the solve() method before inquiring the variable values."
            )
        try:
            if self._values is None:
                self._values = self(self.t, self.solver.y)
        except RecursionError:
            raise RecursionError(
                f"An algebraic loop has been detected when trying to compute the value of variable {self.name}.\n"
                f"Make sure that a variable does not reference itself in `.loop_into()` methods."
            )
        return self._values

    @property
    def t(self) -> NDArray:
        if not self.solver.solved:
            raise Exception(
                "The differential system has not been solved. "
                "Call the solve() method before inquiring the time variable."
            )
        return np.asarray(self.solver.t)

    @property
    def output_type(self):
        if self._output_type is None:
            self._output_type = type(self._first_value())
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

    def to_plot(self, name: str = None) -> None:
        """
        Add the variable to the plotted data on solve.

        :param name: Name of the variable in the legend of the plot.
        """
        if name is None:
            if self.name is None:
                get_expression(self)
            name = self.name
        if self.output_type is np.ndarray:
            [
                self[idx].to_plot(f"{name}[{', '.join(str(i) for i in idx)}]")
                for idx in np.ndindex(self.shape)
            ]
            return
        elif isinstance(self.source, dict):
            [self[key].to_plot(f"{name}[{key}]") for key in self._first_value().keys()]
            return
        self.solver.vars_to_plot[name] = self

    @classmethod
    def from_scenario(
            cls,
            solver: "Solver",
            scenario_table: "pd.DataFrame",
            time_key: str,
            interpolation_kind="linear",
    ) -> "TemporalVar":
        from scipy.interpolate import interp1d

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

    def delayed(self, delay: int, initial_value: T = 0) -> "TemporalVar[T]":
        """
        Create a delayed version of the TemporalVar.
        :param delay: Number of solver steps by which the new TemporalVar is delayed.
        :param initial_value: Value of the delayed variable at the beginning when there is not any value for the original value.
        :return: Delayed version of the TemporalVar
        """
        if delay < 1:
            raise Exception("Delay accept only a positive step.")

        def create_delay(input_variable):
            def previous_value(t, y):
                if np.isscalar(t):
                    if len(input_variable.solver.t) >= delay:
                        index = np.searchsorted(input_variable.solver.t, t, "left")
                        # index = next((i for i, ts in enumerate(input_variable.solver.t) if t <= ts),
                        #              len(input_variable.solver.t))
                        if index - delay < 0:
                            return initial_value
                        previous_t = input_variable.solver.t[index - delay]
                        previous_y = input_variable.solver.y[index - delay]

                        return input_variable(previous_t, previous_y)
                    else:
                        return initial_value
                else:
                    delayed_t = shift_array(t, delay, 0)
                    delayed_y = shift_array(y, delay, initial_value)
                    return input_variable(delayed_t, delayed_y)

            return previous_value

        if 2 * delay > self._cache.maxsize:
            self._cache = LRUCache(maxsize=2 * delay)

        return TemporalVar(self.solver, (create_delay, self),
                           expression=f"#DELAY({delay}) {get_expression(self)}",
                           operator=operator_call,
                           call_mode=CallMode.CALL_FUN_RESULT,
                           is_discrete=True)

    def derivative(self, initial_value=0) -> "TemporalVar[T]":
        """
        Return the derivative of the Temporal Variable.

        Warning: Contrary to integration, this derivative method does not guarantee precision. Use it only as an escape
        hatch.
        :param initial_value: value at t=0
        :return: TemporalVar containing the derivative.
        """
        # Warn the user not to abuse the differentiate function
        warnings.warn("It is recommended to use 'integrate' instead of 'differentiate' for solving IVPs, "
                      "because the solver cannot guarantee precision when computing derivatives.\n"
                      "If you choose to use 'differentiate', consider using a smaller step size for better accuracy.",
                      category=UserWarning, stacklevel=2)

        previous = self.delayed(1)
        time_value = self.solver.time_variable
        previous_time = time_value.delayed(1)
        d_y = self - previous
        d_t = time_value - previous_time
        derived_value = temporal_var_where(self.solver, time_value != 0, np.divide(d_y, d_t, where=d_t != 0),
                                           initial_value)
        derived_value._expression = f"#D/DT {get_expression(self)}"
        return derived_value

    def m(self, method: Callable[P, T]) -> Callable[P, "TemporalVar[T]"]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> TemporalVar:
            inputs_expr = [get_expression(inp) if isinstance(inp, TemporalVar) else str(inp) for inp in args]
            kwargs_expr = [
                f"{key}={get_expression(value) if isinstance(value, TemporalVar) else str(value)}"
                for key, value in kwargs.items()
            ]
            expression = f"{method.__name__}({', '.join(inputs_expr)}"
            if kwargs_expr:
                expression += ", ".join(kwargs_expr)
            expression += ")"
            return TemporalVar(self.solver, (method, self, *args, kwargs),
                               expression=expression, operator=operator_call)

        functools.update_wrapper(wrapper, method)
        return wrapper

    def cross_trigger(self, value: Union["TemporalVar[T]", T],
                      direction: Literal["rising", "falling", "both"] = "both") -> "CrossTriggerVar":
        """
        Create a signal that triggers when the specified crossing occurs.

        :param value: Value to be crossed to cause the triggering.
        :param direction: Direction of the crossing.
        :return: TriggerVar
        """
        if self.output_type in (bool, np.bool, str):
            crossed_variable = self == value
            crossed_variable._expression = f"on {self.name} == {value.name if isinstance(value, TemporalVar) else value}"
        elif issubclass(self.output_type, abc.Iterable):
            raise ValueError(
                "Can not apply crossing detection to a variable containing a collection of values because it is ambiguous."
            )
        else:
            crossed_variable = self - value
            crossed_variable._expression = f"on {self.name} crossing {value.name if isinstance(value, TemporalVar) else value}"
        trigger_var = CrossTriggerVar(self.solver, crossed_variable, direction)
        return trigger_var

    def clear(self):
        self._values = None
        self._cache.clear()

    def _first_value(self):
        return self(0, self.solver.x0 if len(self.solver.x0) else 0)

    def _from_arg(self, value: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        """
        Return a TemporalVar from an argument value. If the argument is already a TemporalVar, return it. If not, create a TemporalVar from the value.
        """
        if isinstance(value, TemporalVar) and value.solver is self.solver:
            return value
        return TemporalVar(self.solver, value)

    def __copy__(self):
        return TemporalVar(self.solver, self.source, self.expression, operator=self.operator)

    def __add__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{get_expression(self)} + {get_expression(other)}"
        return TemporalVar(
            self.solver,
            (self, self._from_arg(other)),
            expression=expression,
            operator=operator.add
        )

    def __radd__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{get_expression(other)} + {get_expression(self)}"
        return TemporalVar(
            self.solver,
            (self._from_arg(other), self),
            expression=expression,
            operator=operator.add
        )

    def __sub__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = (
            f"{get_expression(self)} - {add_necessary_brackets(get_expression(other))}"
        )
        return TemporalVar(
            self.solver,
            (self, self._from_arg(other)),
            expression=expression,
            operator=operator.sub
        )

    def __rsub__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = (
            f"{get_expression(other)} - {add_necessary_brackets(get_expression(self))}"
        )
        return TemporalVar(
            self.solver,
            (self._from_arg(other), self),
            expression=expression,
            operator=operator.sub
        )

    def __mul__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(self))} * {add_necessary_brackets(get_expression(other))}"
        return TemporalVar(
            self.solver,
            (self, self._from_arg(other)),
            expression=expression,
            operator=operator.mul
        )

    def __rmul__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(other))} * {add_necessary_brackets(get_expression(self))}"
        return TemporalVar(
            self.solver,
            (self._from_arg(other), self),
            expression=expression,
            operator=operator.mul
        )

    def __truediv__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(self))} / {add_necessary_brackets(get_expression(other))}"
        return TemporalVar(
            self.solver,
            (self, self._from_arg(other)),
            expression=expression,
            operator=operator.truediv
        )

    def __rtruediv__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(other))} / {add_necessary_brackets(get_expression(self))}"
        return TemporalVar(
            self.solver,
            (self._from_arg(other), self),
            expression=expression,
            operator=operator.truediv
        )

    def __floordiv__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(self))} // {add_necessary_brackets(get_expression(other))}"
        return TemporalVar(
            self.solver,
            (self, self._from_arg(other)),
            expression=expression,
            operator=operator.floordiv
        )

    def __rfloordiv__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(other))} // {add_necessary_brackets(get_expression(self))}"
        return TemporalVar(
            self.solver,
            (self._from_arg(other), self),
            expression=expression,
            operator=operator.floordiv
        )

    def __mod__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(self))} % {add_necessary_brackets(get_expression(other))}"
        return TemporalVar(
            self.solver,
            (self, self._from_arg(other)),
            expression=expression,
            operator=operator.mod
        )

    def __rmod__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(other))} % {add_necessary_brackets(get_expression(self))}"
        return TemporalVar(
            self.solver,
            (self._from_arg(other), self),
            expression=expression,
            operator=operator.mod
        )

    def __pow__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(self))} ** {add_necessary_brackets(get_expression(other))}"
        return TemporalVar(
            self.solver,
            (self, self._from_arg(other)),
            expression=expression,
            operator=operator.pow
        )

    def __rpow__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(other))} ** {add_necessary_brackets(get_expression(self))}"
        return TemporalVar(
            self.solver,
            (self._from_arg(other), self),
            expression=expression,
            operator=operator.pow
        )

    def __pos__(self) -> "TemporalVar[T]":
        expression = f"+ {add_necessary_brackets(get_expression(self))}"
        return TemporalVar(
            self.solver,
            (self,),
            expression=expression,
            operator=operator.pos
        )

    def __neg__(self) -> "TemporalVar[T]":
        expression = f"- {add_necessary_brackets(get_expression(self))}"
        return TemporalVar(
            self.solver,
            (self,),
            expression=expression,
            operator=operator.neg
        )

    def __abs__(self) -> "TemporalVar[T]":
        expression = f"abs({get_expression(self)})"
        return TemporalVar(
            self.solver,
            (self,),
            expression=expression,
            operator=operator.abs
        )

    @overload
    def __eq__(self, other: "TemporalVar[NDArray[T]]") -> "TemporalVar[NDArray[bool]]":
        ...

    @overload
    def __eq__(self, other: Any) -> "TemporalVar[bool]":
        ...

    def __eq__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[bool]":
        expression = f"{add_necessary_brackets(get_expression(self))} == {add_necessary_brackets(get_expression(other))}"
        return TemporalVar(
            self.solver,
            (self, self._from_arg(other)),
            expression=expression,
            operator=operator.eq
        )

    @overload
    def __ne__(self, other: "TemporalVar[NDArray[T]]") -> "TemporalVar[NDArray[bool]]":
        ...

    @overload
    def __ne__(self, other: Any) -> "TemporalVar[bool]":
        ...

    def __ne__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[bool]":
        expression = f"{add_necessary_brackets(get_expression(self))} != {add_necessary_brackets(get_expression(other))}"
        return TemporalVar(
            self.solver,
            (self, self._from_arg(other)),
            expression=expression,
            operator=operator.ne
        )

    @overload
    def __lt__(self, other: "TemporalVar[NDArray[T]]") -> "TemporalVar[NDArray[bool]]":
        ...

    @overload
    def __lt__(self, other: Any) -> "TemporalVar[bool]":
        ...

    def __lt__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[bool]":
        expression = f"{add_necessary_brackets(get_expression(self))} < {add_necessary_brackets(get_expression(other))}"
        return TemporalVar(
            self.solver,
            (self, self._from_arg(other)),
            expression=expression,
            operator=operator.lt
        )

    @overload
    def __le__(self, other: "TemporalVar[NDArray[T]]") -> "TemporalVar[NDArray[bool]]":
        ...

    @overload
    def __le__(self, other: Any) -> "TemporalVar[bool]":
        ...

    def __le__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[bool]":
        expression = f"{add_necessary_brackets(get_expression(self))} <= {add_necessary_brackets(get_expression(other))}"
        return TemporalVar(
            self.solver,
            (self, self._from_arg(other)),
            expression=expression,
            operator=operator.le
        )

    @overload
    def __gt__(self, other: "TemporalVar[NDArray[T]]") -> "TemporalVar[NDArray[bool]]":
        ...

    @overload
    def __gt__(self, other: Any) -> "TemporalVar[bool]":
        ...

    def __gt__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[bool]":
        expression = f"{add_necessary_brackets(get_expression(self))} > {add_necessary_brackets(get_expression(other))}"
        return TemporalVar(
            self.solver,
            (self, self._from_arg(other)),
            expression=expression,
            operator=operator.gt
        )

    @overload
    def __ge__(self, other: "TemporalVar[NDArray[T]]") -> "TemporalVar[NDArray[bool]]":
        ...

    @overload
    def __ge__(self, other: Any) -> "TemporalVar[bool]":
        ...

    def __ge__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[bool]":
        expression = f"{add_necessary_brackets(get_expression(self))} >= {add_necessary_brackets(get_expression(other))}"
        return TemporalVar(
            self.solver,
            (self, self._from_arg(other)),
            expression=expression,
            operator=operator.ge
        )

    def __getitem__(self, item):
        expression = f"{add_necessary_brackets(get_expression(self))}[{item}]"
        return TemporalVar(
            self.solver,
            (self, item),
            expression=expression,
            operator=operator.getitem
        )

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
                (ufunc, *inputs, kwargs),
                expression=expression,
                operator=operator_call
            )

        return NotImplemented

    def __bool__(self):
        raise ValueError("The truth value of a Temporal Variable is ambiguous. Use vip.where() instead.")

    @property
    def expression(self):
        return self._expression

    def __repr__(self) -> str:
        if self.solver.solved:
            return f"{self.name or self._expression} = {self._values}"
        else:
            return f"{self._expression}"

    # NumPy arrays utility methods
    @property
    def shape(self):
        result = self._first_value()
        if isinstance(result, np.ndarray):
            return result.shape
        raise AttributeError("hhape attribute does not exist because this variable does not contain a NumPy array.")

    # Dict utility methods
    def keys(self):
        result = self._first_value()
        if isinstance(result, dict):
            return result.keys()
        raise AttributeError("keys() method does not exist because this variable does not contain a dict.")

    def items(self):
        result = self._first_value()
        if isinstance(result, dict):
            key_list = list(result.keys())
            value_list = [self[key] for key in key_list]
            return zip(key_list, value_list)
        raise AttributeError("items() method does not exist because this variable does not contain a dict.")


def convert_args_to_temporal_var(solver: Solver, arg_list: Iterable) -> List[TemporalVar]:
    def convert(arg):
        if not isinstance(arg, TemporalVar):
            arg = TemporalVar(solver, arg)
        return arg

    return [convert(a) for a in arg_list]


def temporal_var_where(solver, condition: TemporalVar[bool], a: Union[T, TemporalVar[T]],
                       b: Union[T, TemporalVar[T]]) -> \
        TemporalVar[T]:
    def where(condition, a, b):
        result = np.where(condition, a, b)
        if result.size == 1:
            result = result.item()
        return result

    return TemporalVar(
        solver,
        (where, condition, a, b),
        expression=f"({get_expression(a)} if {get_expression(condition)} else {get_expression(b)})",
        operator=operator_call
    )


class LoopNode(TemporalVar[T]):
    def __init__(self, solver: "Solver", shape: Union[int, tuple[int, ...]] = None, strict: bool = True):
        if shape is not None:
            initial_value = np.zeros(shape)
        else:
            initial_value = 0
        self._input_var_content = None
        super().__init__(solver, 0, expression="", child_cls=TemporalVar)
        self._input_var: TemporalVar = TemporalVar(solver, initial_value)
        self._is_set = False
        self._is_strict = strict

    def loop_into(self, value: Union[TemporalVar[T], T, List], force: bool = False):
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
        self.source = self._input_var.source
        self.operator = self._input_var.operator

    @property
    def _input_var(self):
        return self._input_var_content

    @_input_var.setter
    def _input_var(self, value: TemporalVar[T]):
        self._input_var_content = value
        # Overwrite attributes except cache size
        cache = self._cache
        vars(self).update(vars(self._input_var_content))
        self._cache = cache

    def is_valid(self) -> bool:
        """
        Check if the Loop Node is ready to be solved.
        If the Loop Node uses strict mode, its value must be set.
        :return: True if valid, False if incorrect
        """
        return not self._is_strict or self._is_set

    def action_set_to(self) -> None:
        """
        Not implemented
        :return: Raise an exception
        """
        raise NotImplementedError(
            "This method is forbidden for a Loop Node.\n"
            "If you want really want to change the behavior of a Loop Node, create a new variable by doing "
            "'new_var = 1 * loop_node'. You will be able to call `action_set_to()` on `new_var`."
        )


class IntegratedVar(TemporalVar[T]):
    def __init__(
            self,
            solver: "Solver",
            fun: Union[
                Callable[[Union[float, NDArray], NDArray], T], NDArray, dict
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

    def __getitem__(self, item) -> "IntegratedVar":
        return self.source[item]

    @property
    def y_idx(self):
        if isinstance(self.source, np.ndarray):
            return np.vectorize(lambda v: v.y_idx)(self.source)
        elif isinstance(self.source, dict):
            return {key: value.y_idx for key, value in self.source.items()}
        elif self._y_idx is not None:
            return self._y_idx
        raise ValueError("The argument 'y_idx' should be set for IntegratedVar containing a single value.")

    def reset_on(self, trigger: "CrossTriggerVar", new_value: Union[TemporalVar[T], T]) -> "Event":
        """
        Create an action that, when its event is triggered, reset the IntegratedVar output to the specified value.
        :param trigger: Variable that triggers the reset
        :param new_value: Value at which the integrator output is reset to
        :return: Event.
        """
        if not isinstance(new_value, TemporalVar):
            new_value = TemporalVar(self.solver, new_value)

        def action_fun(t, y):
            def set_y0(idx, subvalue):
                if isinstance(idx, np.ndarray):
                    for arr_idx in np.ndindex(idx.shape):
                        y_idx = idx[arr_idx]
                        set_y0(y_idx, new_value[y_idx])
                elif isinstance(idx, dict):
                    for key, idx in idx.items():
                        y[idx] = new_value[key]
                        set_y0(idx, new_value[key])
                else:
                    y[idx] = subvalue(t, y)

            set_y0(self.y_idx, new_value)

        event = trigger.event
        event.action = Action(action_fun, f"Reset {self.name} to {new_value.expression}")
        return event

    def action_set_to(self) -> None:
        """
        Not implemented
        :return: Raise an exception
        """
        raise NotImplementedError(
            "This method is forbidden for an integrated variable.\n"
            "If you want to reset the integrator output, use `action_reset_to()`.\n"
            "If you want really want to change the behavior of an integrated variable, create a new variable by doing "
            "'new_var = 1 * integrated_variable'. You will be able to call `action_set_to()` on `new_var`."
        )


class CrossTriggerVar(TemporalVar[bool]):
    _DIRECTION_MAP = {"rising": 1, "falling": -1, "both": 0}

    def __init__(self, solver: Solver, fun: TemporalVar, direction: Literal["rising", "falling", "both"] = "both"):
        super().__init__(solver)
        self.direction = self._DIRECTION_MAP[direction]
        self.function = fun

        self.t_triggers = []
        # Add to solver
        self.solver.cross_triggers.append(self)

    def __call__(self, t, y):
        if np.isscalar(t):
            return t in self.t_triggers
        else:
            return [self(t[i], y[i]) for i in range(len(t))]


def get_expression(value) -> str:
    if isinstance(value, TemporalVar):
        frame = inspect.currentframe().f_back.f_back
        while (
                "self" in frame.f_locals
                and (
                        isinstance(frame.f_locals["self"], TemporalVar)
                        or isinstance(frame.f_locals["self"], Solver)
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
    _DIRECTION_MAP = {"rising": 1, "falling": -1, "both": 0}
    _DIRECTION_REPR = {1: "rising", 0: "any direction", -1: "falling"}

    def __init__(self, solver: Solver, fun, action: Union["Action", Callable, None] = None,
                 direction: Literal["rising", "falling", "both"] = "both"):
        self.solver = solver
        self.function: TemporalVar = convert_args_to_temporal_var(self.solver, [fun])[0]
        self.action = convert_args_to_action([action])[0] if action is not None else None

        self.direction = self._DIRECTION_MAP[direction]

        self.count = 0

        self.g = None

        self.t_events = []

        self.deletion_time = None  # Time at which the event has been deleted from the simulation.

        self.solver.events.append(self)

    def __call__(self, t, y) -> float:
        return self.function(t, y)

    def __repr__(self):
        return (f"Event({self.function.expression} ({self._DIRECTION_REPR[self.direction]}), "
                f"{self.action or 'No action'})")

    def evaluate(self, t, y) -> None:
        self.g = self(t, y)

    def execute_action(self, t, y):
        if self.action is not None:
            self.action(t, y)

    def clear(self):
        self.count = 0
        self.g = None
        self.t_events = []
        self.deletion_time = None


class Action:
    def __init__(self, fun: Callable, expression: str = None):
        if isinstance(fun, TemporalVar):
            raise ValueError(
                "An action can not be a TemporalVar, because an action is a function with side effects, "
                "while a TemporalVar is a pure function."
            )
        if callable(fun):
            n_args = len(inspect.signature(fun).parameters)
            if n_args == 0:
                self.function = lambda t, y: fun()
            elif n_args == 1:
                self.function = lambda t, y: fun(t)
            else:
                self.function = lambda t, y: fun(t, y)
        self.expression = expression or convert_to_string(fun)

    def __call__(self, t, y):
        return self.function(t, y)

    def __add__(self, other: Union[Callable, "Action"]) -> "Action":
        if not isinstance(other, Action):
            other = Action(other)

        def added_fun(t, y):
            self(t, y)
            other(t, y)

        return Action(added_fun, f"{self.expression} + {other.expression}")

    def __repr__(self):
        return f"Action({self.expression})"


def convert_args_to_action(arg_list: Iterable) -> List[Action]:
    def convert(arg):
        if not isinstance(arg, Action):
            arg = Action(arg)
        return arg

    return [convert(a) for a in arg_list]
