import json
from typing import Dict, Union

from varname import argname
from typing_extensions import ParamSpec

from .base import *
from .utils import *

warnings.simplefilter("once")

_solver_list = []

T = TypeVar('T')
K = TypeVar("K")


@overload
def create_source(value: List[T]) -> TemporalVar[NDArray[T]]: ...


@overload
def create_source(value: Callable[[Union[float, NDArray]], T]) -> TemporalVar: ...


@overload
def create_source(value: int) -> TemporalVar[float]: ...


@overload
def create_source(value: T) -> TemporalVar[T]: ...


def create_source(value: Union[Callable[[Union[float, NDArray]], T], T]) -> TemporalVar[T]:
    """
    Create a source signal from a temporal function or a scalar value.

    :param value: A function f(t) or a scalar value.
    :return: The created TemporalVar.
    """
    solver = _get_current_solver()
    return TemporalVar(solver, value)


def create_scenario(scenario_table: Union["pd.DataFrame", str, dict], time_key: str, interpolation_kind="linear",
                    sep=',') -> TemporalVar:
    """
    Creates a scenario from a given input table, which can be in various formats such as CSV, JSON, dictionary, or DataFrame.
    The maps in the scenario table are interpolated over time and converted into TemporalVar objects.
    The function processes the data and returns a dictionary of TemporalVar objects.

    :param scenario_table: The input data, which can be one of the following formats:
        - A CSV file path (string)
        - A JSON file path (string)
        - A dictionary of data
        - A pandas DataFrame
    :type scenario_table: Union[pd.DataFrame, str, dict]

    :param time_key: The key (column) to use as time for the scenario.
    :type time_key: str

    :param interpolation_kind: The kind of interpolation to use. Default is "linear". This determines how values are
        interpolated between time points.
    :type interpolation_kind: str, optional

    :param sep: The separator to use when reading CSV files. Default is a comma.
    :type sep: str, optional

    :return: A dictionary of TemporalVar objects representing the scenario, where the keys are the variables and the
        values are the corresponding TemporalVar instances.
    :rtype: Dict[Any, TemporalVar]

    :raises ValueError: If the input file type is unsupported or the input type is invalid.
    """
    import pandas as pd

    solver = _get_current_solver()
    if isinstance(scenario_table, str):
        if scenario_table.endswith(".csv"):
            input_data = pd.read_csv(scenario_table, sep=sep)
            print(input_data)
            return TemporalVar.from_scenario(solver, input_data, time_key, interpolation_kind)
        elif scenario_table.endswith(".json"):
            with open(scenario_table, "r") as file:
                dict_data = json.load(file)
            input_data = pd.DataFrame(dict_data)
            return TemporalVar.from_scenario(solver, input_data, time_key, interpolation_kind)
        else:
            raise ValueError("Unsupported file type")
    elif isinstance(scenario_table, dict):
        input_data = pd.DataFrame(scenario_table)
        return TemporalVar.from_scenario(solver, input_data, time_key, interpolation_kind)
    elif isinstance(scenario_table, pd.DataFrame):
        return TemporalVar.from_scenario(solver, scenario_table, time_key, interpolation_kind)
    else:
        raise ValueError("Unsupported input type")


@overload
def integrate(input_value: Union[int, float], x0: Union[int, float], minimum: Union[TemporalVar[T], T, None] = None,
              maximum: Union[TemporalVar[T], T, None] = None) -> IntegratedVar[float]: ...


@overload
def integrate(input_value: Union[TemporalVar[T], T], x0: Union[T, List], minimum: Union[TemporalVar[T], T, None] = None,
              maximum: Union[TemporalVar[T], T, None] = None) -> IntegratedVar[T]: ...


def integrate(input_value: Union[TemporalVar[T], T], x0: Union[T, List], minimum: Union[TemporalVar[T], T, None] = None,
              maximum: Union[TemporalVar[T], T, None] = None) -> IntegratedVar[T]:
    """
    Integrate the input value starting from the initial condition x0.

    :param minimum: Lower integration bound. Can be a TemporalVar
    :param maximum: Higher integration bound. Can be a TemporalVar
    :param input_value: The value to be integrated, can be a TemporalVar or a number.
    :param x0: The initial condition for the integration.
    :return: The integrated TemporalVar.
    """
    solver = _get_current_solver()
    _check_solver_discrepancy(input_value, solver)
    integral_value = solver.integrate(input_value, x0, minimum, maximum)
    return integral_value


@overload
def loop_node(shape: None = None, strict: bool = True) -> LoopNode[float]: ...


@overload
def loop_node(shape: Union[int, tuple[int, ...]] = None, strict: bool = True) -> LoopNode[NDArray]: ...


def loop_node(shape: Union[int, tuple[int, ...]] = None, strict: bool = True) -> LoopNode:
    """
    Create a loop node. Loop node can accept new inputs through its "loop_into()" method after being instantiated.

    :param shape: Shape of the NumPy array contained in the Loop Node. If None, the Loop Node will contain a scalar.
    :param strict: Flag that triggers an error when the Loop Node has not been set before the solving.

    :return: The created LoopNode.
    """
    solver = _get_current_solver()
    return LoopNode(solver, shape, strict)


@overload
def where(condition: Union[TemporalVar[bool], bool], a: Action, b: Action) -> Action: ...


@overload
def where(condition: Union[TemporalVar[bool], bool], a: Union[TemporalVar, T], b: Union[TemporalVar, T]) -> TemporalVar[
    T]: ...


def where(condition, a, b):
    solver = _get_current_solver()
    if isinstance(a, Action) or isinstance(b, Action):
        return action_where(solver, condition, a, b)
    else:
        return temporal_var_where(solver, condition, a, b)


def delay(input_value: TemporalVar[T], n_steps: int, initial_value: T = 0) -> TemporalVar[T]:
    if not isinstance(input_value, TemporalVar):
        raise Exception("Only TemporalVars can be delayed.")
    elif n_steps < 1:
        raise Exception("Delay accept only a positive step.")

    def create_delay(input_variable):
        def previous_value(t, y):
            if np.isscalar(t):
                if len(input_variable.solver.t) >= n_steps:
                    previous_t = input_variable.solver.t[-n_steps]
                    previous_y = input_variable.solver.y[-n_steps]

                    return input_variable(previous_t, previous_y)
                else:
                    return initial_value
            else:
                delayed_t = shift_array(t, n_steps, 0)
                delayed_y = shift_array(y, n_steps, initial_value)
                return input_variable(delayed_t, delayed_y)

        return previous_value

    return TemporalVar(input_value.solver, (create_delay, input_value),
                       expression=f"#DELAY({n_steps}) {get_expression(input_value)}",
                       operator=operator_call,
                       call_mode=CallMode.CALL_FUN_RESULT)


def differentiate(input_value: TemporalVar[float], initial_value=0) -> TemporalVar[float]:
    # Warn the user not to abuse the differentiate function
    warnings.warn("It is recommended to use 'integrate' instead of 'differentiate' for solving IVPs, "
                  "because the solver cannot guarantee precision when computing derivatives.\n"
                  "If you choose to use 'differentiate', consider using a smaller step size for better accuracy.",
                  category=UserWarning, stacklevel=2)

    previous_value = delay(input_value, 1, initial_value)
    time_value = create_source(lambda t: t)
    previous_time = delay(time_value, 1)
    d_y = input_value - previous_value
    d_t = time_value - previous_time
    derived_value = np.divide(d_y, d_t, where=d_t != 0)
    derived_value._expression = f"#D/DT {get_expression(input_value)}"
    return derived_value


P = ParamSpec("P")


def f(func: Callable[P, T]) -> Callable[P, TemporalVar[T]]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> TemporalVar:
        inputs_expr = [get_expression(inp) if isinstance(inp, TemporalVar) else str(inp) for inp in args]
        kwargs_expr = [
            f"{key}={get_expression(value) if isinstance(value, TemporalVar) else str(value)}"
            for key, value in kwargs.items()
        ]
        expression = f"{func.__name__}({', '.join(inputs_expr)}"
        if kwargs_expr:
            expression += ", ".join(kwargs_expr)
        expression += ")"

        return TemporalVar(_get_current_solver(), (func, *args, kwargs),
                           expression=expression, operator=operator_call)

    functools.update_wrapper(wrapper, func)
    return wrapper


# Events

def get_events() -> List[Event]:
    solver = _get_current_solver()
    return solver.events


def _terminate():
    solver = _get_current_solver()
    solver.status = 1


terminate = Action(_terminate, "Terminate simulation")


def set_timeout(action: Union[Action, Callable], delay: float) -> Event:
    solver = _get_current_solver()
    current_time = solver.t_current
    time_variable = create_source(lambda t: t)
    time_variable.name = "Time"
    event = time_variable.on_crossing(current_time + delay, action)
    event.action += event.action_disable
    return event


def set_interval(action: Union[Action, Callable], delay: float) -> Event:
    solver = _get_current_solver()
    current_time = solver.t_current
    time_variable = create_source(lambda t: t)
    time_variable.name = f"Time % {delay}"

    def reset_timer(t_reset, y):
        time_variable.action_set_to(lambda t: t - t_reset)(t_reset, y)

    reset_timer_action = Action(reset_timer, "RESET TIMER")
    event = time_variable.on_crossing((current_time + delay), action + reset_timer_action)
    return event


# Solving

def solve(t_end: float, time_step: Union[float, None] = 0.1, method='RK45', t_eval: Union[List, NDArray] = None,
          include_events_times: bool = True, verbose: bool = False, **options) -> None:
    """
    Solve the equations of the dynamical system through an integration scheme.

    :param verbose: If True, print solving information to the console.
    :param include_events_times: If True, include time points at which events are triggered.
    :param t_end: Time at which the integration stops.
    :param method: Integration method to use. Default is 'RK45'.
    :param time_step: Time step for the integration. If None, use points selected by the solver.
    :param t_eval: Times at which to store the computed solution. If None, use points selected by the solver.
    :param options: Additional options for the solver.
    """
    solver = _get_current_solver()
    solver.solve(t_end, method, time_step, t_eval, include_events_times=include_events_times, verbose=verbose,
                 **options)


def explore(fun: Callable[..., T], t_end: float, bounds=(), time_step: float = None, title: str = "") -> None:
    """
    Explore the function f over the given bounds and solve the system until t_end.
    This function needs the sliderplot package.

    :param title: Title of the plot
    :param time_step: Time step of the simulation
    :param fun: The function to explore.
    :param t_end: Time at which the integration stops.
    :param bounds: Bounds for the exploration.
    """
    solver = _get_current_solver()
    solver.explore(fun, t_end, bounds, time_step, title)


def new_system() -> None:
    """
    Create a new solver system.
    """
    new_solver = Solver()
    _solver_list.append(new_solver)


AVAILABLE_EXPORT_FILE_FORMATS = ["csv", "json"]


def export_to_df(variables: Iterable[TemporalVar] = None) -> "pd.DataFrame":
    import pandas as pd

    solver = _get_current_solver()
    if not solver.solved:
        raise Exception("System must be solved before exporting the results. Please call 'vip.solve(t_end)'.")
    variables_dict = {"Time (s)": solver.t}
    if variables is None:
        variable_dict = {**variables_dict, **{key: var.values for key, var in solver.named_vars.items()}}
    else:
        variable_dict = {**variables_dict, **{get_expression(var): var.values for var in variables}}
    variable_dict["Time (s)"] = solver.t
    return pd.DataFrame(variable_dict)


def export_file(filename: str, variables: Iterable[TemporalVar] = None,
                file_format: Literal["csv", "json"] = None) -> None:
    if file_format is None:
        file_format = Path(filename).suffix.lstrip(".")
    if file_format not in AVAILABLE_EXPORT_FILE_FORMATS:
        raise ValueError(
            f"Unsupported file format: {file_format}. "
            f"The available file formats are {', '.join(AVAILABLE_EXPORT_FILE_FORMATS)}"
        )
    df = export_to_df(variables)
    if file_format == "csv":
        df.to_csv(filename, index=False)
    elif file_format == "json":
        df.to_json(filename, orient="records")


def clear() -> None:
    """
    Clear the current solver's stored information.
    """
    solver = _get_current_solver()
    solver.clear()


def save(*args: TemporalVar) -> None:
    """
    Save the given TemporalVars with their variable names.

    :param args: TemporalVars to be saved.
    :raises ValueError: If any of the arguments is not a TemporalVar.
    """
    solver = _get_current_solver()
    if not all([isinstance(arg, TemporalVar) for arg in args]):
        raise ValueError("Only TemporalVars can be saved.")
    for i, variable in enumerate(args):
        variable_name = argname(f'args[{i}]')
        solver.saved_vars[variable_name] = variable


def get_var(var_name: str) -> TemporalVar:
    """
    Retrieve a saved TemporalVar by its name.

    :param var_name: The name of the saved TemporalVar.
    :return: The retrieved TemporalVar.
    """
    solver = _get_current_solver()
    return solver.saved_vars[var_name]


def plot() -> None:
    """
    Plot the variables that have been marked for plotting.
    """
    solver = _get_current_solver()
    solver.plot()


def _get_current_solver() -> "Solver":
    if not _solver_list:
        new_system()
    return _solver_list[-1]


def _check_solver_discrepancy(input_value: Union["TemporalVar", float], solver: "Solver") -> None:
    """
    Raise an exception if there is a discrepancy between the input solver and the solver of the input variable.
    :param input_value:
    :param solver:
    """
    if isinstance(input_value, TemporalVar) and not solver is input_value.solver:
        raise Exception("Can not use a variable from a previous system.")


def _convert_to_temporal_var(value: Union[T, TemporalVar[T]]) -> TemporalVar[T]:
    if not isinstance(value, TemporalVar):
        value = create_source(value)
    return value
