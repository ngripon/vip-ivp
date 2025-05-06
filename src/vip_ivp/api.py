import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from .base import *
from .utils import *

warnings.simplefilter("once")

_solver_list = []

T = TypeVar('T')
K = TypeVar("K")


@overload
def temporal(value: List[T]) -> TemporalVar[NDArray[T]]: ...


@overload
def temporal(value: Callable[[Union[float, NDArray]], T]) -> TemporalVar: ...


@overload
def temporal(value: int) -> TemporalVar[float]: ...


@overload
def temporal(value: T) -> TemporalVar[T]: ...


def temporal(value: Union[Callable[[NDArray], T], T]) -> TemporalVar[T]:
    """
    Create a Temporal Variable from a temporal function, a scalar value, a dict, a list or a NumPy array.

    If the input value is a list, the variable content will be converted to a NumPy array. As a consequence, a nested
    list must represent a valid rectangular matrix.

    :param value: A function f(t), a scalar value, a dict, a list or a NumPy array.
    :return: The created TemporalVar.
    """
    solver = _get_current_solver()
    return TemporalVar(solver, value)


def create_scenario(scenario_table: Union["pd.DataFrame", str, dict], time_key: str, interpolation_kind: str = "linear",
                    sep: str = ',') -> TemporalVar:
    """
    Creates a scenario from a given input table, which can be in various formats such as CSV, JSON, dictionary, or DataFrame.

    The maps in the scenario table are interpolated over time and converted into TemporalVar objects.
    The function processes the data and returns a TemporalVar containing a dictionary of TemporalVar objects.

    :param scenario_table: The input data, which can be one of the following formats:
        - A CSV file path (string)
        - A JSON file path (string)
        - A dictionary of data
        - A pandas DataFrame
    :param time_key: The key (column) to use as time for the scenario.
    :param interpolation_kind: Specifies the kind of interpolation as a string or as an integer specifying the order of
        the spline interpolator to use. The string has to be one of ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’,
        ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’. ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline
        interpolation of zeroth, first, second or third order; ‘previous’ and ‘next’ simply return the previous or next
        value of the point; ‘nearest-up’ and ‘nearest’ differ when interpolating half-integers (e.g. 0.5, 1.5) in that
        ‘nearest-up’ rounds up and ‘nearest’ rounds down. Default is ‘linear’.
    :param sep: The separator to use when reading CSV files. Default is a comma.
    :return: A dictionary of TemporalVar objects representing the scenario, where the keys are the variables and the values are the corresponding TemporalVar instances.

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

    The integrated output can be bounded with the **minimum** and **maximum** arguments.

    :param input_value: The value to be integrated, can be a TemporalVar or a number.
    :param x0: The initial condition for the integration.
    :param minimum: Lower integration bound. Can be a TemporalVar
    :param maximum: Higher integration bound. Can be a TemporalVar
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
    Create a Loop Node. A Loop Node can accept new inputs through its "loop_into()" method after being instantiated.

    :param shape: Shape of the NumPy array contained in the Loop Node. If None, the Loop Node will contain a scalar.
    :param strict: Flag that triggers an error when the Loop Node has not been set before the solving.

    :return: The created LoopNode.
    """
    solver = _get_current_solver()
    return LoopNode(solver, shape, strict)


@overload
def where(condition: Union[TemporalVar[bool], bool], a: Union[TemporalVar, T], b: Union[TemporalVar, T]) -> TemporalVar[
    T]: ...


def where(condition: Union[TemporalVar[bool], bool], a: Union[TemporalVar, T],
          b: Union[TemporalVar, T]) -> TemporalVar[T]:
    """
    Create a conditional TemporalVar or a conditional Action.
    If condition is `True` at time $t$, the output value will have value **a**, else **b**.

    If **a** and **b** are TemporalVars or scalars, the output will be a TemporalVar.

    If **a** and **b** are Actions, the output will be an Action.

    :param condition: Condition to evaluate through time.
    :param a: Output value or Action to execute if the condition is `True` at time $t$
    :param b: Output value or Action to execute if the condition is `False` at time $t$
    :return: Conditional TemporalVar or conditional Action.
    """
    solver = _get_current_solver()
    return temporal_var_where(solver, condition, a, b)


P = ParamSpec("P")


def f(func: Callable[P, T]) -> Callable[P, TemporalVar[T]]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> TemporalVar:
        inputs_expr = [get_expression(inp) if isinstance(inp, TemporalVar) else str(inp) for inp in args]
        kwargs_expr = [
            f"{key}={get_expression(value) if isinstance(value, TemporalVar) else str(value)}"
            for key, value in kwargs.items()
        ]
        if inspect.isclass(func):
            expression = f"{func.__class__.__name__}({', '.join(inputs_expr)}"
        elif inspect.isfunction(func):
            expression = f"{func.__name__}({', '.join(inputs_expr)}"
        else:
            expression = f"{repr(func)}({', '.join(inputs_expr)}"
        if kwargs_expr:
            expression += ", ".join(kwargs_expr)
        expression += ")"

        return TemporalVar(_get_current_solver(), (func, *args, kwargs),
                           expression=expression, operator=operator_call)

    functools.update_wrapper(wrapper, func)
    return wrapper


def get_time_variable() -> TemporalVar[float]:
    """
    Return the time variable of the system.
    :return: Time variable that returns the time value of each simulation step.
    """
    solver = _get_current_solver()
    return solver.time_variable


# Events

def terminate_on(call_signal: TriggerType) -> Event:
    solver = _get_current_solver()

    def action_terminate():
        solver.status = 1

    action = Action(action_terminate, "Terminate simulation")
    event = Event(solver, call_signal, action)
    return event


def timeout_trigger(delay: float) -> CrossTriggerVar:
    solver = _get_current_solver()
    current_time = solver.t_current
    time_variable = get_time_variable()
    trigger = time_variable.cross_trigger(current_time + delay)
    return trigger


def interval_trigger(delay: float) -> CrossTriggerVar:
    solver = _get_current_solver()
    current_time = solver.t_current
    time_variable = copy(get_time_variable())
    # Convert to a sine wave of period delay
    periodic_sine = where(time_variable < current_time, 0, np.sin(2 * np.pi * (time_variable - current_time) / delay))
    trigger = periodic_sine.cross_trigger(0)
    return trigger


# Solving

def solve(t_end: float, time_step: Union[float, None] = 0.1, method='RK45', t_eval: Union[List, NDArray] = None,
          include_crossing_times: bool = True, plot: bool = True, rtol: float = 1e-3,
          atol: float = 1e-6, max_step=np.inf, verbose: bool = False) -> None:
    """
    Solve the equations of the dynamical system through a hybrid solver.

    The hybrid solver is a modified version of SciPy's solve_ivp() function.

    :param max_step: Maximum allowed step size. Default is np.inf, i.e., the step size is not bounded and determined
        solely by the solver.
    :param plot: If True, a plot will show the result of the simulation for variables that were registered to plot.
    :param verbose: If True, print solving information to the console.
    :param include_crossing_times: If True, include time points at which events are triggered.
    :param t_end: Time at which the integration stops.
    :param method: Integration method to use. Default is 'RK45'. For a list of available methods, see SciPy's
        `solve_ivp()` documentation.
    :param time_step: Time step for the integration. If None, use points selected by the solver.
    :param t_eval: Times at which to store the computed solution. If None, use points selected by the solver.
    :param rtol: Relative tolerance. The solver keeps the local error estimates less than `atol + rtol * abs(y)`.
    :param atol: Absolute tolerance. The solver keeps the local error estimates less than `atol + rtol * abs(y)`.
    """
    solver = _get_current_solver()
    solver.solve(t_end, method, time_step, t_eval, include_crossing_times=include_crossing_times, verbose=verbose,
                 plot=plot, rtol=rtol, atol=atol, max_step=max_step)


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


def export_to_df(*variables: TemporalVar) -> "pd.DataFrame":
    import pandas as pd

    solver = _get_current_solver()
    if not solver.solved:
        raise Exception("System must be solved before exporting the results. Please call 'vip.solve(t_end)'.")
    variables_dict = {"Time (s)": solver.t}
    if not variables:
        variable_dict = {**variables_dict, **{key: var.values for key, var in solver.named_vars.items()}}
    else:
        variable_dict = {**variables_dict, **{get_expression(var): var.values for var in variables}}
    variable_dict["Time (s)"] = solver.t
    return pd.DataFrame(variable_dict)


def export_file(filename: str, variable_list: Iterable[TemporalVar] = None,
                file_format: Literal["csv", "json"] = None) -> None:
    if file_format is None:
        file_format = Path(filename).suffix.lstrip(".")
    if file_format not in AVAILABLE_EXPORT_FILE_FORMATS:
        raise ValueError(
            f"Unsupported file format: {file_format}. "
            f"The available file formats are {', '.join(AVAILABLE_EXPORT_FILE_FORMATS)}"
        )
    df = export_to_df(*variable_list)
    if file_format == "csv":
        df.to_csv(filename, index=False)
    elif file_format == "json":
        df.to_json(filename, orient="records")


def clear() -> None:
    """
    Clear the current solver's stored state.
    Use it to free memory before creating a new simulation.
    """
    solver = _get_current_solver()
    solver.clear()


### I do not like the current implementation, so I disable it for the moment.
#
# def save(*args: TemporalVar) -> None:
#     """
#     Save the given TemporalVars with their variable names.
#
#     :param args: TemporalVars to be saved.
#     :raises ValueError: If any of the arguments is not a TemporalVar.
#     """
#     solver = _get_current_solver()
#     if not all([isinstance(arg, TemporalVar) for arg in args]):
#         raise ValueError("Only TemporalVars can be saved.")
#     for i, variable in enumerate(args):
#         variable_name = argname(f'args[{i}]')
#         solver.saved_vars[variable_name] = variable


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
        value = temporal(value)
    return value
