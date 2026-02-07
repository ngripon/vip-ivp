import json
from pathlib import Path
from typing import Iterable, Literal

from .variables import *
from .system import *
from ..utils import operator_call

if TYPE_CHECKING:
    import pandas as pd

AVAILABLE_EXPORT_FILE_FORMATS = ["csv", "json"]

_solver_list: list[IVPSystemMutable] = []


def new_system() -> None:
    _solver_list.append(IVPSystemMutable())


def temporal(value) -> TemporalVar:
    return TemporalVar(value, system=_get_current_system())


def state(x0: float, lower_bound: float | TemporalVar = None, upper_bound: float | TemporalVar = None) -> IntegratedVar:
    return _get_current_system().add_state(x0, lower_bound, upper_bound)


def n_order_state(
        *initial_conditions: float
) -> tuple[IntegratedVar, ...]:
    system = _get_current_system()
    states = [system.add_state(x0) for x0 in initial_conditions]

    for s, ds in zip(states[:-1], states[1:]):
        s.derivative = ds

    return tuple(states)


def create_scenario(scenario_table: str | dict | pd.DataFrame, time_key: str, interpolation_kind: str = "linear",
                    sep: str = ',') -> TemporalVar[dict]:
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

    solver = _get_current_system()
    if isinstance(scenario_table, str):
        if scenario_table.endswith(".csv"):
            input_data = pd.read_csv(scenario_table, sep=sep)
        elif scenario_table.endswith(".json"):
            with open(scenario_table, "r") as file:
                dict_data = json.load(file)
            input_data = pd.DataFrame(dict_data)
        else:
            raise ValueError("Unsupported file type")
    elif isinstance(scenario_table, dict):
        input_data = pd.DataFrame(scenario_table)
    elif isinstance(scenario_table, pd.DataFrame):
        input_data = scenario_table
    else:
        raise ValueError("Unsupported input type")
    return TemporalVar.from_scenario(input_data, time_key, solver, interpolation_kind)


def where(condition, a, b) -> TemporalVar:
    return temporal_var_where(condition, a, b)


def f(func: Callable[P, T]) -> Callable[P, TemporalVar[T]]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> TemporalVar:
        return TemporalVar((func, *args, kwargs), operator_call, _get_current_system())

    functools.update_wrapper(wrapper, func)
    return wrapper


def solve(t_end: float, method: str = "RK45", t_eval: list[float] | NDArray = None, step_eval: float = None) -> None:
    _get_current_system().solve(t_end, method, t_eval, step_eval)


def when(condition: CrossTriggerVar | TemporalVar[bool], action: Action | SideEffectFun) -> None:
    _get_current_system().add_event(condition, action)


terminate = Action(None, ActionType.TERMINATE)


# Post-processing
def plot(*variables: TemporalVar) -> None:
    # Check
    if not variables:
        raise ValueError("No variable provided")
    system = _get_current_system()
    if not system.is_solved:
        raise RuntimeError("System is not solved")

    # Try to infer names. This is brittle and may fail silently in some contexts
    frame = inspect.currentframe().f_back
    locals_ = frame.f_locals

    def infer_name(obj):
        for name, val in locals_.items():
            if val is obj:
                return name
        return None

    # Plot
    import matplotlib.pyplot as plt

    timestamps = system.t_eval
    # Plot data
    plt.figure("Results")
    for idx, variable in enumerate(variables):
        plt.plot(timestamps, variable.values, label=infer_name(variable) or f"var_{idx}")
    # Label and axis
    plt.title("Simulation results")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.xlim(timestamps[0], timestamps[-1])
    plt.grid()
    plt.tight_layout()
    plt.show()


def export_to_df(*variables: TemporalVar) -> "pd.DataFrame":
    # Try to infer names. This is brittle and may fail silently in some contexts
    frame = inspect.currentframe().f_back
    locals_ = frame.f_locals

    def infer_name(obj):
        for name, val in locals_.items():
            if val is obj:
                return name
        return None


    import pandas as pd

    solver = _get_current_system()
    if not solver.is_solved:
        raise Exception("System must be solved before exporting the results. Please call 'vip.solve(t_end)'.")
    variables_dict = {"Time (s)": solver.t_eval}
    variable_dict = {**variables_dict, **{infer_name(var): var.values for var in variables}}

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

# Utils

def _get_current_system() -> IVPSystemMutable:
    if not _solver_list:
        new_system()
    return _solver_list[-1]
