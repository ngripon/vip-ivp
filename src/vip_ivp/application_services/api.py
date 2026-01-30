import functools
import inspect
from typing import Callable

from numpy.typing import NDArray

from .system import IVPSystemMutable, TemporalVar, IntegratedVar
from .variables import temporal_var_where, P, T, CrossTriggerVar
from ..utils import operator_call

_solver_list: list[IVPSystemMutable] = []


def new_system() -> None:
    _solver_list.append(IVPSystemMutable())


def temporal(value) -> TemporalVar:
    return TemporalVar(value, system=_get_current_system())


def state(x0: float) -> IntegratedVar:
    return _get_current_system().add_state(x0)


def where(condition, a, b) -> TemporalVar:
    return temporal_var_where(condition, a, b)


def f(func: Callable[P, T]) -> Callable[P, TemporalVar[T]]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> TemporalVar:
        return TemporalVar((func, *args, kwargs), operator_call, _get_current_system())

    functools.update_wrapper(wrapper, func)
    return wrapper


def solve(t_end: float, method: str = "RK45", t_eval: list[float] | NDArray = None) -> None:
    _get_current_system().solve(t_end, method, t_eval)


def when(condition:CrossTriggerVar, action)->None:
    _get_current_system().add_event(condition, action)


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


def _get_current_system() -> IVPSystemMutable:
    if not _solver_list:
        new_system()
    return _solver_list[-1]
