from .utils import *

_solver_list = []


def integrate(input_value: Union["TemporalVar", Number], x0: Number) -> "TemporalVar":
    solver = _get_current_solver()
    _check_solver_discrepancy(input_value, solver)
    integral_value = solver.integrate(input_value, x0)
    return integral_value


def loop_node(input_value: Union["TemporalVar", Number]) -> "LoopNode":
    solver = _get_current_solver()
    _check_solver_discrepancy(input_value, solver)
    loop_node = solver.loop_node(input_value)
    return loop_node


def create_source(value: Union[Callable, Number]) -> "TemporalVar":
    solver = _get_current_solver()
    source = solver.create_source(value)
    return source


def solve(t_end: Number, method='RK45', time_step=None, t_eval=None, **options) -> None:
    solver = _get_current_solver()
    solver.solve(t_end, method, time_step, t_eval, **options)


def explore(f: Callable, t_end: Number, bounds=()) -> None:
    solver = _get_current_solver()
    solver.explore(f, t_end, bounds)


def new_system() -> None:
    new_solver = Solver()
    _solver_list.append(new_solver)


def clear() -> None:
    solver = _get_current_solver()
    solver.clear()


def _get_current_solver() -> "Solver":
    if not _solver_list:
        new_system()
    return _solver_list[-1]


def _check_solver_discrepancy(input_value: Union["TemporalVar", Number], solver: "Solver") -> None:
    """
    Raise an exception if there is a discrepancy between the input solver and the solver of the input variable.
    :param input_value:
    :param solver:
    """
    if isinstance(input_value, TemporalVar) and not solver is input_value.solver:
        raise Exception("Can not use a variable from a previous system.")
