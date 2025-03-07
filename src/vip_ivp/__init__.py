from varname import argname
from .utils import *

_solver_list = []


def integrate(input_value: Union["TemporalVar", Number], x0: Number) -> "TemporalVar":
    """
    Integrate the input value starting from the initial condition x0.
    
    :param input_value: The value to be integrated, can be a TemporalVar or a number.
    :param x0: The initial condition for the integration.
    :return: The integrated TemporalVar.
    """
    solver = _get_current_solver()
    _check_solver_discrepancy(input_value, solver)
    integral_value = solver.integrate(input_value, x0)
    return integral_value


def loop_node() -> "LoopNode":
    """
    Create a loop node. Loop node can accept new inputs through its "loop_into()" method after being instantiated.

    :return: The created LoopNode.
    """
    solver = _get_current_solver()
    loop_node = solver.loop_node()
    return loop_node


def create_source(value: Union[Callable, Number]) -> "TemporalVar":
    """
    Create a source signal from a temporal function or a scalar value.
    
    :param value: A function f(t) or a scalar value.
    :return: The created TemporalVar.
    """
    solver = _get_current_solver()
    source = solver.create_source(value)
    return source


def solve(t_end: Number, method='RK45', time_step=None, t_eval=None, **options) -> None:
    """
    Solve the equations of the dynamical system through an integration scheme.
    
    :param t_end: Time at which the integration stops.
    :param method: Integration method to use. Default is 'RK45'.
    :param time_step: Time step for the integration. If None, use points selected by the solver.
    :param t_eval: Times at which to store the computed solution. If None, use points selected by the solver.
    :param options: Additional options for the solver.
    """
    solver = _get_current_solver()
    solver.solve(t_end, method, time_step, t_eval, **options)


def explore(f: Callable, t_end: Number, bounds=(), time_step: float = None, title: str = "") -> None:
    """
    Explore the function f over the given bounds and solve the system until t_end.
    This function needs the sliderplot package.

    :param title: Title of the plot
    :param time_step: Time step of the simulation
    :param f: The function to explore.
    :param t_end: Time at which the integration stops.
    :param bounds: Bounds for the exploration.
    """
    solver = _get_current_solver()
    solver.explore(f, t_end, bounds, time_step, title)


def new_system() -> None:
    """
    Create a new solver system.
    """
    new_solver = Solver()
    _solver_list.append(new_solver)


def clear() -> None:
    """
    Clear the current solver's stored information.
    """
    solver = _get_current_solver()
    solver.clear()


def save(*args) -> None:
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


def _check_solver_discrepancy(input_value: Union["TemporalVar", Number], solver: "Solver") -> None:
    """
    Raise an exception if there is a discrepancy between the input solver and the solver of the input variable.
    :param input_value:
    :param solver:
    """
    if isinstance(input_value, TemporalVar) and not solver is input_value.solver:
        raise Exception("Can not use a variable from a previous system.")
