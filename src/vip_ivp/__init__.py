from typing import ParamSpec

from varname import argname
from .utils import *
from .utils import _get_expression

warnings.simplefilter("once")

_solver_list = []

T = TypeVar('T')


def integrate(input_value: Union["TemporalVar[T]", Number], x0: Number) -> TemporalVar[float]:
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
    return LoopNode(solver)


def create_source(value: Union[Callable[[Union[float, np.ndarray]], T], T]) -> "TemporalVar[T]":
    """
    Create a source signal from a temporal function or a scalar value.

    :param value: A function f(t) or a scalar value.
    :return: The created TemporalVar.
    """
    solver = _get_current_solver()
    return utils.create_source(solver, value)


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


def explore(fun: Callable[..., T], t_end: Number, bounds=(), time_step: float = None, title: str = "") -> None:
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


def delay(input_value: TemporalVar[T], n_steps: int, initial_value: T = 0) -> TemporalVar[T]:
    if not isinstance(input_value, TemporalVar):
        raise Exception("Only TemporalVars can be delayed.")
    elif n_steps < 1:
        raise Exception("Delay accept only a positive step.")

    def previous_value(t, y):
        if np.isscalar(t):
            if len(input_value.solver.t) >= n_steps:
                previous_t = input_value.solver.t[-n_steps]
                previous_y = input_value.solver.y[-n_steps]

                return input_value(previous_t, previous_y)
            else:
                return initial_value
        else:
            delayed_t = shift_array(t, n_steps, 0)
            delayed_y = shift_array(y, n_steps, initial_value)
            return input_value(delayed_t, delayed_y)

    return TemporalVar(input_value.solver, previous_value,
                       expression=f"#DELAY({n_steps}) {_get_expression(input_value)}")


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
    derived_value._expression = f"#D/DT {_get_expression(input_value)}"
    return derived_value


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


P = ParamSpec("P")


def f(func: Callable[P, T]) -> Callable[P, TemporalVar[T]]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> TemporalVar:
        def content(t, y): return func(*[arg(t, y) if isinstance(arg, TemporalVar) else arg for arg in args],
                                       **{key: (arg(t, y) if isinstance(arg, TemporalVar) else arg) for key, arg in
                                          kwargs.items()})

        # Format input for the expression
        inputs_expr = [_get_expression(inp) if isinstance(inp, TemporalVar) else str(inp) for inp in args]
        kwargs_expr = [
            f"{key}={_get_expression(value) if isinstance(value, TemporalVar) else str(value)}"
            for key, value in kwargs.items()
        ]
        expression = f"{func.__name__}({', '.join(inputs_expr)}"
        if kwargs_expr:
            expression += ", ".join(kwargs_expr)
        expression += ")"

        return TemporalVar(_get_current_solver(), content,
                           expression=expression)

    functools.update_wrapper(wrapper, func)
    return wrapper


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
