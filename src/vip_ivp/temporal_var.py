import warnings
import inspect
from typing import TYPE_CHECKING, Dict, Any

from numbers import Number
from pathlib import Path
from typing import Callable, Union, TypeVar, Generic

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .utils import add_necessary_brackets, convert_to_string

if TYPE_CHECKING:
    from solver import Solver

T = TypeVar('T')


class TemporalVar(Generic[T]):
    def __init__(self, solver: "Solver", fun: Callable[[Union[float, np.ndarray], np.ndarray], T] = None,
                 expression: str = None):
        self.solver = solver
        self.init = None
        if isinstance(fun, Callable):
            self.function = fun
        else:
            self.function = lambda t, y: fun
        self._values = None
        # Variable definition
        self._expression = inspect.getsource(
            fun) if expression is None else expression
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
        self.solver.vars_to_plot[name] = self

    def _reset(self):
        self._values = None

    def set_init(self, x0: Union[Number, np.ndarray]):
        self.init = x0
        self.solver.initialized_vars.append(self)

    def __call__(self, t: Union[float, np.ndarray], y: np.ndarray) -> T:
        return self.function(t, y)

    def __add__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{get_expression(self)} + {get_expression(other)}"
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: self(t, y) + other(t, y), expression=expression)
        else:
            return TemporalVar(self.solver, lambda t, y: self(t, y) + other, expression=expression)

    def __radd__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{get_expression(other)} + {get_expression(self)}"
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: other(t, y) + self(t, y), expression=expression)
        else:
            return TemporalVar(self.solver, lambda t, y: other + self(t, y), expression=expression)

    def __sub__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{get_expression(self)} - {add_necessary_brackets(get_expression(other))}"
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: self(t, y) - other(t, y), expression=expression)
        else:
            return TemporalVar(self.solver, lambda t, y: self(t, y) - other, expression=expression)

    def __rsub__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{get_expression(other)} - {add_necessary_brackets(get_expression(self))}"
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: other(t, y) - self(t, y), expression=expression)
        else:
            return TemporalVar(self.solver, lambda t, y: other - self(t, y), expression=expression)

    def __mul__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(self))} * {add_necessary_brackets(get_expression(other))}"
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: self(t, y) * other(t, y), expression=expression)
        else:
            return TemporalVar(self.solver, lambda t, y: other * self(t, y), expression=expression)

    def __rmul__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(other))} * {add_necessary_brackets(get_expression(self))}"
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: self(t, y) * other(t, y), expression=expression)
        else:
            return TemporalVar(self.solver, lambda t, y: other * self(t, y), expression=expression)

    def __truediv__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(self))} / {add_necessary_brackets(get_expression(other))}"
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: self(t, y) / other(t, y), expression=expression)
        else:
            return TemporalVar(self.solver, lambda t, y: self(t, y) / other, expression=expression)

    def __rtruediv__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(other))} / {add_necessary_brackets(get_expression(self))}"
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: other(t, y) / self(t, y), expression=expression)
        else:
            return TemporalVar(self.solver, lambda t, y: other / self(t, y), expression=expression)

    def __floordiv__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(self))} // {add_necessary_brackets(get_expression(other))}"
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: self(t, y) // other(t, y), expression=expression)
        else:
            return TemporalVar(self.solver, lambda t, y: self(t, y) // other, expression=expression)

    def __rfloordiv__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(other))} // {add_necessary_brackets(get_expression(self))}"
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: other(t, y) // self(t, y), expression=expression)
        else:
            return TemporalVar(self.solver, lambda t, y: other // self(t, y), expression=expression)

    def __mod__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(self))} % {add_necessary_brackets(get_expression(other))}"
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: self(t, y) % other(t, y), expression=expression)
        else:
            return TemporalVar(self.solver, lambda t, y: self(t, y) % other, expression=expression)

    def __rmod__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(other))} % {add_necessary_brackets(get_expression(self))}"
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: other(t, y) % self(t, y), expression=expression)
        else:
            return TemporalVar(self.solver, lambda t, y: other % self(t, y), expression=expression)

    def __pow__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(self))} ** {add_necessary_brackets(get_expression(other))}"
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: self(t, y) ** other(t, y), expression=expression)
        else:
            return TemporalVar(self.solver, lambda t, y: self(t, y) ** other, expression=expression)

    def __rpow__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[T]":
        expression = f"{add_necessary_brackets(get_expression(other))} ** {add_necessary_brackets(get_expression(self))}"
        if isinstance(other, TemporalVar):
            return TemporalVar(self.solver, lambda t, y: other(t, y) ** self(t, y), expression=expression)
        else:
            return TemporalVar(self.solver, lambda t, y: other ** self(t, y), expression=expression)

    def __pos__(self) -> "TemporalVar[T]":
        return self

    def __neg__(self) -> "TemporalVar[T]":
        expression = f"-{add_necessary_brackets(get_expression(self))}"
        return TemporalVar(self.solver, lambda t, y: -self(t, y), expression=expression)

    def __abs__(self) -> "TemporalVar[T]":
        expression = f"abs({get_expression(self)})"
        return TemporalVar(self.solver, lambda t, y: abs(self(t, y)), expression=expression)

    def __eq__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[bool]":
        expression = f"{add_necessary_brackets(get_expression(self))} == {add_necessary_brackets(get_expression(other))}"
        return TemporalVar(
            self.solver,
            lambda t, y: self(t, y) == (
                other(t, y) if isinstance(other, TemporalVar) else other),
            expression=expression
        )

    def __ne__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[bool]":
        expression = f"{add_necessary_brackets(get_expression(self))} != {add_necessary_brackets(get_expression(other))}"
        return TemporalVar(
            self.solver,
            lambda t, y: self(t, y) != (
                other(t, y) if isinstance(other, TemporalVar) else other),
            expression=expression
        )

    def __lt__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[bool]":
        expression = f"{add_necessary_brackets(get_expression(self))} < {add_necessary_brackets(get_expression(other))}"
        return TemporalVar(
            self.solver,
            lambda t, y: self(t, y) < (
                other(t, y) if isinstance(other, TemporalVar) else other),
            expression=expression
        )

    def __le__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[bool]":
        expression = f"{add_necessary_brackets(get_expression(self))} <= {add_necessary_brackets(get_expression(other))}"
        return TemporalVar(
            self.solver,
            lambda t, y: self(t, y) <= (
                other(t, y) if isinstance(other, TemporalVar) else other),
            expression=expression
        )

    def __gt__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[bool]":
        expression = f"{add_necessary_brackets(get_expression(self))} > {add_necessary_brackets(get_expression(other))}"
        return TemporalVar(
            self.solver,
            lambda t, y: self(t, y) > (
                other(t, y) if isinstance(other, TemporalVar) else other),
            expression=expression
        )

    def __ge__(self, other: Union["TemporalVar[T]", T]) -> "TemporalVar[bool]":
        expression = f"{add_necessary_brackets(get_expression(self))} >= {add_necessary_brackets(get_expression(other))}"
        return TemporalVar(
            self.solver,
            lambda t, y: self(t, y) >= (
                other(t, y) if isinstance(other, TemporalVar) else other),
            expression=expression
        )

    def __getitem__(self, item):
        expression = f"{add_necessary_brackets(get_expression(self))}[{item}]"
        return TemporalVar(
            self.solver,
            lambda t, y: self(t, y)[item] if np.isscalar(t) else np.array([x[item] for x in self(t, y)]),
            expression=expression)

    def __getattr__(self, item):
        expression = f"{add_necessary_brackets(get_expression(self))}.{item}"
        return TemporalVar(
            self.solver, lambda t, y: getattr(self(t, y), item) if np.isscalar(t) else np.array(
                [getattr(x, item) for x in self(t, y)]),
            expression=expression)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> "TemporalVar":
        inputs_expr = [get_expression(inp) if isinstance(inp, TemporalVar) else str(inp) for inp in inputs]
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
                    *[inp(t, y) if isinstance(inp, TemporalVar)
                      else inp for inp in inputs],
                    **{
                        key: (value(t, y) if isinstance(
                            value, TemporalVar) else value)
                        for key, value in kwargs.items()
                    }
                ),
                expression=expression
            )

        return NotImplemented

    def __array__(self) -> np.ndarray:
        return self.values

    @property
    def expression(self):
        return self._expression

    def __repr__(self) -> str:
        if self.solver.solved:
            return f"{self.values}"
        else:
            return f"{self._expression}"


class LoopNode(TemporalVar[T]):
    def __init__(self, solver: "Solver"):
        self._input_vars: list[TemporalVar] = []
        super().__init__(solver, lambda t, y: 0, expression="")
        self._is_set = False

    def loop_into(
            self,
            value: Union[TemporalVar[T], T],
            force: bool = False
    ):
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
            value = create_source(self.solver, value)
        self._input_vars.append(value)
        self._is_set = True
        self._expression = " + ".join(get_expression(var)
                                      for var in self._input_vars)

    def __call__(self, t: Union[float, np.ndarray], y: np.ndarray) -> T:
        return np.sum(var(t, y) for var in self._input_vars)


def create_source(solver: "Solver", value: Union[Callable[[Union[float, np.ndarray]], T], T]) -> "TemporalVar[T]":
    """
    Create a source signal from a temporal function or a scalar value.

    :param solver: Solver
    :param value: A function f(t) or a scalar value.
    :return: The created TemporalVar.
    """
    expression = convert_to_string(value)
    if callable(value):
        return TemporalVar(solver, lambda t, y: value(t), expression=expression)
    else:
        if np.isscalar(value):
            return TemporalVar(solver, lambda t, y: value if np.isscalar(t) else np.full_like(t, value),
                               expression=expression)
        else:
            return TemporalVar(solver,
                               lambda t, y: value if np.isscalar(t) else np.array([value for _ in range(len(t))]),
                               expression=expression)


def create_scenario(solver: "Solver", scenario_table: pd.DataFrame, time_key: str, interpolation_kind="linear") -> Dict[
    Any, TemporalVar]:
    variables = {}
    for col in scenario_table.columns:
        if col == time_key:
            continue
        fun = interp1d(scenario_table[time_key], scenario_table[col], kind=interpolation_kind, bounds_error=False,
                       fill_value=(scenario_table[col].iat[0], scenario_table[col].iat[-1]))
        variable = create_source(solver, fun)
        variables[col] = variable
    return variables


def get_expression(value) -> str:
    if isinstance(value, TemporalVar):
        frame = inspect.currentframe().f_back.f_back
        if Path(frame.f_code.co_filename).as_posix().endswith("vip_ivp/api.py"):
            frame = frame.f_back
        while frame.f_locals.get("self") and isinstance(frame.f_locals.get("self"), TemporalVar):
            frame = frame.f_back
        found_key = next(
            (key for key, dict_value in frame.f_locals.items() if dict_value is value), None)
        if found_key is not None:
            value.name = found_key
            return value.name
        return value.expression
    else:
        return str(value)
