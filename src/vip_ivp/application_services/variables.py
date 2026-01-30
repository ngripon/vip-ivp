"""
This module contain the TemporalVar value object.

In a IVP system, variables can be defined by a function f(t, y):
    - t is the time vector. Dims: len(t)
    - y is the system solution. Dims: (len(integrated_vars) x len(t))

TemporalVar is a container for this function. This enables laziness: the only stored data is the system solution. The
results of each variable are computed only when needed.
"""

import functools
import inspect
import operator

from typing import Callable, TypeVar, Generic, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import ParamSpec

from ..domain.system import create_system_output_fun, Direction, Action, create_set_system_output_fun, ActionType
from .utils import operator_call, vectorize_source, get_output_info

if TYPE_CHECKING:
    from .system import IVPSystemMutable

T = TypeVar("T")
P = ParamSpec("P")

Source = Callable[[float | NDArray, NDArray], T] | Callable[[float | NDArray], T] | NDArray | dict | float


class TemporalVar(Generic[T]):
    """
    Wrapper for a function f(t, y):
        - t is the time vector. Dims: len(t)
        - y is the system solution. Dims: (len(integrated_vars) x len(t))
        - output dims:
            - If the function returns a scalar : len(t) array or scalar if len(t)==1
            - If the function returns an array : (array.shape x len(t))
            - If the function returns a dict : for each key, the shape described by those possible conditions.

    This class provides the following features:
        - Easy creation of the function for various input types through the constructor. It handles output shape and
        vectorization of the function.
        - Function arithmetic and logic through the usual operators.
        - Computation of the results if the linked system is solved.
    """

    def __init__(
            self,
            source: Source | tuple = None,
            operator_on_source_tuple=None,
            system: Optional["IVPSystemMutable"] = None,
    ):
        """
        Create a temporal variable.

        It represents an f(t, y) function: t is a timestamp, and y the system solution at this timestamp.

        Available sources are:
            - float
            - NumPy Array
            - dict
            - f(t) function
            - f(t, y) function
        If the source input is a tuple, the resulting function will apply the operator input to it.
        :param source: Input from which the internal function is built
        :param operator_on_source_tuple: If the source input is a tuple, create a function that applies the operator to these source items.
        """
        # Object data
        self.system = system

        # Private
        self._func: Callable[[float | NDArray, NDArray], T]
        self._source = source
        self._operator = operator_on_source_tuple
        # Output info
        self._output_type = None
        self._keys: list[str] | None = None
        self._shape: tuple[int, ...] | None = None

        # Create the function and make sources recursive when needed
        if self._operator is not None:
            # Create function for tuple and operator case
            assert type(self._source) is tuple

            def operator_func(t, y):

                def resolve_operator(t_inner, y_inner):
                    """
                    Compute args and kwargs value and call the operator
                    """
                    args = [x(t_inner, y_inner) if isinstance(x, TemporalVar) else x for x in self._source if
                            not isinstance(x, dict)]
                    kwargs = {k: v for d in [x for x in self._source if isinstance(x, dict)] for k, v in d.items()}
                    kwargs = {k: (x(t_inner, y_inner) if isinstance(x, TemporalVar) else x) for k, x in kwargs.items()}
                    return self._operator(*args, **kwargs)

                try:
                    # Assume that the function is vectorized
                    output = resolve_operator(t, y)
                except Exception as e:
                    # If it fails with a scalar t, the function failed for another reason
                    if np.isscalar(t):
                        raise e
                    # If it fails, call it for each t value
                    output = np.array([resolve_operator(t[i], y[:, i]) for i in range(len(t))])


                return output

            self._func = operator_func

        else:
            if isinstance(self._source, TemporalVar):
                self._func = self._source._func
            elif callable(self._source):
                # Source is a function
                n_args = len(inspect.signature(self._source).parameters)

                if n_args == 1:
                    # Function is a temporal function
                    def temporal_func(t, _):
                        return vectorize_source(self._source)(t)

                    self._func = temporal_func
                else:
                    # Function is already a f(t, y) function
                    self._func = self._source
            elif np.isscalar(self._source) or self._source is None:
                # Source is a scalar number
                self._output_type = type(self._source)

                def scalar_func(t, _):
                    if np.isscalar(t):
                        return self._source
                    else:
                        return np.full(len(t), self._source)

                self._func = scalar_func

            elif isinstance(self._source, (list, np.ndarray)):
                # Source is a numpy array
                self._output_type = np.ndarray
                self._source = np.array([TemporalVar(x, system=system) for x in self._source])

                def array_func(t, y):
                    return np.array([x(t, y) for x in self._source])

                self._func = array_func

            elif isinstance(self._source, dict):
                self._output_type = dict
                self._source = {key: TemporalVar(val, system=system) for key, val in self._source.items()}

                def dict_func(t, y):
                    return {key: x(t, y) for key, x in self._source.items()}

                self._func = dict_func
            else:
                raise ValueError(f"Unsupported type: {type(self._source)}.")

        # Get output type by calling the func
        self._output_type, self._keys, self._shape = get_output_info(self._func)

    def __call__(self, t: float | NDArray, y: Optional[NDArray] = None) -> T:
        if y is not None:
            return self._func(t, y)
        if self.system.sol is not None:
            return self._func(t, self.system.sol(t))
        else:
            raise RuntimeError(
                "The system has not been solved.\n"
                "Call the solve() method before inquiring the variable values."
            )

    @property
    def values(self):
        return self(self.system.t_eval)

    @property
    def t(self):
        return self.system.t_eval

    @classmethod
    def from_scenario(
            cls,
            scenario_table: pd.DataFrame,
            time_key: str,
            system: Optional["IVPSystemMutable"] = None,
            interpolation_kind: str = "linear",
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
        return cls(variables, system=system)

    def m(self, method: Callable[P, T]) -> Callable[P, "TemporalVar[T]"]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> TemporalVar:
            return TemporalVar((method, self, *args, kwargs), operator_on_source_tuple=operator_call,
                               system=self.system)

        functools.update_wrapper(wrapper, method)
        return wrapper

    def crosses(self, value: "float|TemporalVar[float]", direction: Direction = "both") -> "CrossTriggerVar":
        return self.system.add_crossing_detection(self - value, direction)

    # Magic methods
    def __getitem__(self, item):
        return TemporalVar(
            (self, item),
            operator_on_source_tuple=operator.getitem,
            system=self.system
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> "TemporalVar":
        if method == "__call__":
            return TemporalVar(
                (ufunc, *inputs, kwargs),
                operator_on_source_tuple=operator_call,
                system=self.system
            )

        return NotImplemented

    def __bool__(self):
        raise ValueError("The truth value of a Temporal Variable is ambiguous. Use vip.where() instead.")

    # Addition
    def __add__(self, other):
        return TemporalVar((self, other), operator_on_source_tuple=operator.add, system=self.system)

    def __radd__(self, other):
        return TemporalVar((other, self), operator_on_source_tuple=operator.add, system=self.system)

    # Subtraction
    def __sub__(self, other):
        return TemporalVar((self, other), operator_on_source_tuple=operator.sub, system=self.system)

    def __rsub__(self, other):
        return TemporalVar((other, self), operator_on_source_tuple=operator.sub, system=self.system)

    # Multiplication
    def __mul__(self, other):
        return TemporalVar((self, other), operator_on_source_tuple=operator.mul, system=self.system)

    def __rmul__(self, other):
        return TemporalVar((other, self), operator_on_source_tuple=operator.mul, system=self.system)

    # True division
    def __truediv__(self, other):
        return TemporalVar((self, other), operator_on_source_tuple=operator.truediv, system=self.system)

    def __rtruediv__(self, other):
        return TemporalVar((other, self), operator_on_source_tuple=operator.truediv, system=self.system)

    # Floor division
    def __floordiv__(self, other):
        return TemporalVar((self, other), operator_on_source_tuple=operator.floordiv, system=self.system)

    def __rfloordiv__(self, other):
        return TemporalVar((other, self), operator_on_source_tuple=operator.floordiv, system=self.system)

    # Modulo
    def __mod__(self, other):
        return TemporalVar((self, other), operator_on_source_tuple=operator.mod, system=self.system)

    def __rmod__(self, other):
        return TemporalVar((other, self), operator_on_source_tuple=operator.mod, system=self.system)

    # Power
    def __pow__(self, other):
        return TemporalVar((self, other), operator_on_source_tuple=operator.pow, system=self.system)

    def __rpow__(self, other):
        return TemporalVar((other, self), operator_on_source_tuple=operator.pow, system=self.system)

    # Unary plus
    def __pos__(self):
        return TemporalVar((self,), operator_on_source_tuple=operator.pos, system=self.system)

    # Unary minus
    def __neg__(self):
        return TemporalVar((self,), operator_on_source_tuple=operator.neg, system=self.system)

    # Absolute value
    def __abs__(self):
        return TemporalVar((self,), operator_on_source_tuple=operator.abs, system=self.system)

    def __eq__(self, other):
        return TemporalVar((self, other), operator_on_source_tuple=operator.eq, system=self.system)

    def __ne__(self, other):
        return TemporalVar((self, other), operator_on_source_tuple=operator.ne, system=self.system)

    def __lt__(self, other):
        return TemporalVar((self, other), operator_on_source_tuple=operator.lt, system=self.system)

    def __le__(self, other):
        return TemporalVar((self, other), operator_on_source_tuple=operator.le, system=self.system)

    def __gt__(self, other):
        return TemporalVar((self, other), operator_on_source_tuple=operator.gt, system=self.system)

    def __ge__(self, other):
        return TemporalVar((self, other), operator_on_source_tuple=operator.ge, system=self.system)

    @staticmethod
    def _apply_logical(logical_fun: Callable, a, b):
        result = logical_fun(a, b)
        if result.size == 1:
            result = result.item()
        return result

    def __and__(self, other) -> "TemporalVar[bool]":
        return TemporalVar((self._apply_logical, np.logical_and, self, other), operator_call, system=self.system)

    def __rand__(self, other) -> "TemporalVar[bool]":
        return TemporalVar((self._apply_logical, np.logical_and, other, self), operator_call, system=self.system)

    def __or__(self, other) -> "TemporalVar[bool]":
        return TemporalVar((self._apply_logical, np.logical_or, self, other), operator_call, system=self.system)

    def __ror__(self, other) -> "TemporalVar[bool]":
        return TemporalVar((self._apply_logical, np.logical_or, other, self), operator_call, system=self.system)

    def __xor__(self, other) -> "TemporalVar[bool]":
        return TemporalVar((self._apply_logical, np.logical_xor, self, other), operator_call, system=self.system)

    def __rxor__(self, other) -> "TemporalVar[bool]":
        return TemporalVar((self._apply_logical, np.logical_xor, other, self), operator_call, system=self.system)

    @staticmethod
    def _logical_not(a):
        result = np.logical_not(a)
        if result.size == 1:
            result = result.item()
        return result

    def __invert__(self) -> "TemporalVar[bool]":
        return TemporalVar((self._logical_not, self), operator_call, system=self.system)


class IntegratedVar(TemporalVar[float]):
    def __init__(self, index: int, system: "IVPSystemMutable"):
        func = create_system_output_fun(index)
        super().__init__(func, system=system)
        self._eq_idx: int = index

    @property
    def derivative(self):
        return self.system.derivatives[self._eq_idx]

    @derivative.setter
    def derivative(self, value: float | TemporalVar[float]):
        if not isinstance(value, TemporalVar):
            value = TemporalVar(value, system=self.system)
        self.system.set_derivative(value, self._eq_idx)

    def reinit(self, value: float | TemporalVar[float]) -> Action:
        if not isinstance(value, TemporalVar):
            value = TemporalVar(value)
        return Action(create_set_system_output_fun(self._eq_idx, value), ActionType.UPDATE_SYSTEM)


class CrossTriggerVar(TemporalVar[float]):
    def __init__(self, func: TemporalVar[float], direction: Direction, event_idx: int, system: "IVPSystemMutable"):
        self.direction = direction
        super().__init__(func, system=system)
        self.event_idx = event_idx

    def __call__(self, t, y=None):
        return np.isin(t, self.system.events_trigger[self.event_idx])

    def guard(self, t, y=None):
        return super().__call__(t, y)


def temporal_var_where(
        condition: TemporalVar[bool], a: T | TemporalVar[T], b: T | TemporalVar[T]
) -> TemporalVar[T]:
    assert_system_sameness(condition, a, b)

    def where(condition_, a_, b_):
        result = np.where(condition_, a_, b_)
        if result.size == 1:
            result = result.item()
        return result

    return TemporalVar(
        (where, condition, a, b),
        operator_call,
        condition.system
    )


# Utils
def assert_system_sameness(*values) -> None:
    """
    Raise an error if values contains TemporalVar instances from different systems.
    :param values: Arbitrary inputs
    :raises: ValueError
    """
    systems = []
    for value in values:
        if isinstance(value, TemporalVar):
            system = value.system
            if systems and system not in systems:
                raise ValueError("Variables from different systems can not be combined.")
            systems.append(system)
