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

from typing import Callable, TypeVar, Generic, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")

Source = Callable[[float | NDArray, NDArray], T] | Callable[[float | NDArray], T] | NDArray | dict | float


class TemporalVar(Generic[T]):
    """
    Function f(t, y):
        - t is the time vector. Dims: len(t)
        - y is the system solution. Dims: (len(integrated_vars) x len(t))
    """
    def __init__(
            self,
            source: Source | tuple[Source, ...] = None,
            operator_on_source_tuple=None,
            is_discrete=False,
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
        :param is_discrete:
        """
        # Object data
        self.func: Callable[[float | NDArray, NDArray], T]
        # Private
        self._source = source
        self._operator = operator_on_source_tuple

        self._is_discrete = is_discrete
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

                if self._operator is operator_call and not np.isscalar(t):
                    output = np.array([resolve_operator(t[i], y[i]) for i in range(len(t))])
                    if output.ndim > 1:
                        output = np.moveaxis(output, 0, -1)
                else:
                    output = resolve_operator(t, y)
                return output

            self.func = operator_func

        else:
            if callable(self._source):
                # Source is a function
                n_args = len(inspect.signature(self._source).parameters)

                if n_args == 1:
                    # Function is a temporal function
                    def temporal_func(t, _):
                        return vectorize_source(self._source)(t)

                    self.func = temporal_func
                else:
                    # Function is already a f(t, y) function
                    self.func = self._source
            elif np.isscalar(self._source) or self._source is None:
                # Source is a scalar number
                self._output_type = type(self._source)

                def scalar_func(t, _):
                    if np.isscalar(t):
                        return self._source
                    else:
                        return np.full(len(t), self._source)

                self.func = scalar_func

            elif isinstance(self._source, (list, np.ndarray)):
                # Source is a numpy array
                self._output_type = np.ndarray
                self._source = np.array([TemporalVar(x) for x in self._source])

                def array_func(t, y):
                    return np.array([x.func(t, y) for x in self._source])

                self.func = array_func

            elif isinstance(self._source, dict):
                self._output_type = dict
                self._source = {key: TemporalVar(val) for key, val in self._source.items()}

                def dict_func(t, y):
                    return {key: x(t, y) for key, x in self._source.items()}

                self.func = dict_func
            else:
                raise ValueError(f"Unsupported type: {type(self._source)}.")

        # Get output type by calling the func
        self._output_type, self._keys, self._shape = get_output_info(self.func)

    def __call__(self, t: float | NDArray, y: NDArray) -> T:
        return self.func(t, y)

    @classmethod
    def from_scenario(
            cls,
            scenario_table: pd.DataFrame,
            time_key: str,
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
        return cls(variables)

    def m(self, method: Callable[P, T]) -> Callable[P, "TemporalVar[T]"]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> TemporalVar:
            return TemporalVar((method, self, *args, kwargs), operator_on_source_tuple=operator_call)

        functools.update_wrapper(wrapper, method)
        return wrapper

    @staticmethod
    def _apply_logical(logical_fun: Callable, a, b):
        result = logical_fun(a, b)
        if result.size == 1:
            result = result.item()
        return result

    def __getitem__(self, item):
        return TemporalVar(
            (self, item),
            operator_on_source_tuple=operator.getitem
        )

    @staticmethod
    def __array_ufunc__(ufunc, method, *inputs, **kwargs) -> "TemporalVar":
        if method == "__call__":
            return TemporalVar(
                (ufunc, *inputs, kwargs),
                operator_on_source_tuple=operator_call
            )

        return NotImplemented

    def __bool__(self):
        raise ValueError("The truth value of a Temporal Variable is ambiguous. Use vip.where() instead.")


# Utils

def operator_call(obj, /, *args, **kwargs):
    """operator.call function source code copy in order to be used with Python version <3.11"""
    return obj(*args, **kwargs)


def vectorize_source(fun: Callable[[float], Any]) -> Callable[[float], Any]:
    """
    Vectorize a temporal function.
    :param fun: Temporal function input by the user
    :return: Vectorized temporal function
    """
    accept_arrays, has_scalar_mode, is_constant = check_if_vectorized(fun)

    def vectorized_wrapper(t):
        if np.isscalar(t):
            return fun(t)
        else:
            return np.array([fun(ti) for ti in t])

    def handle_scalar_wrapper(t):
        output = fun(t)
        if np.isscalar(t):
            return output[0]
        return output

    def constant_wrapper(t):
        if np.isscalar(t):
            return fun(t)
        else:
            value = fun(0)
            if np.isscalar(value):
                return np.full(len(t), value)
            else:
                return np.moveaxis(np.broadcast_to(value, (len(t),) + value.shape), 0, -1)

    if is_constant:
        return constant_wrapper
    elif accept_arrays and has_scalar_mode:
        return fun
    elif not has_scalar_mode:
        return handle_scalar_wrapper
    else:
        print(f"Warning: the function '{fun.__name__}' is not vectorizable.")
        return vectorized_wrapper


def check_if_vectorized(fun) -> tuple[bool, bool, bool]:
    accept_arrays = True
    has_scalar_mode = True
    is_constant = False

    # Test with scalar
    scalar_output = fun(0)

    # Find a t length that does not match the len of a dimension of a scalar output
    array_len = 3
    if not np.isscalar(scalar_output):
        while array_len in scalar_output.shape:
            array_len += 1

    # Test with array input
    array_input = np.zeros(array_len)
    try:
        array_output = fun(array_input)
        if np.isscalar(array_output) or not np.isscalar(scalar_output) and array_output.shape == scalar_output.shape:
            is_constant = True
            raise ValueError("The array output does not create a vector")
        scalar_ndim = 0 if np.isscalar(scalar_output) else np.asarray(scalar_output).ndim
        if array_output.ndim == scalar_ndim:
            has_scalar_mode = False
        elif scalar_ndim != array_output.ndim - 1:
            raise ValueError("There is something wrong in the output dimensions of the function.")
    except ValueError:
        accept_arrays = False

    return accept_arrays, has_scalar_mode, is_constant


def get_output_info(
        fun: Callable[[float | NDArray, NDArray], NDArray]
) -> tuple[type, list[str] | None, list[int] | None]:
    """
    Get the output information of a system function.
    :param fun: System function
    :return: Type, keys and shape
    """
    # Get a scalar output but watch out for IndexError
    y = np.zeros(1)
    found_output = False
    while not found_output:
        try:
            output = fun(0, y)
            found_output = True
        except IndexError:
            y = np.zeros(len(y) * 2)
            pass
    # Analyse the output
    output_type = type(output)
    if isinstance(output, dict):
        keys = list(output.keys())
    else:
        keys = None
    if isinstance(output, np.ndarray):
        shape = output.shape
    else:
        shape = None
    return output_type, keys, shape
