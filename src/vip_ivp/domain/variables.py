import enum
import functools
import inspect
import operator

from typing import Callable, TypeVar, Generic

import pandas as pd
from numpy.typing import NDArray
from typing_extensions import ParamSpec

from ..solver_utils import *
from ..utils import operator_call, shift_array, vectorize_source

T = TypeVar("T")
P = ParamSpec("P")


class CallMode(enum.Enum):
    CALL_ARGS_FUN = 0
    CALL_FUN_RESULT = 1


class TemporalVar(Generic[T]):
    def __init__(
            self,
            source: Callable[[float | NDArray, NDArray], T] |
                    Callable[[float | NDArray], T] |
                    NDArray |
                    dict |
                    float |
                    tuple = None,
            child_cls=None,
            operator=None,
            call_mode: CallMode = CallMode.CALL_ARGS_FUN,
            is_discrete=False
    ):
        self._output_type = None
        self._is_source = False
        self._call_mode = call_mode
        self.is_discrete = is_discrete
        # Recursive building
        self.operator = operator
        child_cls = child_cls or type(self)
        if self.operator is not None:
            self.source = source
        else:
            self._is_source = True
            if callable(source) and not isinstance(source, child_cls):
                n_args = len(inspect.signature(source).parameters)
                if n_args == 1:
                    self.source = lambda t, y: vectorize_source(source)(t)
                else:
                    self.source = lambda t, y: source(t, y)
            elif np.isscalar(source):
                self.source = source
                self._output_type = type(source)
            elif isinstance(source, (list, np.ndarray)):
                self._output_type = np.ndarray
                self.source = np.vectorize(lambda f: child_cls(f))(
                    np.array(source)
                )
            elif isinstance(source, TemporalVar):
                vars(self).update(vars(source))
            elif isinstance(source, dict):
                self._output_type = dict
                self.source = {key: child_cls(val) for key, val in source.items()}
            elif source is None:
                self.source = None
            else:
                raise ValueError(f"Unsupported type: {type(source)}.")

    def __call__(self, t: float | NDArray, y: NDArray) -> T:
        # Handle dict in a recursive way
        if isinstance(self.source, dict):
            return {key: val(t, y) for key, val in self.source.items()}
        else:
            if isinstance(self.source, np.ndarray):
                if np.isscalar(t):
                    output = np.stack(np.frompyfunc(lambda f: f(t, y), 1, 1)(self.source))
                else:
                    output = np.stack(
                        np.frompyfunc(lambda f: f(t, y), 1, 1)(self.source.ravel())
                    ).reshape((*self.source.shape, *np.array(t).shape))
            elif self.operator is not None:
                if self.operator is operator_call and not np.isscalar(t):
                    output = np.array([self._resolve_operator(t[i], y[i]) for i in range(len(t))])
                    if output.ndim > 1:
                        output = np.moveaxis(output, 0, -1)
                else:
                    output = self._resolve_operator(t, y)
            else:
                if callable(self.source):
                    output = self.source(t, y)
                elif np.isscalar(t):
                    output = self.source
                else:
                    output = np.full(len(t), self.source)
        return output

    def _resolve_operator(self, t, y):
        if self._call_mode == CallMode.CALL_ARGS_FUN:
            args = [x(t, y) if isinstance(x, TemporalVar) else x for x in self.source if
                    not isinstance(x, dict)]
            kwargs = {k: v for d in [x for x in self.source if isinstance(x, dict)] for k, v in d.items()}
            kwargs = {k: (x(t, y) if isinstance(x, TemporalVar) else x) for k, x in kwargs.items()}
            return self.operator(*args, **kwargs)
        elif self._call_mode == CallMode.CALL_FUN_RESULT:
            args = [x for x in self.source if not isinstance(x, dict)]
            kwargs = {k: v for d in [x for x in self.source if isinstance(x, dict)] for k, v in d.items()}
            return self.operator(*args, **kwargs)(t, y)
        else:
            raise ValueError(f"Unknown call mode: {self._call_mode}.")

    # @property
    # def output_type(self):
    #     if self._output_type is None:
    #         self._output_type = type(self._first_value())
    #     return self._output_type

    @classmethod
    def from_scenario(
            cls,
            scenario_table: "pd.DataFrame",
            time_key: str,
            interpolation_kind="linear",
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

    def delayed(self, delay: int, initial_value: T = 0) -> "TemporalVar[T]":
        """
        Create a delayed version of the TemporalVar.
        :param delay: Number of solver steps by which the new TemporalVar is delayed.
        :param initial_value: Value of the delayed variable at the beginning when there is not any value for the original value.
        :return: Delayed version of the TemporalVar
        """
        if delay < 1:
            raise Exception("Delay accept only a positive step.")

        def create_delay(input_variable):
            def previous_value(t, y):
                if np.isscalar(t):
                    if len(input_variable.solver.t) >= delay:
                        index = np.searchsorted(input_variable.solver.t, t, "left")
                        # index = next((i for i, ts in enumerate(input_variable.solver.t) if t <= ts),
                        #              len(input_variable.solver.t))
                        if index - delay < 0:
                            return initial_value
                        previous_t = input_variable.solver.t[index - delay]
                        previous_y = input_variable.solver.y[index - delay]

                        return input_variable(previous_t, previous_y)
                    else:
                        return initial_value
                else:
                    delayed_t = shift_array(t, delay, 0)
                    delayed_y = shift_array(y, delay, initial_value)
                    return input_variable(delayed_t, delayed_y)

            return previous_value

        return TemporalVar((create_delay, self),
                           operator=operator_call,
                           call_mode=CallMode.CALL_FUN_RESULT,
                           is_discrete=True)

    def m(self, method: Callable[P, T]) -> Callable[P, "TemporalVar[T]"]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> TemporalVar:
            return TemporalVar((method, self, *args, kwargs), operator=operator_call)

        functools.update_wrapper(wrapper, method)
        return wrapper

    # def crosses(self, value: "TemporalVar[T]|T",
    #             direction: Literal["rising", "falling", "both"] = "both") -> "CrossTriggerVar":
    #     """
    #     Create a signal that triggers when the specified crossing occurs.
    #
    #     :param value: Value to be crossed to cause the triggering.
    #     :param direction: Direction of the crossing.
    #     :return: TriggerVar
    #     """
    #     if self.output_type in (bool, np.bool, str):
    #         crossed_variable = self == value
    #     elif issubclass(self.output_type, abc.Iterable):
    #         raise ValueError(
    #             "Can not apply crossing detection to a variable containing a collection of values because it is ambiguous."
    #         )
    #     else:
    #         crossed_variable = self - value
    #     trigger_var = CrossTriggerVar(crossed_variable, direction)
    #     return trigger_var

    @staticmethod
    def _from_arg(value: "TemporalVar[T]|T") -> "TemporalVar[T]":
        """
        Return a TemporalVar from an argument value. If the argument is already a TemporalVar, return it. If not, create a TemporalVar from the value.
        """
        if isinstance(value, TemporalVar):
            return value
        return TemporalVar(value)

    @staticmethod
    def _apply_logical(logical_fun: Callable, a, b):
        result = logical_fun(a, b)
        if result.size == 1:
            result = result.item()
        return result

    @staticmethod
    def _logical_not(a):
        result = np.logical_not(a)
        if result.size == 1:
            result = result.item()
        return result

    def __invert__(self) -> "TemporalVar[bool]":
        return TemporalVar(
            (self._logical_not, self),
            operator=operator_call
        )

    def __getitem__(self, item):
        return TemporalVar(
            (self, item),
            operator=operator.getitem
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> "TemporalVar":
        if method == "__call__":
            return TemporalVar(
                (ufunc, *inputs, kwargs),
                operator=operator_call
            )

        return NotImplemented

    def __bool__(self):
        raise ValueError("The truth value of a Temporal Variable is ambiguous. Use vip.where() instead.")

    # NumPy arrays utility methods
    # @property
    # def shape(self):
    #     result = self._first_value()
    #     if isinstance(result, np.ndarray):
    #         return result.shape
    #     raise AttributeError("shape attribute does not exist because this variable does not contain a NumPy array.")

    # Dict utility methods
    # def keys(self):
    #     result = self._first_value()
    #     if isinstance(result, dict):
    #         return result.keys()
    #     raise AttributeError("keys() method does not exist because this variable does not contain a dict.")
    #
    # def items(self):
    #     result = self._first_value()
    #     if isinstance(result, dict):
    #         key_list = list(result.keys())
    #         value_list = [self[key] for key in key_list]
    #         return zip(key_list, value_list)
    #     raise AttributeError("items() method does not exist because this variable does not contain a dict.")
