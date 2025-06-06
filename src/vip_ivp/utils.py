import inspect
import operator
import types
import warnings
from typing import Any, Generator, Callable

import numpy as np


def vectorize_source(fun: Callable) -> Callable:
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


def check_if_vectorized(fun) -> (bool, bool, bool):
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


def shift_array(arr: np.ndarray, n: int, fill_value: float = 0):
    shifted = np.roll(arr, n, axis=-1)  # Shift the array
    if n > 0:
        shifted[..., :n] = fill_value  # Fill first n elements
    elif n < 0:
        shifted[..., n:] = fill_value  # Fill last n elements
    return shifted


def convert_to_string(content):
    try:
        if inspect.isfunction(content):
            name = getattr(content, "__name__")
            if name != "<lambda>":
                return name + str(inspect.signature(content))
            fun_string = inspect.getsourcelines(content)[0][0].strip()
            if "temporal(" in fun_string:
                lambda_content = fun_string.split("temporal(")[1].strip()[0:-1]
                return lambda_content
            for word in ["execute_on"]:
                if word in fun_string:
                    start_index = fun_string.find(word, len(word))
                    lambda_content = ", ".join(fun_string[start_index:].split(",")[1:])[:-1].strip()
                    return lambda_content
            if "=" in fun_string:
                fun_string = fun_string.split("=")[1].strip()
            return fun_string
        elif inspect.isclass(content):
            return content.__repr__()
    except Exception as e:
        warnings.warn(f"Developer error: Failed to create a string representation from {content} because of error {e}.")
    return str(content)


def add_necessary_brackets(expression: str) -> str:
    operators = ["+", "-", "=", "<", ">", "and", "or", "not", "xor"]
    begin = expression.split("(")[0]
    end = expression.split(")")[-1]
    if any(op in begin for op in operators) or any(op in end for op in operators):
        return f"({expression})"
    else:
        return expression


def is_custom_class(obj: Any) -> bool:
    # Check if the object is a built-in type like list, dict, scalar, or ndarray
    if isinstance(obj, (list, dict, int, float, np.ndarray, bool, str, types.FunctionType, types.LambdaType)):
        return False

    # Check if the object is an instance of a custom class
    if isinstance(obj, object):
        return True

    return False


def operator_call(obj, /, *args, **kwargs):
    """operator.call function source code copy in order to be used with Python version <3.11"""
    return obj(*args, **kwargs)
