from typing import Callable, Any

import numpy as np
from numpy.typing import NDArray
from ..domain.system import SystemFun


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
        fun: SystemFun
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
