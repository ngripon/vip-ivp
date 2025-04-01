import inspect
import types
from typing import Any, Generator, Callable

import numpy as np


def shift_array(arr: np.ndarray, n: int, fill_value: float = 0):
    shifted = np.roll(arr, n, axis=-1)  # Shift the array
    if n > 0:
        shifted[..., :n] = fill_value  # Fill first n elements
    elif n < 0:
        shifted[..., n:] = fill_value  # Fill last n elements
    return shifted


def convert_to_string(content):
    if inspect.isfunction(content):
        name = getattr(content, "__name__")
        if name != "<lambda>":
            return name + str(inspect.signature(content))
        fun_string = inspect.getsourcelines(content)[0][0].strip()
        if "create_source" in fun_string:
            lambda_content = fun_string.split("create_source")[1].strip()[1:-1]
            return lambda_content
        for word in ["set_timeout", "set_interval"]:
            if word in fun_string:
                start_index = fun_string.find(word, len(word))
                lambda_content = ", ".join(fun_string[start_index:].split(",")[:-1])
                return lambda_content
        if "=" in fun_string:
            fun_string = fun_string.split("=")[1].strip()
        return fun_string
    elif inspect.isclass(content):
        return content.__repr__()
    return str(content)


def add_necessary_brackets(expression: str) -> str:
    operators = ["+", "-", "=", "<", ">"]
    begin = expression.split("(")[0]
    end = expression.split(")")[-1]
    if any(op in begin for op in operators) or any(op in end for op in operators):
        return f"({expression})"
    else:
        return expression


def iter_structure(data: Any) -> Generator:
    """
    Recursively iterates over an arbitrary structure and yields (container, key/index, value)
    so the caller can modify the structure in-place if it's mutable.

    Args:
        data: The arbitrary structure (list, dict, object, etc.)

    Yields:
        (parent, key/index, value) where:
        - parent: The container holding the value
        - key/index: The key (for dicts) or index (for lists)
        - value: The actual value
    """
    if isinstance(data, list):
        for i, item in enumerate(data):
            yield data, i, item  # Yield reference to modify in-place
            yield from iter_structure(item)

    elif isinstance(data, dict):
        for key, value in data.items():
            yield data, key, value  # Yield reference to modify in-place
            yield from iter_structure(value)

    elif hasattr(data, "__dict__"):  # Check if it's an object
        for key, value in vars(data).items():
            yield data, key, value  # Yield reference to modify in-place
            yield from iter_structure(value)

    else:
        yield None, None, data  # Base case: Scalars (no modification needed)


def is_custom_class(obj: Any) -> bool:
    # Check if the object is a built-in type like list, dict, scalar, or ndarray
    if isinstance(obj, (list, dict, int, float, np.ndarray, bool, str, types.FunctionType, types.LambdaType)):
        return False

    # Check if the object is an instance of a custom class
    if isinstance(obj, object):
        return True

    return False
