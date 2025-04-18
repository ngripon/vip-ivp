import inspect
import operator
import types
from typing import Any, Generator

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
        if "temporal(" in fun_string:
            lambda_content = fun_string.split("temporal(")[1].strip()[0:-1]
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
