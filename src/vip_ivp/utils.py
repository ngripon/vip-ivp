import inspect

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
        fun_string = fun_string.split(" = ")[1]
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


def flatten(input_data):
    if isinstance(input_data, dict):
        flat_input = list(input_data.values())
    elif isinstance(input_data, np.ndarray):
        flat_input = input_data.flatten().tolist()
    elif isinstance(input_data, list):
        flat_input = np.array(input_data, dtype=object).flatten().tolist()
    else:
        flat_input = [input_data]
    return flat_input


def unflatten(flat_output, input_data):
    if isinstance(input_data, dict):
        keys = list(input_data.keys())
        return dict(zip(keys, flat_output))
    elif isinstance(input_data, np.ndarray):
        return np.array(flat_output).reshape(input_data.shape)
    elif isinstance(input_data, list):
        return np.array(flat_output, dtype=object).reshape(np.array(input_data, dtype=object).shape).tolist()
    else:
        return flat_output[0]
