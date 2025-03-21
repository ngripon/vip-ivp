import vip_ivp as vip
import numpy as np

def test_source_expression():
    lambda_fun = lambda t: t

    def fun(t):
        return t

    class Foo:
        def __init__(self, bar):
            self.bar = bar

    class_value = Foo(5)

    constant = vip.create_source(5)
    anonymous = vip.create_source(lambda t: t)
    lambda_source = vip.create_source(lambda_fun)
    fun_source = vip.create_source(fun)
    dict_source = vip.create_source({"a": 5, "b": 4})
    list_source = vip.create_source([1, 2, 3])
    array_source = vip.create_source(np.linspace(0, 10, 11))
    object_source = vip.create_source(class_value)

    assert constant._expression == str(5)
    assert anonymous._expression == "lambda t: t"
    assert lambda_source._expression == "lambda t: t"  # Assigned lambda should be identical
    assert fun_source._expression == "fun(t)"
    assert dict_source._expression == str({"a": 5, "b": 4})  # Convert to string
    assert list_source._expression == str([1, 2, 3])  # Convert list to string
    assert array_source._expression == str(np.linspace(0, 10, 11))  # Convert array to list then string
    assert object_source._expression == repr(class_value)  # Use object representation