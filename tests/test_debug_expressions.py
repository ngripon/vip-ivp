import vip_ivp as vip
import numpy as np


def test_source_expression():
    lambda_fun = lambda t: t

    def fun(t):
        return t

    constant = vip.create_source(5)
    anonymous = vip.create_source(lambda t: t)
    lambda_source = vip.create_source(lambda_fun)
    fun_source = vip.create_source(fun)

    dict_source = vip.create_source({"a": 5, "b": 4})
    list_source = vip.create_source([1, 2, 3])
    array_source = vip.create_source(np.linspace(0, 10, 11))

    # Scalars
    assert constant._expression == str(5)
    assert anonymous._expression == "lambda t: t"
    assert lambda_source._expression == "lambda t: t"  # Assigned lambda should be identical
    assert fun_source._expression == "fun(t)"
    # Collection
    assert repr(dict_source) == str({"a": 5, "b": 4})  # Convert to string
    assert repr(list_source) == str([1, 2, 3])  # Convert list to string
    assert repr(array_source) == str(np.linspace(0, 10, 11))  # Convert array to list then string
    # Collection elements
    assert dict_source["a"]._expression == 'dict_source[a]'
    assert list_source[1]._expression == "list_source[1]"
    assert array_source[2]._expression == "array_source[2]"


def test_operations():
    a = vip.create_source(0)
    b = vip.create_source(1)
    c = vip.create_source(2)
    d = vip.create_source(3)

    # Testing addition
    ab = a + b
    assert ab.expression == "a + b"

    # Testing subtraction
    ac = a - c
    assert ac.expression == "a - c"

    # Testing multiplication
    ad = a * d
    assert ad.expression == "a * d"

    # Testing division
    ab_div = a / b
    assert ab_div.expression == "a / b"

    # Testing power
    ab_pow = a ** b
    assert ab_pow.expression == "a ** b"

    # Testing modulo
    ab_mod = a % b
    assert ab_mod.expression == "a % b"

    # Testing mixed operations
    e = a + b * c
    assert e.expression == "a + b * c"

    f = (a + b) * c
    assert f.expression == "(a + b) * c"

    g = a * (b + c)
    assert g.expression == "a * (b + c)"

    h = (a + b) ** 2
    assert h.expression == "(a + b) ** 2"

    i = a * b + c
    assert i.expression == "a * b + c"

    # Check more complex combinations
    j = (a + b) * (c + d)
    assert j.expression == "(a + b) * (c + d)"

    k = a + b + c + d
    assert k.expression == "a + b + c + d"

    l = (a + b) * c + d
    assert l.expression == "(a + b) * c + d"

    n = a + b * (c + d)
    assert n.expression == "a + b * (c + d)"

    # Testing more complicated expressions involving addition and subtraction
    p = a + b - c + d
    assert p.expression == "a + b - c + d"

    # Check multiple chained operations
    r = a + b + c + d
    assert r.expression == "a + b + c + d"

    # Check with other operations mixed in
    s = a + b * c + d
    assert s.expression == "a + b * c + d"

    t = np.sin(a, where=b > 1)
    assert t.expression == "sin(a, where=b > 1)"

    u = -(a + b)
    assert u.expression == "-(a + b)"

    v = a - (b + c)
    assert v.expression == "a - (b + c)"


def test_diff_equation():
    ddx = vip.loop_node()
    dx = vip.integrate(ddx, 0)
    x = vip.integrate(dx, 0)
    ddx.loop_into(2 * x - 4 * dx)

    assert dx.expression == "#INTEGRATE ddx"
    assert x.expression == "#INTEGRATE dx"
    assert ddx.expression == "2 * x - 4 * dx"


def test_transformations():
    a = vip.create_source(5)

    da = vip.differentiate(a)
    ia = vip.integrate(a, 0)
    delay_a = vip.delay(a, 1)

    def foo(bar):
        return bar + 1

    foo_a = vip.f(foo)(a)

    assert da.expression == "#D/DT a"
    assert ia.expression == "#INTEGRATE a"
    assert delay_a.expression == "#DELAY(1) a"
    assert foo_a.expression == "foo(a)"
