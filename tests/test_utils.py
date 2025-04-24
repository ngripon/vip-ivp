import numpy as np

from src.vip_ivp.utils import check_if_vectorized, vectorize_source


def test_check_if_vectorized():
    res1 = check_if_vectorized(scalar_only_fun)
    res2 = check_if_vectorized(constant_fun)
    res3 = check_if_vectorized(scalar_mode_fun)
    res4 = check_if_vectorized(vectorized_function)

    assert res1[0] == False
    assert res2[0] == False
    assert res3[0] == True and res3[1] == True
    assert res4[0] == True and res4[1] == False
    print(res3)


def test_vectorization():
    scalar_input = 1
    array_input = np.linspace(0, 5, 6)
    # Test 0D functions
    fun_list = [scalar_only_fun, scalar_mode_fun, vectorized_function]
    for fun in fun_list:
        assert vectorize_source(fun)(scalar_input) == scalar_input
        assert np.array_equal(vectorize_source(fun)(array_input), array_input)
    # Test constant function
    assert vectorize_source(constant_fun)(scalar_input) == 1
    assert np.array_equal(vectorize_source(constant_fun)(array_input), np.ones_like(array_input))


def scalar_only_fun(t):
    return max(t, 0)


def scalar_mode_fun(t):
    if np.isscalar(t):
        return t
    else:
        return t


def vectorized_function(t):
    if np.isscalar(t):
        return np.array([t])
    return np.array(t)


def constant_fun(t):
    return 1
