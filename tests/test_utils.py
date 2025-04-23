import numpy as np

from src.vip_ivp.utils import check_if_vectorized


def test_check_if_vectorized():
    def scalar_only_fun(t):
        return max(1.0 - 0.0004 * t, 0)

    def array_only_fun(t_arr):
        if np.isscalar(t_arr):
            raise ValueError
        return t_arr

    def scalar_mode_fun(t):
        if np.isscalar(t):
            return 1
        else:
            return t

    def vectorized_function(t):
        if np.isscalar(t):
            return np.array([t])
        return np.array(t)

    res1=check_if_vectorized(scalar_only_fun)
    # res2 = check_if_vectorized(array_only_fun)
    res3=check_if_vectorized(scalar_mode_fun)
    res4=check_if_vectorized(vectorized_function)

    assert res1[0]==False
    # assert res2[0]==True and res2[1]
    assert res3[0]==True and res3[1]==True
    assert res4[0]==True and res4[1]==False
    print(res3)