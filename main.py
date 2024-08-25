import time
from typing import Callable

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class Solver:
    def __init__(self):
        self.dim = 0
        self.vars = []
        self.feed_vars = []
        self.initialized_vars = []
        self.t = None
        self.y = None
        self.solved = False

    def create_variables(self, x0: tuple) -> list:
        try:
            n = len(x0)
        except TypeError:
            n = 1
            x0=(x0,)
        var_list = []
        var = TemporalVar(lambda t, y, idx=self.dim: y[idx])
        var.set_init(x0[0])
        var_list.append(var)
        for i in range(n):
            if i != n - 1:
                var = FeedVar(lambda t, y, idx=self.dim + i + 1: y[idx])
                var.set_init(x0[1 + i])
            else:
                var = FeedVar()
            var_list.append(var)
        self.dim += n
        return var_list

    def solve(self, t_end: float):
        # Apply checks before attempting to solve
        self._check_feed_init()
        x0 = [x.init for x in self.initialized_vars]
        # Reinit values
        [var.reset() for var in self.vars]
        start = time.time()
        try:
            res = solve_ivp(self._dy, (0, t_end), x0)
        except RecursionError:
            raise RecursionError("An algebraic loop has been detected in the system. "
                                 "Please check in the set_value() methods if a variable use itself for computing "
                                 "its value.")
        print(f"Performance = {time.time() - start}")
        self.t = res.t
        self.y = res.y
        self.solved = True
        return res

    def _dy(self, t, y):
        result = []
        for var in self.feed_vars:
            result.append(var.function(t, y))
        return result

    def _check_feed_init(self):
        uninitialized_vars = [var for var in self.feed_vars if var.function is None]
        if uninitialized_vars:
            raise ValueError(f"The following variables have not been set a value: {uninitialized_vars}. "
                            f"Call the set_value() method of each of these variables.")


solver = Solver()


class TemporalVar:
    def __init__(self, fun: Callable = None):
        self.solver = solver
        self.init = None
        self.function = fun
        self.initialized = False
        self._values = None

        self.solver.vars.append(self)

    @property
    def values(self):
        if not self.solver.solved:
            raise Exception("The differential system has not been solved. "
                            "Call the solve() method before inquiring the variable values.")
        if self._values is None:
            self._values = self.function(self.solver.t, self.solver.y)
        return self._values

    @property
    def t(self):
        if not self.solver.solved:
            raise Exception("The differential system has not been solved. "
                            "Call the solve() method before inquiring the time variable.")
        return self.solver.t

    def reset(self):
        self._values = None

    def set_init(self, x0: float):
        self.init = x0
        self.initialized = True
        self.solver.initialized_vars.append(self)

    def __add__(self, other):
        if isinstance(other, TemporalVar):
            return TemporalVar(lambda t, y: self.function(t, y) + other.function(t, y))
        else:
            return TemporalVar(lambda t, y: other + self.function(t, y))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, TemporalVar):
            return TemporalVar(lambda t, y: self.function(t, y) - other.function(t, y))
        else:
            return TemporalVar(lambda t, y: self.function(t, y) - other)

    def __rsub__(self, other):
        if isinstance(other, TemporalVar):
            return TemporalVar(lambda t, y: other.function(t, y) - self.function(t, y))
        else:
            return TemporalVar(lambda t, y: other - self.function(t, y))

    def __mul__(self, other):
        if isinstance(other, TemporalVar):
            return TemporalVar(lambda t, y: self.function(t, y) * other.function(t, y))
        else:
            return TemporalVar(lambda t, y: other * self.function(t, y))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return TemporalVar(lambda t, y: - self.function(t, y))

    def __repr__(self):
        if self.solver.solved:
            return f"{self.values}"
        for key, value in globals().items():
            if value is self:
                return str(key)
        else:
            return "There must be a bug somewhere else..."


class FeedVar(TemporalVar):
    def __init__(self, fun: Callable = None):
        super().__init__(fun)
        self.solver.feed_vars.append(self)

    def set_value(self, value):
        if isinstance(value, TemporalVar):
            if value.function is not None:
                self.function = value.function
            else:
                raise RecursionError("There is an algebraic loop with this variable.")
        else:
            self.function = lambda t, y: value


if __name__ == '__main__':
    def mass_spring_damper(t, y):
        ddy = 1 / m * (-c * y[1] - k * y[0])
        return y[1], ddy


    t_final = 50
    m = 5
    k = 2
    c = 0.5
    v0 = 2
    x0 = 5
    pos, vit, acc = solver.create_variables((x0, v0))
    acc.set_value(1 / m * (-c * vit - k * pos + acc))

    res_awesome = solver.solve(t_final)

    # Comparison
    # start = time.time()
    # res_normal = solve_ivp(mass_spring_damper, [0, t_final], (x0, v0))
    # print(time.time() - start)
    # print(res_normal)



    # plt.plot(res_normal.t, res_normal.y[1])
    # plt.plot(acc.t, acc.values)
    # plt.plot(my_test.t, my_test.values)
    # plt.show()
