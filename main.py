from typing import Callable

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class Solver:
    def __init__(self):
        self.dim = 0
        self.vars = []
        self.feed_vars = []

    def create_variables(self, x0: tuple) -> list:
        n = len(x0)
        var_list = []
        var = TemporalVar(lambda t, y: y[self.dim])
        var.set_init(x0[-1])
        var_list.append(var)
        for i in range(n):
            if i != n - 1:
                var = FeedVar(lambda t, y: y[self.dim + i + 1])
                var.set_init(x0[-2 - i])
            else:
                var = FeedVar()
            var_list.append(var)
        self.dim += n
        return var_list


solver = Solver()


class TemporalVar:
    def __init__(self, fun: Callable = None):
        self.solver = solver
        self.content = []
        self.function = fun

        self.solver.vars.append(self)

    def set_init(self, x0: float):
        self.content.append(x0)

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

    def __repr__(self):
        return f"{self.content=}"


class FeedVar(TemporalVar):
    def __init__(self, fun: Callable = None):
        super().__init__(fun)
        self.solver.feed_vars.append(self)

    def set_value(self, value):
        if isinstance(value, TemporalVar):
            self.function = value.function
        else:
            self.function = lambda t, y: value


def mass_spring_damper(t, y):
    ddy = 1 / m * (-c * y[1] - k * y[0])
    return y[1], ddy


t_final = 50
m = 5
k = 1
c = 0.5
v0 = 0
x0 = 1
# Comparison
res_normal = solve_ivp(mass_spring_damper, [0, t_final], (x0, v0))
print(res_normal)

pos, vit, acc = solver.create_variables((x0, v0))
acc.set_value(1 / m * (-c * vit - k * pos))

plt.plot(res_normal.t, res_normal.y[0])
plt.show()