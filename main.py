from typing import Callable


class Solver:
    def __init__(self):
        self.dim = 0
        self.vars = []
        self.feed_vars = []

    def create_variables(self, x0: list) -> list:
        n = len(x0)
        var0 = FeedVar()
        var_list = [var0]
        for i in range(n):
            if i != n - 1:
                var = FeedVar(lambda t, y: y[self.dim + n - i])
            else:
                var = TemporalVar(lambda t, y: y[self.dim])
            var.set_init(x0[i])
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

    def __repr__(self):
        return f"{self.content=}"


class FeedVar(TemporalVar):
    def __init__(self, fun: Callable = None):
        super().__init__(fun)

    def set_value(self, value):
        if isinstance(value, TemporalVar):
            self.function = value.function
        else:
            self.function = lambda t, y: value


m = 5
k = 1
c = 0.5
acc, vit, pos = solver.create_variables([0, 1])
