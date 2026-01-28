"""
This module contain the domain class for an Initial Value Problem (IVP).

An IVP is defined by the system y'(t) = f(t, y(t)), and an initial point (t0, y0). By using an integration scheme, the
solution y(t) is computed.


"""
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import OdeSolution, solve_ivp

EPS = np.finfo(float).eps
CROSSING_TOLERANCE = 1e-12

SystemFun = Callable[[float | NDArray, NDArray], NDArray | float]


class SystemEvent:
    def __init__(self, condition: SystemFun, action: None) -> None:
        self.condition = condition
        self.action = action

        # Cache current value
        self._current_t = None
        self._current_value = None

    def compute_root(self, t, t_next, sol) -> float | None:
        if not self._check_zero_crossing(t, t_next, sol):
            return None

        from scipy.optimize import brentq

        if abs(self.condition(t, sol(t))) <= CROSSING_TOLERANCE:
            return t
        elif abs(self.condition(t_next, sol(t_next))) <= CROSSING_TOLERANCE:
            return t_next
        return brentq(lambda t_: self.condition(t_, sol(t_)), t, t_next, xtol=4 * EPS, rtol=4 * EPS)

    def _check_zero_crossing(self, t, t_next, sol) -> bool:
        if self._current_value is None or t != self._current_t:
            y0 = self.condition(t, sol(t))
        else:
            y0 = self._current_value
        y1 = self.condition(t_next, sol(t_next))
        self._cache_current_value(t_next, y1)
        return np.sign(y0) != np.sign(y1)

    def _cache_current_value(self, t, value):
        self._current_t = t
        self._current_value = value


class IVPSystem:
    def __init__(self, derivative_expressions: tuple[SystemFun, ...],
                 initial_conditions: tuple[float, ...]):
        assert len(derivative_expressions) == len(initial_conditions)
        self.derivatives = derivative_expressions
        self.initial_conditions = initial_conditions

    @property
    def n_equations(self) -> int:
        return len(self.derivatives)

    def solve(self, t_end: float, method: str = "RK45") -> OdeSolution:
        # Check
        for der_idx, der in enumerate(self.derivatives):
            if der is None:
                raise ValueError(f"Derivative at index {der_idx} is None. Solving aborted.")
        # Solve
        result = solve_ivp(self._dy, [0, t_end], self.initial_conditions, method=method, dense_output=True)
        return result.sol

    def _dy(self, t, y):
        try:
            return np.array([f(t, y) for f in self.derivatives])
        except RecursionError:
            raise RecursionError(
                "An algebraic loop has been detected."
            )


def create_system_output(idx: int) -> SystemFun:
    def system_output(t: float | NDArray, y: NDArray, i=idx) -> NDArray:
        return y[i]

    return system_output
