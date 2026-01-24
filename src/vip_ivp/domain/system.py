"""
This module contain the domain class for an Initial Value Problem (IVP).

An IVP is defined by the system y'(t) = f(t, y(t)), and an initial point (t0, y0). By using an integration scheme, the
solution y(t) is computed.


"""
import numpy as np
from scipy.integrate import OdeSolution, solve_ivp

from src.vip_ivp.domain.variables import TemporalVar


class IVPSystem:
    def __init__(self, derivative_expressions: tuple[TemporalVar[float]], initial_conditions: tuple[float]):
        assert len(derivative_expressions) == len(initial_conditions)
        self._derivatives = derivative_expressions
        self._initial_conditions = initial_conditions

    def _dy(self, t, y):
        try:
            return np.array([f(t, y) for f in self._derivatives])
        except RecursionError:
            raise RecursionError(
                "An algebraic loop has been detected."
            )

    def solve(self, t_end: float, method: str = "RK45") -> OdeSolution:
        result = solve_ivp(self._dy, [0, t_end], self._initial_conditions, method=method, dense_output=True)
        return result.sol


def get_integrated_variable(system_index: int) -> TemporalVar[float]:
    return TemporalVar(lambda t, y: y[system_index])
