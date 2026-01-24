"""
This module contain the domain class for an Initial Value Problem (IVP).

An IVP is defined by the system y'(t) = f(t, y(t)), and an initial point (t0, y0). By using an integration scheme, the
solution y(t) is computed.


"""
from typing import Optional

import numpy as np
from scipy.integrate import OdeSolution, solve_ivp

from src.vip_ivp.domain.variables import TemporalVar


class IVPSystem:
    def __init__(self, derivative_expressions: tuple[Optional[TemporalVar[float]], ...],
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


def create_integrated_variable(equation_idx: int) -> TemporalVar[float]:
    return TemporalVar(lambda t, y: y[equation_idx])
