"""
This module contain the domain class for an Initial Value Problem (IVP).

An IVP is defined by the system y'(t) = f(t, y(t)), and an initial point (t0, y0). By using an integration scheme, the
solution y(t) is computed.


"""
from src.vip_ivp.domain.variables import TemporalVar


class IVPSystem:
    def __init__(self, derivative_expressions:tuple[TemporalVar], initial_conditions: tuple[float]):
        assert len(derivative_expressions) == len(initial_conditions)
        self._derivatives = derivative_expressions
        self._initial_conditions = initial_conditions