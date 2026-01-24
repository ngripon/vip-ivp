import numpy as np
from scipy.integrate import OdeSolution

from src.vip_ivp.domain.system import IVPSystem, create_integrated_variable

from typing import TypeVar, Generic, Optional
from numpy.typing import NDArray

from src.vip_ivp.domain.variables import TemporalVar

T = TypeVar("T")


class IVPSystemMutable:
    def __init__(self):
        self.sol: OdeSolution | None = None  # Continuous results function
        self.t_eval: Optional[NDArray] = None

        self._system: IVPSystem = IVPSystem(tuple(), tuple())

    @property
    def is_solved(self) -> bool:
        return self.sol is not None

    def add_state(self, x0: float) -> "IntegratedVar":
        self._add_equation(None, x0)
        return IntegratedVar(self._system.n_equations - 1, self)

    def solve(self, t_end: float, method: str = "RK45") -> None:
        self.sol = self._system.solve(t_end, method)
        self.t_eval = np.linspace(0, t_end, 100)

    def set_derivative(self, variable: TemporalVar[float], eq_idx: int) -> None:
        derivatives = list(self._system.derivatives)
        derivatives[eq_idx] = variable
        self._set_system(IVPSystem(tuple(derivatives), tuple(self._system.initial_conditions)))

    def _set_system(self, system: IVPSystem) -> None:
        self._system = system
        self.sol = None

    def _add_equation(self, variable: Optional[TemporalVar[float]], x0: float) -> None:
        derivatives = list(self._system.derivatives)
        initial_conditions = list(self._system.initial_conditions)

        derivatives.append(variable)
        initial_conditions.append(x0)

        self._set_system(IVPSystem(tuple(derivatives), tuple(initial_conditions)))


class TemporalVarState(Generic[T]):
    def __init__(self, variable: TemporalVar, system: "IVPSystemMutable"):
        self._variable = variable
        self._system = system

    def __call__(self, t):
        if self._system.sol is not None:
            return self._variable(t, self._system.sol(t))
        else:
            raise RuntimeError(
                "The system has not been solved.\n"
                "Call the solve() method before inquiring the variable values."
            )

    @property
    def values(self):
        return self(self._system.t_eval)

    @property
    def t(self):
        return self._system.t_eval

    @staticmethod
    def _unwrap(value) -> TemporalVar:
        if isinstance(value, TemporalVarState):
            return value._variable
        return value

    # Addition
    def __add__(self, other):
        return TemporalVarState(self._variable + self._unwrap(other), self._system)

    def __radd__(self, other):
        return TemporalVarState(self._unwrap(other) + self._variable, self._system)

    # Subtraction
    def __sub__(self, other):
        return TemporalVarState(self._variable - self._unwrap(other), self._system)

    def __rsub__(self, other):
        return TemporalVarState(self._unwrap(other) - self._variable, self._system)

    # Multiplication
    def __mul__(self, other):
        return TemporalVarState(self._variable * self._unwrap(other), self._system)

    def __rmul__(self, other):
        return TemporalVarState(self._unwrap(other) * self._variable, self._system)

    # True division
    def __truediv__(self, other):
        return TemporalVarState(self._variable / self._unwrap(other), self._system)

    def __rtruediv__(self, other):
        return TemporalVarState(self._unwrap(other) / self._variable, self._system)

    # Floor division
    def __floordiv__(self, other):
        return TemporalVarState(self._variable // self._unwrap(other), self._system)

    def __rfloordiv__(self, other):
        return TemporalVarState(self._unwrap(other) // self._variable, self._system)

    # Modulo
    def __mod__(self, other):
        return TemporalVarState(self._variable % self._unwrap(other), self._system)

    def __rmod__(self, other):
        return TemporalVarState(self._unwrap(other) % self._variable, self._system)

    # Power
    def __pow__(self, other):
        return TemporalVarState(self._variable ** self._unwrap(other), self._system)

    def __rpow__(self, other):
        return TemporalVarState(self._unwrap(other) ** self._variable, self._system)

    # Unary plus
    def __pos__(self):
        return TemporalVarState(self._variable, self._system)

    # Unary minus
    def __neg__(self):
        return TemporalVarState(-self._variable, self._system)

    # Absolute value
    def __abs__(self):
        return TemporalVarState(abs(self._variable), self._system)

    # Logical
    def __eq__(self, other):
        return TemporalVarState(self._variable == self._unwrap(other), self._system)

    def __ne__(self, other):
        return TemporalVarState(self._variable != self._unwrap(other), self._system)

    def __lt__(self, other):
        return TemporalVarState(self._variable < self._unwrap(other), self._system)

    def __le__(self, other):
        return TemporalVarState(self._variable <= self._unwrap(other), self._system)

    def __gt__(self, other):
        return TemporalVarState(self._variable > self._unwrap(other), self._system)

    def __ge__(self, other):
        return TemporalVarState(self._variable >= self._unwrap(other), self._system)

    def __and__(self, other) -> "TemporalVarState[bool]":
        return TemporalVarState(self._variable & self._unwrap(other), self._system)

    def __rand__(self, other) -> "TemporalVarState[bool]":
        return TemporalVarState(self._unwrap(other) & self._variable, self._system)

    def __or__(self, other) -> "TemporalVarState[bool]":
        return TemporalVarState(self._variable | self._unwrap(other), self._system)

    def __ror__(self, other) -> "TemporalVarState[bool]":
        return TemporalVarState(self._unwrap(other) | self._variable, self._system)

    def __xor__(self, other) -> "TemporalVarState[bool]":
        return TemporalVarState(self._variable ^ self._unwrap(other), self._system)

    def __rxor__(self, other) -> "TemporalVarState[bool]":
        return TemporalVarState(self._unwrap(other) ^ self._variable, self._system)

    def __invert__(self) -> "TemporalVar[bool]":
        return TemporalVar(~self._variable)


class IntegratedVar(TemporalVarState[float]):
    def __init__(self, index: int, system: "IVPSystemMutable"):
        variable = create_integrated_variable(index)
        super().__init__(variable, system)
        self._derivative: Optional[TemporalVarState[float]] = None
        self._eq_idx: int = index

    @property
    def derivative(self):
        return self._derivative

    @derivative.setter
    def derivative(self, value: TemporalVarState[float]):
        self._derivative = value
        self._system.set_derivative(value._variable, self._eq_idx)
