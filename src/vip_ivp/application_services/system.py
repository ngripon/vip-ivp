from scipy.integrate import OdeSolution

from src.vip_ivp.domain.system import IVPSystem, create_integrated_variable

from typing import TypeVar, Generic, Optional

from src.vip_ivp.domain.variables import TemporalVar

T = TypeVar("T")


class IVPSystemMutable:
    def __init__(self):
        self.sol: OdeSolution | None = None  # Continuous results function

        self._system: IVPSystem = IVPSystem(tuple(), tuple())

    def add_state(self, x0: float) -> "IntegratedVar":
        self._add_equation(None, x0)
        return IntegratedVar(self._system.n_equations - 1, self)

    def solve(self, t_end: float, method: str = "RK45") -> None:
        self.sol = self._system.solve(t_end, method)

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
            raise Exception(
                "The system has not been solved.\n"
                "Call the solve() method before inquiring the variable values."
            )

    @staticmethod
    def _get_variable(value) -> TemporalVar:
        if isinstance(value, TemporalVarState):
            return value._variable
        return value

    # Addition
    def __add__(self, other):
        return TemporalVarState(self._variable + self._get_variable(other), self._system)

    def __radd__(self, other):
        return TemporalVarState(self._get_variable(other) + self._variable, self._system)

    # Subtraction
    def __sub__(self, other):
        return TemporalVarState(self._variable - self._get_variable(other), self._system)

    def __rsub__(self, other):
        return TemporalVarState(self._get_variable(other) - self._variable, self._system)

    # Multiplication
    def __mul__(self, other):
        return TemporalVarState(self._variable * self._get_variable(other), self._system)

    def __rmul__(self, other):
        return TemporalVarState(self._get_variable(other) * self._variable, self._system)


class IntegratedVar(TemporalVarState[float]):
    def __init__(self, index: int, system: "IVPSystemMutable"):
        variable = create_integrated_variable(index)
        super().__init__(variable, system)
        self._derivative: TemporalVarState[float] | None = None
        self._eq_idx: int = index

    @property
    def derivative(self):
        return self._derivative

    @derivative.setter
    def derivative(self, value: TemporalVarState[float]):
        self._derivative = value
        self._system.set_derivative(value._variable, self._eq_idx)
