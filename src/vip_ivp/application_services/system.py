from scipy.integrate import OdeSolution

from src.vip_ivp.domain.system import IVPSystem, create_integrated_variable

from typing import TypeVar, Generic, Optional

from src.vip_ivp.domain.variables import TemporalVar

T = TypeVar("T")


class IVPSystemMutable:
    def __init__(self):
        self._system: IVPSystem = IVPSystem(tuple(), tuple())
        self.sol: OdeSolution | None = None  # Continuous results function

    def add_state(self, x0: float) -> "IVPSystemMutable.IntegratedVar":
        self._add_equation(None, x0)
        return self.IntegratedVar(self._system.n_equations, self)

    def _set_system(self, system: IVPSystem) -> None:
        self._system = system
        self.sol = None

    def _set_derivative(self, variable: TemporalVar[float], eq_idx: int) -> None:
        derivatives = list(self._system.derivatives)
        derivatives[eq_idx] = variable
        self._set_system(IVPSystem(tuple(derivatives), tuple(self._system.initial_conditions)))

    def _add_equation(self, variable: Optional[TemporalVar[float]], x0: float) -> None:
        derivatives = list(self._system.derivatives)
        initial_conditions = list(self._system.initial_conditions)

        derivatives.append(variable)
        initial_conditions.append(x0)

        self._set_system(IVPSystem(tuple(derivatives), tuple(self._system.initial_conditions)))

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

    class IntegratedVar(TemporalVarState[float]):
        def __init__(self, index: int, system: "IVPSystemMutable"):
            variable = create_integrated_variable(index)
            super().__init__(variable, system)
            self._derivative: "IVPSystemMutable.TemporalVarState[float] | None" = None
            self._eq_idx: int = index

        @property
        def derivative(self):
            return self._derivative

        @derivative.setter
        def derivative(self, value: "IVPSystemMutable.TemporalVarState[float]"):
            self._derivative = value
            self._system._set_derivative(value._variable, self._eq_idx)
