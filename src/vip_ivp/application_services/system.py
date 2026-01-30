from typing import TypeVar, Optional

from scipy.integrate import OdeSolution
from numpy.typing import NDArray

from .variables import TemporalVar, IntegratedVar, CrossTriggerVar
from ..domain.system import IVPSystem, EventCondition, Direction, EventTriggers

T = TypeVar("T")


class IVPSystemMutable:
    def __init__(self):
        self.sol: OdeSolution | None = None  # Continuous results function
        self.t_eval: Optional[NDArray] = None
        self.events_trigger: EventTriggers = ()

        self._system: IVPSystem = IVPSystem(tuple(), tuple())

    @property
    def is_solved(self) -> bool:
        return self.sol is not None

    def add_state(self, x0: float) -> "IntegratedVar":
        self._add_equation(None, x0)
        return IntegratedVar(self._system.n_equations - 1, self)

    def add_crossing_detection(self, variable: TemporalVar[float], direction: Direction) -> CrossTriggerVar:
        # Create variable
        cross_trigger = CrossTriggerVar(variable, direction, self._system.n_events - 1, self)

        # Update system
        events = list(self._system.event_conditions)
        new_event = EventCondition(cross_trigger.guard, cross_trigger.direction)
        events.append(new_event)

        self._set_system(
            IVPSystem(self._system.derivatives, self._system.initial_conditions, events)
        )

        return cross_trigger

    # def add_event(self, crossing_variable: CrossTriggerVar, action=None) -> None:
    #     assert isinstance(crossing_variable, CrossTriggerVar), ("Condition should be a cross-trigger variable. "
    #                                                             "Use the .crosses(value) method instead.")
    #     events = list(self._system.event_conditions)
    #     new_event = EventCondition(crossing_variable.guard)
    #     events.append(new_event)
    #
    #     self._set_system(
    #         IVPSystem(self._system.derivatives, self._system.initial_conditions, events)
    #     )

    def solve(self, t_end: float, method: str = "RK45", t_eval: list[float] = None) -> None:
        self.t_eval, self.sol, self.events_trigger = self._system.solve(t_end, method)
        if t_eval is not None:
            self.t_eval = t_eval

    def set_derivative(self, variable: TemporalVar[float], eq_idx: int) -> None:
        derivatives = list(self._system.derivatives)
        derivatives[eq_idx] = variable
        self._set_system(IVPSystem(tuple(derivatives), tuple(self._system.initial_conditions)))

    def _set_system(self, system: IVPSystem) -> None:
        self._system = system
        self.sol = None
        self.t_eval = None
        self.events_trigger = None

    def _add_equation(self, variable: Optional[TemporalVar[float]], x0: float) -> None:
        derivatives = list(self._system.derivatives)
        initial_conditions = list(self._system.initial_conditions)

        derivatives.append(variable)
        initial_conditions.append(x0)

        self._set_system(IVPSystem(tuple(derivatives), tuple(initial_conditions)))
