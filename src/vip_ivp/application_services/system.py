from typing import TypeVar, Optional

import numpy as np
from scipy.integrate import OdeSolution
from numpy.typing import NDArray

from .variables import TemporalVar, IntegratedVar, CrossTriggerVar
from ..domain.system import IVPSystem, EventCondition, Direction, EventTriggers, Event, Action

T = TypeVar("T")


class IVPSystemMutable:
    def __init__(self):
        # Results
        self.sol: OdeSolution | None = None  # Continuous results function
        self.t_eval: Optional[NDArray] = None
        self.events_trigger: EventTriggers = ()

        # System inputs
        self._derivatives: list[Optional[TemporalVar]] = []
        self._initial_conditions: list[float] = []
        self._events: list[Event] = []

    @property
    def is_solved(self) -> bool:
        return self.sol is not None

    @property
    def n_equations(self) -> int:
        return len(self._derivatives)

    @property
    def n_events(self) -> int:
        return len(self._events)

    def solve(self, t_end: float, method: str = "RK45", t_eval: list[float] = None) -> None:
        system = IVPSystem(tuple(self._derivatives), tuple(self._initial_conditions),
                           tuple(self._events))

        self.t_eval, self.sol, self.events_trigger = system.solve(t_end, method)
        if t_eval is not None:
            # Add trigger instants to t_eval
            new_t_eval = list(t_eval)
            [new_t_eval.extend(te) for te in self.events_trigger]
            new_t_eval = np.sort(new_t_eval)
            self.t_eval = new_t_eval

    def add_state(self, x0: float) -> "IntegratedVar":
        self._derivatives.append(None)
        self._initial_conditions.append(x0)
        return IntegratedVar(self.n_equations - 1, self)

    def add_crossing_detection(self, variable: TemporalVar[float], direction: Direction) -> CrossTriggerVar:
        # Create variable
        cross_trigger = CrossTriggerVar(variable, direction, self.n_events, self)

        # Add to system
        new_event_condition = EventCondition(cross_trigger.guard, cross_trigger.direction)
        new_event = Event(new_event_condition)
        self._events.append(new_event)

        return cross_trigger

    def set_event_action(self, condition: CrossTriggerVar, action: Action) -> None:
        event_idx = condition.event_idx
        self._events[event_idx].action = action

    def set_derivative(self, variable: TemporalVar[float], eq_idx: int) -> None:
        self._derivatives[eq_idx] = variable
