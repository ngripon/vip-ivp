import inspect
from typing import TypeVar, Optional, Callable

import numpy as np
from scipy.integrate import OdeSolution
from numpy.typing import NDArray

from .variables import TemporalVar, IntegratedVar, CrossTriggerVar
from ..domain.system import IVPSystem, Crossing, Direction, CrossingTriggers, Event, Action, ActionType

T = TypeVar("T")

SideEffectFun = Callable[[], None] | Callable[[float], None]


class IVPSystemMutable:
    N_T_EVAL_DEFAULT = 500

    def __init__(self):
        # Results
        self.sol: OdeSolution | None = None  # Continuous results function
        self.t_eval: Optional[NDArray] = None
        self.crossing_triggers: CrossingTriggers = ()

        # System inputs
        self.derivatives: list[Optional[TemporalVar]] = []
        self.initial_conditions: list[float] = []
        self.bounds: list[tuple[None | TemporalVar[float], None | TemporalVar[float]]] = []
        self._events: list[Event] = []
        self._crossings: list[Crossing] = []

    @property
    def is_solved(self) -> bool:
        return self.sol is not None

    @property
    def n_equations(self) -> int:
        return len(self.derivatives)

    @property
    def n_events(self) -> int:
        return len(self._events)

    def solve(self, t_end: float, method: str = "RK45", t_eval: list[float] = None, step_eval: float = None,
              atol: float = 1e-6, rtol: float = 1e-3, verbose: bool = False) -> None:

        system = IVPSystem(
            tuple(self.derivatives),
            tuple(self.initial_conditions),
            tuple(self.bounds),
            tuple(self._crossings),
            tuple(self._events),
            on_crossing_detection=self._update_crossing_triggers,
            on_solution_update=self._update_sol
        )

        self.t_eval, self.sol, self.crossing_triggers = system.solve(t_end, method, atol, rtol, verbose)

        if t_eval is not None:
            # Add trigger instants to t_eval
            new_t_eval = np.array(t_eval)
            new_t_eval = np.unique(np.concatenate((new_t_eval, *self.crossing_triggers)))
            self.t_eval = new_t_eval
        elif step_eval is not None:
            new_t_eval = np.arange(self.t_eval[0], self.t_eval[-1], step_eval)
            new_t_eval = np.unique(np.concatenate((new_t_eval, *self.crossing_triggers)))
            self.t_eval = new_t_eval
        elif len(self.t_eval) < self.N_T_EVAL_DEFAULT:
            new_t_eval = np.linspace(self.t_eval[0], self.t_eval[-1], self.N_T_EVAL_DEFAULT)
            new_t_eval = np.unique(np.concatenate((new_t_eval, self.t_eval)))
            self.t_eval = new_t_eval

    def add_state(self, x0: float, lower=None, upper=None) -> "IntegratedVar":
        self.derivatives.append(None)
        self.initial_conditions.append(x0)
        self.bounds.append((lower, upper))
        return IntegratedVar(self.n_equations - 1, self)

    def add_crossing_detection(self, variable: TemporalVar[float], direction: Direction) -> CrossTriggerVar:
        # Create variable
        cross_trigger = CrossTriggerVar(variable, direction, self.n_events, self)

        # Add to system
        new_crossing = Crossing(cross_trigger.guard, cross_trigger.direction)
        self._crossings.append(new_crossing)

        return cross_trigger

    def add_event(self, condition: CrossTriggerVar | TemporalVar[bool], action: Action | SideEffectFun) -> None:
        if not isinstance(action, Action):
            n_args = len(inspect.signature(action).parameters)
            if n_args == 0:
                action_fun = lambda t, y: action()
            elif n_args == 1:
                action_fun = lambda t, y: action(t)
            else:
                action_fun = action
            new_action = Action(action_fun, ActionType.SIDE_EFFECT)
        else:
            new_action = action

        self._events.append(Event(condition, new_action))

    def set_derivative(self, variable: TemporalVar[float], eq_idx: int) -> None:
        self.derivatives[eq_idx] = variable

    def set_bound(self, variable: TemporalVar[float], eq_idx: int, is_lower: bool) -> None:
        current_bounds = self.bounds[eq_idx]
        if is_lower:
            self.bounds[eq_idx] = (variable, current_bounds[1])
        else:
            self.bounds[eq_idx] = (current_bounds[0], variable)

    def _update_crossing_triggers(self, crossing_triggers: CrossingTriggers) -> None:
        self.crossing_triggers = crossing_triggers

    def _update_sol(self, sol: OdeSolution) -> None:
        self.sol = sol
