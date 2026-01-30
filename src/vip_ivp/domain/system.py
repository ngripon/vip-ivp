"""
This module contain the domain class for an Initial Value Problem (IVP).

An IVP is defined by the system y'(t) = f(t, y(t)), and an initial point (t0, y0). By using an integration scheme, the
solution y(t) is computed.


"""
from enum import Enum
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import OdeSolution, RK23, RK45, DOP853, Radau, BDF, LSODA, OdeSolver

EPS = np.finfo(float).eps
CROSSING_TOLERANCE = 1e-12

SystemFun = Callable[[float | NDArray, NDArray], NDArray | float]
Direction = Literal["both", "rising", "falling"]


class EventCondition:
    def __init__(self, condition: SystemFun, direction: Direction = "both") -> None:
        self.condition = condition
        self.direction = direction

        # Cache current value
        self._current_t = None
        self._current_value = None

    def compute_root(self, t, t_next, sol) -> float | None:
        # Check zero crossing
        if self._current_value is None or t != self._current_t:
            y0 = self.condition(t, sol(t))
        else:
            y0 = self._current_value
        y1 = self.condition(t_next, sol(t_next))

        # Return if there is no crossing
        if not self.check_zero_crossing(y0, y1, self.direction):
            return None

        # Find root
        from scipy.optimize import brentq, bisect

        discontinuous = not isinstance(y0, (float, int))
        if discontinuous:
            return bisect(lambda t_: 1 if self.condition(t_, sol(t_)) else -1, t, t_next, xtol=4 * EPS, rtol=4 * EPS)
        else:
            if abs(self.condition(t, sol(t))) <= CROSSING_TOLERANCE:
                return t
            elif abs(self.condition(t_next, sol(t_next))) <= CROSSING_TOLERANCE:
                return t_next
            return brentq(lambda t_: self.condition(t_, sol(t_)), t, t_next, xtol=4 * EPS, rtol=4 * EPS)

    def _cache_current_value(self, t, value):
        self._current_t = t
        self._current_value = value

    @staticmethod
    def check_zero_crossing(y0: float | bool, y1: float | bool, direction: Direction) -> bool:
        assert type(y0) is type(y1)

        if isinstance(y0, (bool, np.bool)):
            value0 = float(y0)
            value1 = float(y1)
        elif isinstance(y0, (float, int)):
            value0 = np.sign(y0)
            value1 = np.sign(y1)
        else:
            raise ValueError(f"y0 must be either bool or float. Got {type(y0)}")

        if direction == "both":
            zero_crossing = value0 != value1
        elif direction == "rising":
            zero_crossing = value0 != value1 and value0 < value1
        elif direction == "falling":
            zero_crossing = value0 != value1 and value0 > value1
        return zero_crossing


class ActionType(Enum):
    UPDATE_SYSTEM = 0
    TERMINATE = 1
    ASSERT = 2
    SIDE_EFFECT = 3


class Action:
    def __init__(self, func: SystemFun, action_type: ActionType) -> None:
        self.func = func
        self.action_type = action_type

    def __call__(self, t, y):
        return self.func(t, y)


class Event:
    def __init__(self, condition: EventCondition, action: Action):
        self.condition = condition
        self.action = action


class IVPSystem:
    METHODS: dict[str, type[OdeSolver]] = {'RK23': RK23,
                                           'RK45': RK45,
                                           'DOP853': DOP853,
                                           'Radau': Radau,
                                           'BDF': BDF,
                                           'LSODA': LSODA}
    MESSAGES = {0: "The solver successfully reached the end of the integration interval.",
                1: "A termination event occurred."}

    def __init__(
            self,
            derivative_expressions: tuple[SystemFun, ...],
            initial_conditions: tuple[float, ...],
            event_conditions: tuple[EventCondition, ...] = None,
    ):
        assert len(derivative_expressions) == len(initial_conditions)
        self.derivatives = derivative_expressions
        self.initial_conditions = initial_conditions
        self.event_conditions = event_conditions or []

    @property
    def n_equations(self) -> int:
        return len(self.derivatives)

    def solve(self, t_end: float, method: str = "RK45") -> tuple[NDArray, OdeSolution]:
        # Check
        for der_idx, der in enumerate(self.derivatives):
            if der is None:
                raise ValueError(f"Derivative at index {der_idx} is None. Solving aborted.")

        # Solve
        # result = solve_ivp(self._dy, [0, t_end], self.initial_conditions, method=method, dense_output=True)

        # Init
        t0 = 0.0
        # Data to fill
        interpolants = []
        ts = [t0]
        # Init solver
        solver_method = self.METHODS[method]
        solver = solver_method(self._dy, t0, self.initial_conditions, t_end, vectorized=False)

        # Step loop
        status = None
        while status is None:
            # Do step
            message = solver.step()

            if solver.status == 'finished':
                status = 0
            elif solver.status == 'failed':
                status = -1
                break

            t_old = solver.t_old
            t = solver.t
            y = solver.y
            sub_sol = solver.dense_output()
            interpolants.append(sub_sol)

            # Handle events
            te: float = None
            first_event: EventCondition | None = None
            for ec in self.event_conditions:
                root = ec.compute_root(t_old, t, sub_sol)
                if root is None:
                    continue
                if te is None or root < te:
                    te = root
                    first_event = ec

            # If there are events, roll back time to the first event
            if te is not None:
                t = te

            ts.append(t)
        # End loop
        message = self.MESSAGES[status]

        ts = np.array(ts)
        sol = OdeSolution(ts, interpolants, alt_segment=True if solver_method in [BDF, LSODA] else False)

        return ts, sol

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
