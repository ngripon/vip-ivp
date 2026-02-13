"""
This module contain the domain class for an Initial Value Problem (IVP).

An IVP is defined by the system y'(t) = f(t, y(t)), and an initial point (t0, y0). By using an integration scheme, the
solution y(t) is computed.


"""
from enum import Enum
from numbers import Real
from typing import Callable, Literal, Optional, Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import OdeSolution, RK23, RK45, DOP853, Radau, BDF, LSODA, OdeSolver

# Constants
EPS = np.finfo(float).eps
CROSSING_TOLERANCE = 1e-12

# Types
SystemFun = Callable[[NDArray, NDArray], NDArray] | Callable[[float, NDArray], float]
Direction = Literal["both", "rising", "falling"]
CrossingTriggers = tuple[list[float], ...]


class Crossing:
    def __init__(self, guard: SystemFun, direction: Direction = "both") -> None:
        self.guard = guard
        self.direction = direction

        # Cache current value
        self._current_t = None
        self._current_value = None

    def compute_root(self, t, t_next, sol) -> float | None:
        # Check zero crossing
        if self._current_value is None or t != self._current_t:
            y0 = self.guard(t, sol(t))
        else:
            y0 = self._current_value
        y1 = self.guard(t_next, sol(t_next))
        self._cache_current_value(t_next, y1)

        # Return if there is no crossing
        if not self.check_zero_crossing(y0, y1, self.direction):
            return None

        # Find root
        from scipy.optimize import brentq, bisect

        discontinuous = not isinstance(y0, (float, int))
        if discontinuous:
            return bisect(lambda t_: 1 if self.guard(t_, sol(t_)) else -1, t, t_next, xtol=4 * EPS, rtol=4 * EPS)
        else:
            if abs(self.guard(t, sol(t))) <= CROSSING_TOLERANCE:
                return t
            elif abs(self.guard(t_next, sol(t_next))) <= CROSSING_TOLERANCE:
                return t_next
            return brentq(lambda t_: self.guard(t_, sol(t_)), t, t_next, xtol=4 * EPS, rtol=4 * EPS)

    def _cache_current_value(self, t, value):
        self._current_t = t
        self._current_value = value

    @staticmethod
    def check_zero_crossing(y0: float | bool, y1: float | bool, direction: Direction) -> bool:
        if isinstance(y0, (bool, np.bool)):
            value0 = float(y0)
            value1 = float(y1)
        elif isinstance(y0, Real):
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
    def __init__(self, func: SystemFun | None, action_type: ActionType) -> None:
        self.func = func
        self.action_type = action_type

    def __call__(self, t, y):
        if self.action_type == ActionType.TERMINATE:
            return None
        return self.func(t, y)


class Event:
    def __init__(self, condition: Callable, action: Optional[Action] = None):
        self.condition = condition
        self.action = action

    def evaluate(self, t: float, y: NDArray) -> bool:
        return bool(self.condition(t, y))


class SystemSolution:
    def __init__(self, sol: OdeSolution, t_crossings: Iterable[Iterable[float]]) -> None:
        self.continuous_solution = sol
        self.t_crossings = t_crossings # For each crossing idx, the list of crossing times

    @property
    def timestamps(self)->list[float]:
        return self.continuous_solution.ts


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
            output_bounds: tuple[tuple[SystemFun | None, SystemFun | None], ...],
            crossings: tuple[Crossing, ...] = None,
            events: tuple[Event, ...] = None
    ):
        assert len(derivative_expressions) == len(initial_conditions)
        self.derivatives = derivative_expressions
        self.initial_conditions = initial_conditions
        self.output_bounds = output_bounds
        self.crossings = crossings or []
        self.events = events or []

    @property
    def n_equations(self) -> int:
        return len(self.derivatives)

    @property
    def n_events(self) -> int:
        return len(self.events)

    def solve(self, t_end: float, method: str = "RK45", atol: float = 1e-6, rtol: float = 1e-3,
              verbose: bool = False) -> SystemSolution:

        # Check
        for der_idx, der in enumerate(self.derivatives):
            if der is None:
                raise ValueError(f"Derivative at index {der_idx} is None. Solving aborted.")

        # Init
        solver_method = self.METHODS[method]

        def init_solver(t0, y0):
            return solver_method(self._dy, t0, y0, t_end, vectorized=False, atol=atol, rtol=rtol)

        t0 = 0.0
        # Data to fill
        interpolants = []
        ts = [t0]
        crossing_triggers = tuple([[] for _ in range(len(self.crossings))])

        # Init solver
        solver = init_solver(t0, self.initial_conditions)

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
            sub_sol = self._bound_sol(solver.dense_output())

            if verbose:
                print(f"Computed major step to T = {t} s")

            # CROSSING HANDLING
            tc: float | None = None
            first_crossing_idx: int | None = None
            for c_idx, crossing in enumerate(self.crossings):
                root = crossing.compute_root(t_old, t, sub_sol)
                if root is None:
                    continue
                if tc is None or root < tc:
                    tc = root
                    first_crossing_idx = c_idx
            # If there is a crossing, roll back time
            if tc is not None and tc > t_old:
                t = tc
                solver = init_solver(t, sub_sol(t))
                if verbose:
                    print(f"Crossing detected: Roll back time to T = {t} s")
                crossing_triggers[first_crossing_idx].append(t)

            # EVENT HANDLING
            for event in self.events:
                # Pass if condition is False
                if not event.evaluate(t, sub_sol(t)):
                    continue
                # Apply action
                action = event.action
                if verbose:
                    print(f"Action executed: type = {action.action_type}")
                if action is None:
                    pass
                elif action.action_type == ActionType.UPDATE_SYSTEM:
                    # Update state and restart the solver to handle the discontinuity
                    y = action(t, sub_sol(t))
                    solver = init_solver(t, y)
                elif action.action_type == ActionType.TERMINATE:
                    status = 0
                    break
                elif action.action_type == ActionType.ASSERT:
                    ...
                elif action.action_type == ActionType.SIDE_EFFECT:
                    action(t, sub_sol(t))

            # Update solution
            ts.append(t)
            interpolants.append(sub_sol)

        # End loop
        message = self.MESSAGES[status]
        if verbose:
            print(message)

        ts = np.array(ts)
        sol = OdeSolution(ts, interpolants, alt_segment=True if solver_method in [BDF, LSODA] else False)

        return SystemSolution(sol, crossing_triggers)

    def _dy(self, t, y):
        try:
            return np.array([f(t, y) for f in self.derivatives])
        except RecursionError:
            raise RecursionError(
                "An algebraic loop has been detected."
            )

    def _bound_sol(self, sol):

        def wrapper(t: float | NDArray):
            y = sol(t)
            lower, upper = self._get_bounds(t, y)
            if upper:
                y = np.where(y <= upper, y, upper)
            if lower:
                y = np.where(y >= lower, y, lower)
            return y

        return wrapper

    def _get_bounds(self, t, y):
        upper_bounds = []
        lower_bounds = []
        for der_idx, (lower, upper) in enumerate(self.output_bounds):
            maximum = upper(t, y) if upper is not None else np.full(t.shape, np.inf) if not np.isscalar(t) else np.inf
            minimum = lower(t, y) if lower is not None else np.full(t.shape, -np.inf) if not np.isscalar(t) else -np.inf
            if np.any(minimum > maximum):
                raise ValueError(
                    f"Lower bound {minimum} is greater than upper bound {maximum} a time {t} s for equation {der_idx}")
            upper_bounds.append(maximum)
            lower_bounds.append(minimum)
        return lower_bounds, upper_bounds


def create_system_output_fun(idx: int) -> SystemFun:
    def system_output(_, y: NDArray, i=idx) -> NDArray:
        return y[i]

    return system_output


def create_set_system_output_fun(idx: int, value_fun: Callable[[float, NDArray], float]) -> SystemFun:
    def set_system_output(t: float | NDArray, y: NDArray, i=idx) -> NDArray:
        y_new = np.copy(y)
        y_new[i] = value_fun(t, y)
        return y_new

    return set_system_output
