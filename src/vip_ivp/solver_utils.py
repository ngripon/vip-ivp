import inspect
from typing import TYPE_CHECKING, List

import numpy as np
from scipy.integrate._ivp.bdf import BDF
from scipy.integrate._ivp.radau import Radau
from scipy.integrate._ivp.rk import RK23, RK45, DOP853
from scipy.integrate._ivp.lsoda import LSODA
from scipy.optimize import OptimizeResult
from scipy.integrate._ivp.common import EPS, OdeSolution
from scipy.integrate._ivp.base import OdeSolver

if TYPE_CHECKING:
    from src.vip_ivp.base import Event

METHODS = {'RK23': RK23,
           'RK45': RK45,
           'DOP853': DOP853,
           'Radau': Radau,
           'BDF': BDF,
           'LSODA': LSODA}

MESSAGES = {0: "The solver successfully reached the end of the integration interval.",
            1: "A termination event occurred."}


class OdeResult(OptimizeResult):
    pass


def solve_event_equation(event, sol, t_old, t, is_discrete: bool = False, t_eval: float = None):
    """Solve an equation corresponding to an ODE event.

    The equation is ``event(t, y(t)) = 0``, here ``y(t)`` is known from an
    ODE solver using some sort of interpolation. It is solved by
    `scipy.optimize.brentq` with xtol=atol=4*EPS.

    Parameters
    ----------
    event : callable
        Function ``event(t, y)``.
    sol : callable
        Function ``sol(t)`` which evaluates an ODE solution between `t_old`
        and  `t`.
    t_old, t : float
        Previous and new values of time. They will be used as a bracketing
        interval.

    Returns
    -------
    root : float
        Found solution.
    """
    from scipy.optimize import brentq
    if is_discrete:
        if t_eval is None:
            return t
        else:
            t_eval_i_new = np.searchsorted(t_eval, t, side="right")
            t_eval_step = t_eval[:t_eval_i_new]
            t_eval_step = t_eval_step[t_eval_step > t_old]
            initial_state = event(t_old, sol(t_old))
            new_state = event(t, sol(t))
            assert initial_state != new_state
            return next((t for t in t_eval_step if event(t, sol(t)) != initial_state), t)
    else:
        return brentq(lambda t: event(t, sol(t)), t_old, t,
                      xtol=4 * EPS, rtol=4 * EPS)


def is_discrete(event: "Event") -> bool:
    return event.function.output_type in (str, bool, np.bool)


def handle_events(sol, events, active_events_indices, t_old, t, t_eval):
    """Helper function to handle events.

    Parameters
    ----------
    sol : DenseOutput
        Function ``sol(t)`` which evaluates an ODE solution between `t_old`
        and  `t`.
    events : list of callables, length n_events
        Event functions with signatures ``event(t, y)``.
    active_events : ndarray
        Indices of events which occurred.
    event_count : ndarray
        Current number of occurrences for each event.
    max_events : ndarray, shape (n_events,)
        Number of occurrences allowed for each event before integration
        termination is issued.
    t_old, t : float
        Previous and new values of time.

    Returns
    -------
    root_indices : ndarray
        Indices of events which take zero between `t_old` and `t` and before
        a possible termination.
    roots : ndarray
        Values of t at which events occurred.
    terminate : bool
        Whether a terminal event occurred.
    """
    active_events = [events[idx] for idx in active_events_indices]
    roots = [solve_event_equation(e, sol, t_old, t, is_discrete(e), t_eval)
             for e in active_events]
    roots = np.asarray(roots)
    return active_events, roots


def find_active_events(events, sol, t_eval, t, t_old):
    """Find which event occurred during an integration step.

    Parameters
    ----------
    g, g_new : array_like, shape (n_events,)
        Values of event functions at a current and next points.
    direction : ndarray, shape (n_events,)
        Event "direction" according to the definition in `solve_ivp`.

    Returns
    -------
    active_events : ndarray
        Indices of events which occurred during the step.
    """

    g = [e(t_old, sol(t_old)) for e in events]
    direction = np.array([e.direction for e in events])

    if t_eval is None:
        t_list = []
    else:
        t_eval_i_new = np.searchsorted(t_eval, t, side="right")
        t_eval_step = t_eval[:t_eval_i_new]
        t_list = t_eval_step[t_eval_step > t_old]


    for t_ev in [*t_list, t]:
        g_new = [e(t_ev, sol(t_ev)) for e in events]
        active_events_indices = find_active_events_in_step(g, g_new, direction)
        if active_events_indices.size > 0:
            print(g_new)
            return active_events_indices, t_ev
    return np.array([]), t


def find_active_events_in_step(g, g_new, direction):
    """Find which event occurred during an integration step.

    Parameters
    ----------
    g, g_new : array_like, shape (n_events,)
        Values of event functions at a current and next points.
    direction : ndarray, shape (n_events,)
        Event "direction" according to the definition in `solve_ivp`.

    Returns
    -------
    active_events : ndarray
        Indices of events which occurred during the step.
    """

    # replace False values by -1 because False being equal 0 breaks the 0 crossing detection.
    g = [x if not isinstance(x, (bool, np.bool)) or x == True else -1 for x in g]
    g_new = [x if not isinstance(x, (bool, np.bool)) or x == True else -1 for x in g_new]

    g, g_new, direction = np.asarray(g), np.asarray(g_new), np.asarray(direction)
    up = (g <= 0) & (g_new >= 0)
    down = (g >= 0) & (g_new <= 0)
    either = up | down
    mask = (up & (direction > 0) |
            down & (direction < 0) |
            either & (direction == 0))
    active_events_indices = np.nonzero(mask)[0]
    return active_events_indices
