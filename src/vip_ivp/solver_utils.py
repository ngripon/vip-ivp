import inspect
import numpy as np
from scipy.integrate._ivp.bdf import BDF
from scipy.integrate._ivp.radau import Radau
from scipy.integrate._ivp.rk import RK23, RK45, DOP853
from scipy.integrate._ivp.lsoda import LSODA
from scipy.optimize import OptimizeResult
from scipy.integrate._ivp.common import EPS, OdeSolution
from scipy.integrate._ivp.base import OdeSolver

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


def prepare_events(events):
    """Standardize event functions and extract attributes."""
    if callable(events):
        events = (events,)

    max_events = np.empty(len(events))
    direction = np.empty(len(events))
    for i, event in enumerate(events):
        terminal = getattr(event, 'terminal', None)
        direction[i] = getattr(event, 'direction', 0)

        message = ('The `terminal` attribute of each event '
                   'must be a boolean or positive integer.')
        if terminal is None or terminal == 0:
            max_events[i] = np.inf
        elif int(terminal) == terminal and terminal > 0:
            max_events[i] = terminal
        else:
            raise ValueError(message)

    return events, max_events, direction


def solve_event_equation(event, sol, t_old, t):
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
    return brentq(lambda t: event(t, sol(t)), t_old, t,
                  xtol=4 * EPS, rtol=4 * EPS)


def handle_events(sol, events, active_events, event_count, max_events,
                  t_old, t):
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
    roots = [solve_event_equation(events[event_index], sol, t_old, t)
             for event_index in active_events]

    roots = np.asarray(roots)

    if np.any(event_count[active_events] >= max_events[active_events]):
        if t > t_old:
            order = np.argsort(roots)
        else:
            order = np.argsort(-roots)
        active_events = active_events[order]
        roots = roots[order]
        t = np.nonzero(event_count[active_events]
                       >= max_events[active_events])[0][0]
        active_events = active_events[:t + 1]
        roots = roots[:t + 1]
        terminate = True
    else:
        terminate = False

    return active_events, roots, terminate


def find_active_events(g, g_new, direction):
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
    g, g_new = np.asarray(g), np.asarray(g_new)
    up = (g <= 0) & (g_new >= 0)
    down = (g >= 0) & (g_new <= 0)
    either = up | down
    mask = (up & (direction > 0) |
            down & (direction < 0) |
            either & (direction == 0))

    return np.nonzero(mask)[0]
