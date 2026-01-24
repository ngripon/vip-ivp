from src.vip_ivp.application_services.system import IVPSystemMutable

_solver_list: list[IVPSystemMutable] = []


def new_system() -> None:
    _solver_list.append(IVPSystemMutable())


def temporal(value) -> IVPSystemMutable.TemporalVarState:
    return IVPSystemMutable.TemporalVarState(value, _get_current_solver())


def state(x0: float) -> IVPSystemMutable.IntegratedVar:
    return _get_current_solver().add_state(x0)


def solve(t_end: float, method: str = "RK45") -> None:
    _get_current_solver().solve(t_end, method)


def _get_current_solver() -> IVPSystemMutable:
    if not _solver_list:
        new_system()
    return _solver_list[-1]
