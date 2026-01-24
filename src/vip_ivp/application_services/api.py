from src.vip_ivp.application_services.system import IVPSystemMutable, TemporalVarState, IntegratedVar

_solver_list: list[IVPSystemMutable] = []


def new_system() -> None:
    _solver_list.append(IVPSystemMutable())


def temporal(value) -> TemporalVarState:
    return TemporalVarState(value, _get_current_system())


def state(x0: float) -> IntegratedVar:
    return _get_current_system().add_state(x0)


def solve(t_end: float, method: str = "RK45") -> None:
    _get_current_system().solve(t_end, method)


# Post-processing
def plot(*variables: TemporalVarState) -> None:
    # Check
    if not variables:
        raise ValueError("No variable provided")
    system = _get_current_system()
    if not system.is_solved:
        raise RuntimeError("System is not solved")

    # Plot
    import matplotlib.pyplot as plt

    timestamps = system.t_eval
    # Plot data
    plt.figure("Results")
    for variable in variables:
        plt.plot(timestamps, variable.values, label="TODO")
    # Label and axis
    plt.title("Simulation results")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.xlim(timestamps[0], timestamps[-1])
    plt.grid()
    plt.tight_layout()
    plt.show()


def _get_current_system() -> IVPSystemMutable:
    if not _solver_list:
        new_system()
    return _solver_list[-1]
