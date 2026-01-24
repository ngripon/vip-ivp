from typing import TypeVar, Generic

from src.vip_ivp.application_services.system import IVPSystemMutable
from src.vip_ivp.domain.variables import TemporalVar

T = TypeVar("T")


class VariableState(Generic[T]):
    def __init__(self, variable: TemporalVar, system: IVPSystemMutable):
        self._variable = variable
        self._system = system

    def __call__(self, t):
        if self._system.results is not None:
            return self._variable(t, self._system.results(t))
        else:
            raise Exception(
                "The system has not been solved.\n"
                "Call the solve() method before inquiring the variable values."
            )

