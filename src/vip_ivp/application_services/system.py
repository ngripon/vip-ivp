from scipy.integrate import OdeSolution

from src.vip_ivp.domain.system import IVPSystem


class IVPSystemMutable:
    def __init__(self):
        self._system: IVPSystem = IVPSystem(tuple(), tuple())
        self.results: OdeSolution | None = None

    def _set_system(self, system: IVPSystem):
        self._system = system
        self.results = None
