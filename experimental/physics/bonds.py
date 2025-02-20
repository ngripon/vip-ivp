import vip_ivp as vip


class Bond:
    def __init__(self, flow, effort):
        self._flow = flow
        self._effort = effort

    @property
    def effort(self):
        return self._effort

    @effort.setter
    def effort(self, value: vip.TemporalVar):
        self._effort = value

    @property
    def flow(self):
        return self._flow

    @flow.setter
    def flow(self, value: vip.TemporalVar):
        self._flow = value

    @property
    def power(self):
        return self.flow * self.effort


def rename_effort(new_name):
    def decorator(cls):
        if hasattr(cls, "effort"):
            setattr(cls, new_name, cls.effort)
        return cls

    return decorator


def create_bond_types(name: str, effort_name: str, flow_name: str):
    @rename_effort(effort_name)
    class BondFlow(Bond):
        def __init__(self, value: float = 0):
            super().__init__(value, vip.loop_node())

        @Bond.effort.setter
        def effort(self, value):
            self.effort.loop_into(value)

    @rename_effort(effort_name)
    class BondEffort(Bond):
        def __init__(self, value: float = 0):
            super().__init__(vip.loop_node(), value)

        @Bond.flow.setter
        def flow(self, value):
            self.flow.loop_into(value)

    BondEffort.__name__ = f"{name}Effort"
    setattr(BondEffort, flow_name, BondEffort.flow)
    BondFlow.__name__ = f"{name}Flow"
    setattr(BondFlow, flow_name, BondFlow.flow)

    return BondEffort, BondFlow


Mechanical1DEffort, Mechanical1DFlow = create_bond_types("Mechanical1D", "force", "speed")
