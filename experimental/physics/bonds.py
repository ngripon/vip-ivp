import enum

import vip_ivp as vip


class Intensive(enum.Enum):
    EFFORT = 0
    FLOW = 1


class Bond:
    def __init__(self, intensive_variable: Intensive = Intensive.FLOW):
        self._flow = vip.loop_node()
        self._effort = vip.loop_node()
        self.intensive_variable = intensive_variable

    @property
    def effort(self) -> vip.TemporalVar:
        return self._effort

    @effort.setter
    def effort(self, value: vip.TemporalVar):
        self._effort.loop_into(value)

    @property
    def flow(self) -> vip.TemporalVar:
        return self._flow

    @flow.setter
    def flow(self, value: vip.TemporalVar):
        self._flow.loop_into(value)

    @property
    def power(self):
        return self.flow * self.effort


class Mechanical1DBond(Bond):
    @property
    def force(self):
        return self.effort

    @force.setter
    def force(self, value: vip.TemporalVar):
        self.effort = value

    @property
    def speed(self):
        return self.flow

    @speed.setter
    def speed(self, value: vip.TemporalVar):
        self.flow = value


class Mechanical1DEffort(Mechanical1DBond):
    def __init__(self, effort=None):
        super().__init__()
        if effort is not None:
            self.effort = effort


class Mechanical1DFlow(Mechanical1DBond):
    def __init__(self, flow=None):
        super().__init__()
        if flow is not None:
            self.flow = flow
