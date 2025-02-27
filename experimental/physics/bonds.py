import enum
from typing import List

import vip_ivp as vip


class Intensive(enum.Enum):
    EFFORT = 0
    FLOW = 1


class Bond:
    def __init__(self, intensive_variable: Intensive = Intensive.FLOW):
        self._flow = vip.loop_node()
        self._effort = vip.loop_node()
        self.intensive_variable = intensive_variable
        self._loop_nodes_to_update: List[vip.LoopNode] = []

    @property
    def effort(self) -> vip.LoopNode:
        return self._effort

    @effort.setter
    def effort(self, value: vip.TemporalVar):
        self._effort.loop_into(value)

    @property
    def flow(self) -> vip.LoopNode:
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
        self._loop_nodes_to_update=[self.flow]

    @classmethod
    def from_bond(cls, bond: 'Mechanical1DEffort'):
        new_bond = cls(bond.effort)
        new_bond._loop_nodes_to_update.extend(bond._loop_nodes_to_update)
        return new_bond

    @property
    def flow(self):
        return super().flow

    @flow.setter
    def flow(self, value: vip.TemporalVar):
        for node in self._loop_nodes_to_update:
            node.loop_into(value, force=False)
        # self._flow.loop_into(value)

    def __add__(self, other: "Mechanical1DEffort") -> "Mechanical1DEffort":
        new_bond = Mechanical1DEffort(other.effort + self.effort)
        new_bond._loop_nodes_to_update.extend((other.flow, self.flow))
        return new_bond


class Mechanical1DFlow(Mechanical1DBond):
    def __init__(self, flow=None):
        super().__init__()
        if flow is not None:
            self.flow = flow
        self._loop_nodes_to_update=[self.effort]

    @classmethod
    def from_bond(cls, bond: 'Mechanical1DFlow'):
        new_bond = cls(bond.flow)
        new_bond._loop_nodes_to_update.extend(bond._loop_nodes_to_update)
        return new_bond

    @property
    def effort(self):
        return super().effort

    @effort.setter
    def effort(self, value: vip.TemporalVar):
        for node in self._loop_nodes_to_update:
            node.loop_into(value, force=True)
        # self._effort.loop_into(value)
