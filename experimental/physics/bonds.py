from numbers import Number
from typing import Union, Type

import vip_ivp as vip


class Bond:
    def __init__(self):
        self.flow = None
        self.effort = None
        self.loop_node = None

    @property
    def effort(self):
        return self.effort

    @effort.setter
    def effort(self, value: vip.TemporalVar):
        self.effort = value

    @property
    def flow(self):
        return self.flow

    @flow.setter
    def flow(self, value: vip.TemporalVar):
        self.flow = value

    @property
    def power(self):
        return self.flow * self.effort


class BondFlow(Bond):
    def __init__(self, value: float, loop_node):
        super().__init__()
        self.flow = value
        self.effort = None
        self.loop_node = loop_node

    @classmethod
    def from_effort(cls, bond: "BondEffort"):
        if isinstance(bond, BondEffort):
            loop_node = vip.loop_node(bond.effort)
        elif isinstance(bond, Number):
            loop_node = vip.loop_node(bond)
        else:
            raise Exception(f"Incompatible type: {bond} of type {type(bond)}.")
        new_bond = cls(0, loop_node)
        return new_bond, loop_node

    @Bond.effort.setter
    def effort(self, value):
        self.effort = value
        self.loop_node.loop_into(self.effort)


class BondEffort(Bond):
    def __init__(self, value: float, loop_node):
        super().__init__()
        self.flow = None
        self.effort = value
        self.loop_node = loop_node

    @classmethod
    def from_flow(cls, bond: "BondFlow"):
        if isinstance(bond, BondFlow):
            loop_node = vip.loop_node(bond.flow)
        elif isinstance(bond, Number):
            loop_node = vip.loop_node(bond)
        else:
            raise Exception(f"Incompatible type: {bond} of type {type(bond)}.")
        new_bond = cls(0, loop_node)
        return new_bond, loop_node

    @Bond.flow.setter
    def flow(self, value):
        self.flow = value
        self.loop_node.loop_into(-self.flow)
