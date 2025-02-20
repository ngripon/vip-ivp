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


def create_bond_types(name: str, effort_name: str, flow_name: str) -> tuple[Type["Bond"], Type["Bond"]]:
    """Dynamically creates BondEffort and BondFlow classes with renamed attributes."""

    class BondEffort(Bond):
        """Effort-driven bond."""

        def __init__(self, value: float, loop_node):
            super().__init__()
            setattr(self, flow_name, None)
            setattr(self, effort_name, value)
            self.loop_node = loop_node

        @classmethod
        def from_flow(cls, bond):
            if isinstance(bond, BondFlow):
                loop_node = vip.loop_node(getattr(bond, flow_name))
            elif isinstance(bond, Number):
                loop_node = vip.loop_node(bond)
            else:
                raise Exception(f"Incompatible type: {bond} of type {type(bond)}.")
            new_bond = cls(0, loop_node)
            return new_bond, loop_node

        @property
        def flow(self):
            return getattr(self, flow_name)

        @flow.setter
        def flow(self, value):
            setattr(self, flow_name, value)
            self.loop_node.loop_into(-value)

    class BondFlow(Bond):
        """Flow-driven bond."""

        def __init__(self, value: float, loop_node):
            super().__init__()
            setattr(self, flow_name, value)
            setattr(self, effort_name, None)
            self.loop_node = loop_node

        @classmethod
        def from_effort(cls, bond):
            if isinstance(bond, BondEffort):
                loop_node = vip.loop_node(getattr(bond, effort_name))
            elif isinstance(bond, Number):
                loop_node = vip.loop_node(bond)
            else:
                raise Exception(f"Incompatible type: {bond} of type {type(bond)}.")
            new_bond = cls(0, loop_node)
            return new_bond, loop_node

        @property
        def effort(self):
            return getattr(self, effort_name)

        @effort.setter
        def effort(self, value):
            setattr(self, effort_name, value)
            self.loop_node.loop_into(value)

    BondEffort.__name__ = f"{name}Effort"
    BondFlow.__name__ = f"{name}Flow"

    return BondEffort, BondFlow


Mechanical1DEffort, Mechanical1DFlow = create_bond_types("Mechanical1D", "force", "velocity")
