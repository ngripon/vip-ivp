import ast
from dataclasses import dataclass, field
from typing import List

import vip_ivp as vip


def get_ast(filename: str):
    with open(filename, "r") as f:
        src_file = f.read()

    tree = ast.parse(src_file)
    return tree


@dataclass
class Node:
    name: str
    var: vip.TemporalVar
    inputs: List["Node"] = field(default_factory=list)
    outputs: List["Node"] = field(default_factory=list)


if __name__ == '__main__':
    node_dict = {}
    tree = get_ast("../demos/exponential_decay.py")
    # print(tree)
    # for node in ast.walk(tree):
    #     if isinstance(node, ast.Assign):
    #         if node.value.func.attr == "loop_node":
    #             new_node = Node(node.targets[0].id)
    #
    #     print(node)
