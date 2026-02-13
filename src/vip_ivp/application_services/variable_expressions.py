"""
The goal of variable expression is to be able to reconstruct the way the system is built in the code.

The systems are flat: if a variable is created in a function, it has no way to know that. Expression data aim to fill
this knowledge gap.

Variable expression stores 2 information:
    - Declaration data : in which function frame the variable has been created, and its expression in it.
    - Name data : which name has the variable in each frame.

The ultimate goal is to reconstruct the system graph. To build the system graph:
    1. Named variables must be stored.
    2. Pick a variable and explore its sources:
        a. In the creation frame, draw a block containing the expression.
        b. The block inputs are the first named variables attained while searching the sources.
        c. If there is a frame change on the chain:
            - If the frame is deeper, on the current level the variable is coming out of the function of the frame. In
            the frame, the current variable is an output.
            - If the frame is shallower, on the current level the variable is coming from an input. In the frame, the
            current variable goes into an input of the function that was on the current level.
"""

import inspect
from pathlib import Path
from types import FrameType

from ..config import PACKAGE_ROOT


class VariableExpression:
    def __init__(self, variable_id, expression: str):
        self.variable_id = variable_id

        frame = inspect.currentframe().f_back.f_back
        self.creation_expression: str = expression
        self.creation_frame = get_first_frame_outside_package(frame)

        self.name_frames: dict[FrameType, str] = {}

    def set_name(self):
        frame = inspect.currentframe().f_back.f_back
        while frame is not None:
            if not _is_inside_package(frame):
                for name, variable in frame.f_locals.items():
                    if id(variable) == self.variable_id:
                        self.name_frames[frame] = name
                        break
            frame = frame.f_back

    def get_name(self):
        frame = get_first_frame_outside_package(inspect.currentframe().f_back)
        if frame in self.name_frames:
            return self.name_frames[frame]
        elif frame is self.creation_frame:
            return self.creation_expression
        else:
            # In this case, the variable has been created in a deeper frame, so we get the function name
            current = self.creation_frame
            while True:
                parent_fun = current.f_back
                if parent_fun is None:
                    return "NOT_FOUND"
                if parent_fun is frame:
                    return f"{current.f_code.co_name}()"
                current = current.f_back


def _is_inside_package(frame: FrameType) -> bool:
    filename = frame.f_code.co_filename
    if not filename:
        return False
    return Path(frame.f_code.co_filename).is_relative_to(PACKAGE_ROOT)


def get_first_frame_outside_package(frame: FrameType) -> FrameType | None:
    while frame is not None:
        if not _is_inside_package(frame):
            return frame
        frame = frame.f_back
    return None
