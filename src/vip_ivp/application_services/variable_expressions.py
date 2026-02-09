import inspect
from pathlib import Path
from types import FrameType

from src.vip_ivp.config import PACKAGE_ROOT


class VariableExpression:
    def __init__(self, variable_id, expression: str):
        self.variable_id = variable_id
        self.creation_expression: str = expression

        frame = inspect.currentframe().f_back.f_back
        self.creation_frame_path: list[FrameType] = get_path(frame)

        self.name_frames = {}

    def get_name(self):
        frame = inspect.currentframe().f_back.f_back
        while frame is not None:
            if not is_inside_package(frame):
                for name, variable in frame.f_locals.items():
                    if id(variable) == self.variable_id:
                        self.name_frames[name] = frame
                        break
            frame = frame.f_back


def is_inside_package(frame: FrameType) -> bool:
    filename = frame.f_code.co_filename
    if not filename:
        return False
    return Path(frame.f_code.co_filename).is_relative_to(PACKAGE_ROOT)


def get_path(frame: FrameType) -> list[FrameType]:
    frame_path = []
    while frame is not None:
        if not is_inside_package(frame):
            frame_path.append(frame)
        frame = frame.f_back
    return frame_path
