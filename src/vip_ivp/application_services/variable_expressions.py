import inspect
from dataclasses import dataclass
from types import FrameType


class VariableExpression:
    def __init__(self, variable_id, expression: str):
        self.variable_id = variable_id
        self.creation_expression: str = expression

        frame = inspect.currentframe().f_back.f_back
        self.creation_frame_path: list[FrameType] = self.get_path(frame)

        self.name_frames = {}

    def get_name(self):
        frame = inspect.currentframe().f_back.f_back
        while frame is not None:
            for name, variable in frame.f_locals.items():
                if id(variable) == self.variable_id:
                    self.name_frames[name] = frame
                    break
            frame = frame.f_back

    @staticmethod
    def get_path(frame: FrameType) -> list[FrameType]:
        frame_path = []
        while frame is not None:
            frame_path.append(frame)
            frame = frame.f_back
        return frame_path

