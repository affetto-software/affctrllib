"""This module handles the robot model.

It provides functionalities to handle the fundamental kinematic model of a robot.
"""

from __future__ import annotations

from enum import Enum, auto


class JointType(Enum):
    """Set of joint types."""

    FIXED = auto()
    REVOLUTE = auto()
    PRISMATIC = auto()

    @classmethod
    def from_str(cls, name: str) -> JointType:
        """Get joint type from its name.

        Parameters
        ----------
        name : str
            Joint type name.

        Returns
        -------
        JointType
            JointType instance.

        Raises
        ------
        ValueError
            If unrecognized joint type name is given.
        """

        for e in cls:
            if e.name.lower() == name.lower():
                return e
        raise ValueError(f"unrecognized joint type: {name}")


class Link(object):
    """Link model."""

    _name: str
    _jointtype: JointType
    _motion_range: tuple[float, float] | None
    _frame: list[list[float]] | None
    _parent: Link | None

    def __init__(
        self,
        name: str,
        jointtype: JointType | str,
        motion_range: tuple[float, float] | None = None,
        frame: list[list[float]] | None = None,
        parent: Link | None = None,
    ) -> None:
        self.name = name
        self.jointtype = jointtype
        self.motion_range = motion_range
        self.frame = frame
        self.parent = parent

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, newname: str) -> str:
        self._name = newname
        return self._name

    @property
    def jointtype(self) -> JointType:
        return self._jointtype

    @jointtype.setter
    def jointtype(self, jointtype: str | JointType) -> JointType:
        if isinstance(jointtype, str):
            self._jointtype = JointType.from_str(jointtype)
        else:
            self._jointtype = jointtype
        return self._jointtype

    @property
    def motion_range(self) -> tuple[float, float] | None:
        return self._motion_range

    @motion_range.setter
    def motion_range(self, motion_range: tuple[float, float] | None) -> tuple[float, float] | None:
        self._motion_range = motion_range
        return self._motion_range

    @property
    def frame(self) -> list[list[float]] | None:
        return self._frame

    @frame.setter
    def frame(self, frame: list[list[float]] | None) -> list[list[float]] | None:
        self._frame = frame
        return self._frame

    @property
    def parent(self) -> Link | None:
        return self._parent

    @parent.setter
    def parent(self, link: Link | None) -> Link | None:
        self._parent = link
        return self._parent
