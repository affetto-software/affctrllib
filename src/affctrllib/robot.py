"""This module handles the robot model.

It provides functionalities to handle the fundamental kinematic model of a robot.
"""

from __future__ import annotations

from enum import Enum, auto
from functools import cache, cached_property
from traceback import format_tb
from typing import Any, Iterator


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

    def __str__(self) -> str:
        return self.name.lower()


class Link(object):
    """Link model."""

    _name: str
    _jointtype: JointType
    _motion_range: tuple[float, float] | None
    _frame: list[list[float]] | None
    _parent: str | None

    def __init__(
        self,
        name: str,
        jointtype: JointType | str,
        motion_range: tuple[float, float] | None = None,
        frame: list[list[float]] | None = None,
        parent: str | None = None,
    ) -> None:
        """Initialize the Link class.

        Parameters
        ----------
        name : str
            Name of the link. Required.
        jointtype : JointType | str
            Joint type of the link. Required.
        motion_range : tuple[float, float], optional
            Motion range of the link. Optional.
        frame : list[list[float]], optional
            Adjacent transformation matrix. Optional.
        parent : str, optional
            Parent link name. Optional.
        """

        self.name = name
        self.jointtype = jointtype
        self.motion_range = motion_range
        self.frame = frame
        self.parent = parent

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Link:
        """Create a Link object from a configuration.

        Parameters
        ----------
        config : dict[str, Any]
            Dictionary that contains information to create Link
        object.

        Returns
        -------
        Link
            Initialized Link object.
        """

        name = config["name"]
        jointtype = JointType.from_str(config["jointtype"])
        motion_range = config.get("range", None)
        if motion_range is not None:
            motion_range = tuple(motion_range)
        frame = config.get("frame", None)
        parent = config.get("parent", None)
        return cls(name, jointtype, motion_range, frame, parent)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, newname: str) -> None:
        self._name = newname

    @property
    def jointtype(self) -> JointType:
        return self._jointtype

    @jointtype.setter
    def jointtype(self, jointtype: str | JointType) -> None:
        if isinstance(jointtype, str):
            self._jointtype = JointType.from_str(jointtype)
        else:
            self._jointtype = jointtype

    @property
    def motion_range(self) -> tuple[float, float] | None:
        return self._motion_range

    @motion_range.setter
    def motion_range(self, motion_range: tuple[float, float] | None) -> None:
        self._motion_range = motion_range

    @property
    def frame(self) -> list[list[float]] | None:
        return self._frame

    @frame.setter
    def frame(self, frame: list[list[float]] | None) -> None:
        self._frame = frame

    @property
    def parent(self) -> str | None:
        return self._parent

    @parent.setter
    def parent(self, link: str | None) -> None:
        self._parent = link


class Chain(object):
    """Kinematic chain of a robot."""

    _name: str
    _links: list[Link]
    _link_table: dict[str, Link]
    _chain: dict[str, list[Link]]

    def __init__(self, links: list[Link], name: str = "") -> None:
        """Initialize the Chain class.

        Parameters
        ----------
        links : list[Link]
            A list of link objects to construct a kinematic chain.
        name : str
            Name of the chain.
        """

        self.name = name
        self._links = links
        self._construct_link_table()
        self._construct_chain()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Chain:
        """Create a Chain object from a configuration.

        Parameters
        ----------
        config : dict[str, Any]
            A dictionary that contains a configuration of the chain.

        Returns
        -------
        Chain
            Initialized Chain object.
        """

        name = config.get("name", "")
        link_config = config.get("link", [])
        links: list[Link] = []
        for c in link_config:
            links.append(Link.from_config(c))
        return Chain(links, name)

    def __len__(self) -> int:
        """Return the number of links the chain has.

        Returns
        -------
        int
            The number of links.
        """

        return len(self._links)

    def __iter__(self) -> Iterator[Link]:
        """Return an iterator that iterates links in the chain.

        Returns
        -------
        Iterator[Link]
            An iterator of links.
        """

        return iter(self._links)

    @cached_property
    def dof(self) -> int:
        """Return the total degree of freedom of the chain.

        Returns
        -------
        int
            The degree of freedom.
        """

        dof = 0
        for link in self._links:
            if link.jointtype in (JointType.REVOLUTE, JointType.PRISMATIC):
                dof += 1
        return dof

    @property
    def name(self) -> str:
        """Return the name of the chain.

        Returns
        -------
        str
            Name of the chain.
        """

        return self._name

    @name.setter
    def name(self, newname: str) -> None:
        """Set a new name for the chain.

        Parameters
        ----------
        newname : str
            A string to name the chain.
        """

        self._name = newname

    @property
    def links(self) -> list[Link]:
        """Return a list of links in the chain.

        Returns
        -------
        list[Link]
            List of links in the chain.
        """

        return self._links

    @cache
    def get_link_names(self) -> list[str]:
        """Return a list of names of links.

        Returns
        -------
        list[str]
            Return a list of strings
        """

        return [link.name for link in self._links]

    def _construct_link_table(self) -> None:
        # Generate a link table that maps a link name to the
        # corresponding link object. It is utilized to make finding a
        # link object by its name faster.
        self._link_table = {}
        for link in self._links:
            self._link_table[link.name] = link

    def _construct_chain(self) -> None:
        # Generate a kinematic chain structure from provided link
        # objects. It maps a parent link name to a list of link
        # objects that connect to the parent link.
        self._chain = {}
        for link in self._links:
            if link.parent:
                children = self._chain.get(link.parent, [])
                children.append(link)
                self._chain[link.parent] = children

    def get_link(self, link_name: str) -> Link:
        """Get a link object by its name.

        Parameters
        ----------
        link_name : str
            Name of the link.

        Returns
        -------
        Link
            Link object.

        Raises
        ------
        KeyError
            If the given name is not found in the chain.
        """

        try:
            return self._link_table[link_name]
        except KeyError as err:
            msg = f"Link was not found in chain: {link_name}\n" + format_tb(err.__traceback__)[0] + err.args[0]
            raise KeyError(msg) from None

    def get_children(self, link_name: str) -> list[Link]:
        """Get a list of links from their parent name.

        Parameters
        ----------
        link_name : str
            Name of a parent link name.

        Returns
        -------
        list[Link]
            List of child links of a parent link.

        Raises
        ------
        KeyError
            If the give name is not found in the chain.
        """

        try:
            return self._chain[link_name]
        except KeyError as err:
            msg = f"Link was not found in chain: {link_name}\n" + format_tb(err.__traceback__)[0] + err.args[0]
            raise KeyError(msg) from None
