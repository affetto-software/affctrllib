"""Tests for `robot` module."""

from __future__ import annotations

import pytest

from affctrllib.robot import JointType, Link


@pytest.mark.parametrize(
    "jointtype,expected",
    [
        ("fixed", JointType.FIXED),
        ("revolute", JointType.REVOLUTE),
        ("prismatic", JointType.PRISMATIC),
    ],
)
def test_jointtype_from_str(jointtype: str, expected: JointType) -> None:
    assert JointType.from_str(jointtype) is expected


def test_jointtype_from_str_raise_error() -> None:
    with pytest.raises(ValueError):
        _ = JointType.from_str("hoge")


@pytest.mark.parametrize("jointtype", ["fixed", "revolute", "prismatic"])
def test_jointtype_convert_to_str(jointtype: str) -> None:
    j = JointType.from_str(jointtype)
    assert str(j) == jointtype


@pytest.fixture
def link() -> Link:
    return Link("link", "fixed")


class TestLink:

    def test_init(self) -> None:
        link = Link("link", "fixed")
        assert link.name == "link"
        assert link.jointtype is JointType.FIXED
        assert link.motion_range is None
        assert link.frame is None
        assert link.parent is None

    @pytest.mark.parametrize(
        "config",
        [
            {"name": "link0", "jointtype": "fixed"},
            {"name": "link1", "jointtype": "revolute", "range": [0, 90], "parent": "torso"},
            {"name": "link2", "jointtype": "prismatic", "range": [0, 0.01], "parent": "waist"},
        ],
    )
    def test_from_config(self, config: dict) -> None:
        link = Link.from_config(config)
        assert link.name == config["name"]
        assert str(link.jointtype) == config["jointtype"]
        if "range" in config:
            assert link.motion_range == tuple(config["range"])
        else:
            assert link.motion_range is None
        if "parent" in config:
            assert link.parent == config["parent"]
        else:
            assert link.parent is None

    @pytest.mark.parametrize("name", ["link0", "link1", "link2"])
    def test_name(self, link: Link, name: str) -> None:
        link.name = name
        assert link.name == name

    @pytest.mark.parametrize(
        "jointtype",
        [
            JointType.FIXED,
            JointType.REVOLUTE,
            JointType.PRISMATIC,
        ],
    )
    def test_jointtype(self, link: Link, jointtype: JointType) -> None:
        link.jointtype = jointtype
        assert link.jointtype is jointtype

    @pytest.mark.parametrize(
        "jointtype,expected",
        [
            ("fixed", JointType.FIXED),
            ("revolute", JointType.REVOLUTE),
            ("prismatic", JointType.PRISMATIC),
        ],
    )
    def test_jointtype_str(self, link: Link, jointtype: str, expected: JointType) -> None:
        link.jointtype = jointtype
        assert link.jointtype is expected

    @pytest.mark.parametrize("motion_range", [None, (0, 90), (-90, 90), (0.0, 180.0)])
    def test_motion_range(self, link: Link, motion_range: tuple[float, float] | None) -> None:
        link.motion_range = motion_range
        assert link.motion_range == motion_range

    @pytest.mark.parametrize(
        "frame",
        [
            None,
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ],
            [
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 1, 1],
            ],
        ],
    )
    def test_frame(self, link: Link, frame: list[list[float]] | None) -> None:
        link.frame = frame
        assert link.frame == frame
