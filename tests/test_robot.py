"""Tests for `robot` module."""

from __future__ import annotations

import pytest

from affctrllib.robot import Chain, JointType, Link


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


CHAIN_EXAMPLE_01 = (
    # chain configuration
    {
        "name": "chain#01",
        "link": [
            {"name": "link#00", "jointtype": "fixed"},
            {"name": "link#01", "jointtype": "revolute", "parent": "link#00"},
            {"name": "link#02", "jointtype": "revolute", "parent": "link#01"},
            {"name": "link#03", "jointtype": "revolute", "parent": "link#02"},
        ],
    },
    # dof
    3,
)
CHAIN_EXAMPLE_02 = (
    # chain configuration
    {
        "name": "chain#02",
        "link": [
            {"name": "link#00", "jointtype": "fixed"},
            {"name": "link#01", "jointtype": "prismatic", "parent": "link#00"},
        ],
    },
    # dof
    1,
)
CHAIN_EXAMPLE_03 = (
    # chain configuration
    {
        "name": "chain#03",
        "link": [
            {"name": "link#00", "jointtype": "fixed"},
            {"name": "link#01", "jointtype": "revolute", "parent": "link#00"},
            {"name": "link#02", "jointtype": "revolute", "parent": "link#01"},
            {"name": "link#03", "jointtype": "revolute", "parent": "link#02"},
            {"name": "link#04", "jointtype": "revolute", "parent": "link#02"},
            {"name": "link#05", "jointtype": "revolute", "parent": "link#03"},
            {"name": "link#06", "jointtype": "revolute", "parent": "link#04"},
        ],
    },
    # dof
    6,
)


@pytest.fixture
def chain01() -> Chain:
    config = CHAIN_EXAMPLE_01[0]
    return Chain.from_config(config)


@pytest.fixture
def chain02() -> Chain:
    config = CHAIN_EXAMPLE_02[0]
    return Chain.from_config(config)


@pytest.fixture
def chain03() -> Chain:
    config = CHAIN_EXAMPLE_03[0]
    return Chain.from_config(config)


class TestChain:
    def test_init(self) -> None:
        chain = Chain([])
        assert chain.dof == 0
        assert len(chain) == 0
        assert chain.name == ""

    @pytest.mark.parametrize(
        "chain_config",
        [
            CHAIN_EXAMPLE_01,
            CHAIN_EXAMPLE_02,
            CHAIN_EXAMPLE_03,
        ],
    )
    def test_from_config(self, chain_config: tuple[dict, int]) -> None:
        chain = Chain.from_config(chain_config[0])
        assert chain.name == chain_config[0]["name"]
        assert chain.dof == chain_config[1]

    @pytest.mark.parametrize("name", ["robot01", "robot02", "robot03"])
    def test_name(self, chain01: Chain, name: str) -> None:
        chain01.name = name
        assert chain01.name == name

    def test_iterator(self, chain01: Chain) -> None:
        chain = chain01
        for link, expected in zip(chain, chain.links):
            assert link is expected

    @pytest.mark.parametrize(
        "chain_config,expected",
        [
            (CHAIN_EXAMPLE_01[0], ["link#00", "link#01", "link#02", "link#03"]),
            (CHAIN_EXAMPLE_02[0], ["link#00", "link#01"]),
            (CHAIN_EXAMPLE_03[0], ["link#00", "link#01", "link#02", "link#03", "link#04", "link#05", "link#06"]),
        ],
    )
    def test_get_link_names(self, chain_config: dict, expected: list[str]) -> None:
        chain = Chain.from_config(chain_config)
        assert chain.get_link_names() == expected

    @pytest.mark.parametrize("name", ["link#00", "link#01", "link#02"])
    def test_get_link(self, chain01: Chain, name: str) -> None:
        chain = chain01
        link = chain.get_link(name)
        assert link.name == name

    def test_get_link_error_when_not_found(self, chain01: Chain) -> None:
        chain = chain01
        with pytest.raises(KeyError) as err:
            _ = chain.get_link("invalid_link_name")
        assert "Link was not found in chain: invalid_link_name" in str(err.value)

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("link#00", ["link#01"]),
            ("link#01", ["link#02"]),
            ("link#02", ["link#03", "link#04"]),
            ("link#03", ["link#05"]),
            ("link#04", ["link#06"]),
        ],
    )
    def test_get_children(self, chain03: Chain, name: str, expected: list[str]) -> None:
        chain = chain03
        children = chain.get_children(name)
        assert len(children) == len(expected)
        for c, e in zip(children, expected):
            assert c.name == e

    def test_get_children_error_when_not_found(self, chain03: Chain) -> None:
        chain = chain03
        with pytest.raises(KeyError) as err:
            _ = chain.get_children("invalid_link_name")
        assert "Link was not found in chain: invalid_link_name" in str(err.value)
