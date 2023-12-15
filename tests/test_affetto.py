import os
import sys

import pytest

from affctrllib.affetto import Affetto, Chain

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib


CONFIG_DIR_PATH = os.path.join(os.path.dirname(__file__), "config")

CHAIN_LINK_EXAMPLE_01 = (
    # link configuration
    [
        {"name": "link#00", "jointtype": "fixed"},
        {"name": "link#01", "jointtype": "revolute", "parent": "link#00"},
        {"name": "link#02", "jointtype": "revolute", "parent": "link#01"},
    ],
    # dof
    2,
)
CHAIN_LINK_EXAMPLE_02 = (
    # link configuration
    [
        {"name": "link#00", "jointtype": "fixed"},
        {"name": "link#01", "jointtype": "prismatic", "parent": "link#00"},
    ],
    # dof
    1,
)
CHAIN_LINK_EXAMPLE_03 = (
    # link configuration
    [
        {"name": "link#00", "jointtype": "fixed"},
        {"name": "link#01", "jointtype": "revolute", "parent": "link#00"},
        {"name": "link#02", "jointtype": "revolute", "parent": "link#01"},
        {"name": "link#03", "jointtype": "revolute", "parent": "link#02"},
        {"name": "link#04", "jointtype": "revolute", "parent": "link#03"},
        {"name": "link#05", "jointtype": "revolute", "parent": "link#04"},
        {"name": "link#06", "jointtype": "revolute", "parent": "link#05"},
    ],
    # dof
    6,
)


class TestChain:
    def test_init(self) -> None:
        chain = Chain()
        assert isinstance(chain, Chain)

    @pytest.mark.parametrize(
        "linkchain",
        [
            CHAIN_LINK_EXAMPLE_01,
            CHAIN_LINK_EXAMPLE_02,
            CHAIN_LINK_EXAMPLE_03,
        ],
    )
    def test_init_with_config(self, linkchain) -> None:
        c = {"link": linkchain[0]}
        chain = Chain(c)
        assert chain.dof == linkchain[1]

    @pytest.mark.parametrize(
        "linkchain",
        [
            CHAIN_LINK_EXAMPLE_01,
            CHAIN_LINK_EXAMPLE_02,
            CHAIN_LINK_EXAMPLE_03,
        ],
    )
    def test_load(self, linkchain) -> None:
        c = {"link": linkchain[0]}
        chain = Chain()
        chain.load(c)
        assert chain.dof == linkchain[1]

    @pytest.mark.parametrize("cfile,dof", [("chain1.toml", 2), ("chain2.toml", 6)])
    def test_load_from_file(self, cfile, dof) -> None:
        cpath = os.path.join(CONFIG_DIR_PATH, cfile)
        with open(cpath, "rb") as f:
            config = tomllib.load(f)
        chain = Chain(config["chain"])
        assert chain.dof == dof


class TestAffetto:
    def test_init(self) -> None:
        affetto = Affetto()
        assert isinstance(affetto, Affetto)

    def test_init_with_config_path(self) -> None:
        cpath = os.path.join(CONFIG_DIR_PATH, "default.toml")
        affetto = Affetto(cpath)
        assert str(affetto.config_path) == cpath
        assert affetto.name == "affetto"
        assert affetto.dof == 13

    def test_load_config(self) -> None:
        cpath = os.path.join(CONFIG_DIR_PATH, "default.toml")
        affetto = Affetto()
        affetto.load_config_path(cpath)
        assert str(affetto.config_path) == cpath
        assert affetto.name == "affetto"
        assert affetto.dof == 13

    def test_set_config_error(self) -> None:
        affetto = Affetto()
        with pytest.raises(KeyError) as excinfo:
            affetto.load_config({})
        assert "'affetto'" == str(excinfo.value)

    def test_set_config_warning(self) -> None:
        affetto = Affetto()
        with pytest.warns(UserWarning) as record:
            affetto.load_config({"affetto": {}})
        assert "'chain' field is not defined" == str(record[0].message)

    @pytest.mark.parametrize(
        "linkchain",
        [
            CHAIN_LINK_EXAMPLE_01,
            CHAIN_LINK_EXAMPLE_02,
            CHAIN_LINK_EXAMPLE_03,
        ],
    )
    def test_load_chain(self, linkchain) -> None:
        c = {"chain": {"link": linkchain[0]}}
        affetto = Affetto()
        affetto.load_chain(c)
        assert affetto.dof == linkchain[1]

    def test_dof_default(self) -> None:
        affetto = Affetto()
        assert affetto.dof == 13
