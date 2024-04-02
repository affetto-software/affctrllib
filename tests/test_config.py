"""Tests for `config` module."""

from __future__ import annotations

from typing import Any

import pytest

from affctrllib.config import ConfigBase
from tests import ALTERNATIVE_CONFIG_FILE, SAMPLE_CONFIG_FILE


class SampleConfig(ConfigBase):
    @staticmethod
    def load_from_mapping(mapping: dict[str, Any], ignore_load_error: bool = False) -> SampleConfig:
        c = SampleConfig()
        c.set_mapping(mapping)
        _ = ignore_load_error
        c.load()
        return c

    def load(self) -> bool:
        return True


@pytest.fixture
def sample_config() -> SampleConfig:
    config = SampleConfig(SAMPLE_CONFIG_FILE)
    return config


@pytest.fixture
def alternative_config() -> SampleConfig:
    config = SampleConfig(ALTERNATIVE_CONFIG_FILE)
    return config


class TestConfig:
    def test_init(self):
        config = SampleConfig()
        assert hasattr(config, "_path") is False
        assert hasattr(config, "_mapping") is False
        assert config.is_loaded_from_file() is False

    def test_path_raise_exception(self):
        config = SampleConfig()
        with pytest.raises(AttributeError) as excinfo:
            _ = config.path
            assert "'SampleConfig' object is not loaded from configuration file.\n" in str(excinfo.value)

    def test_mapping_raise_exception(self):
        config = SampleConfig()
        with pytest.raises(AttributeError) as excinfo:
            _ = config.mapping
            assert "'Configuration' object has not initialized.\n" in str(excinfo.value)

    def test_init_with_file(self, sample_config):
        assert sample_config.path == SAMPLE_CONFIG_FILE

    def test_load_mapping(self):
        c = SampleConfig()
        c.load_mapping(SAMPLE_CONFIG_FILE)
        assert c.path == SAMPLE_CONFIG_FILE
        assert c.mapping["robot"]["name"] == "sample"
        assert c.is_loaded_from_file() is True

    def test_load_mapping_alternative(self):
        c = SampleConfig()
        c.load_mapping(SAMPLE_CONFIG_FILE)
        assert c.path == SAMPLE_CONFIG_FILE
        assert c.mapping["robot"]["name"] == "sample"

        c.load_mapping(ALTERNATIVE_CONFIG_FILE)
        assert c.path == ALTERNATIVE_CONFIG_FILE
        assert c.mapping["robot"]["name"] == "alternative"

    def test_set_mapping(self):
        c = SampleConfig()
        c.set_mapping({"robot": {"name": "sample"}})
        with pytest.raises(AttributeError):
            _ = c.path
        assert c.mapping["robot"]["name"] == "sample"
        assert c.is_loaded_from_file() is False

    def test_set_mapping_after_loaded(self):
        c = SampleConfig()
        c.load_mapping(SAMPLE_CONFIG_FILE)
        assert c.is_loaded_from_file() is True
        # c.set_mapping removes the path attribute.
        c.set_mapping({"robot": {"name": "alternative"}})
        assert c.mapping["robot"]["name"] == "alternative"
        assert c.is_loaded_from_file() is False

    def test_set_mapping_after_loaded_not_delete_path(self):
        c = SampleConfig()
        c.load_mapping(SAMPLE_CONFIG_FILE)
        assert c.is_loaded_from_file() is True
        # if the `delete_path` is False, c.set_mapping will not remove
        # the path attribute.
        c.set_mapping({"robot": {"name": "alternative"}}, delete_path=False)
        assert c.mapping["robot"]["name"] == "alternative"
        assert c.is_loaded_from_file() is True
        assert c.path == SAMPLE_CONFIG_FILE

    def test_get_value(self, sample_config: SampleConfig):
        c = sample_config
        assert c.get_value("robot", "name") == "sample"

    def test_get_value_with_default(self, sample_config: SampleConfig):
        c = sample_config
        assert c.get_value("robot", "chain", "dof", default=13) == 13

    def test_get_value_with_no_args(self, sample_config: SampleConfig):
        c = sample_config
        m = c.get_value()
        assert isinstance(m, dict)
        assert m["robot"]["name"] == "sample"

    def test_get_value_raise_error_when_invalid_keys_given(self, sample_config: SampleConfig):
        c = sample_config
        with pytest.raises(KeyError):
            _ = c.get_value("robot", "invalidkey")

    def test_get_value_raise_error_when_invalid_keys_and_default_given(self, sample_config: SampleConfig):
        c = sample_config
        with pytest.raises(KeyError):
            _ = c.get_value("robot", "invalidkey", "dof", default=13)

    def test_get_value_with_default_special_value(self, sample_config: SampleConfig):
        """Test if literal '-1' is not treated as CONFIG_GET_VALUE_NO_DEFAULT."""
        c = sample_config
        assert c.get_value("robot", "invalidkey", default=-1) == -1
