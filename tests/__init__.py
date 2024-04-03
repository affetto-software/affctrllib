"""Tests suite for `affctrllib`"""

from __future__ import annotations

from pathlib import Path

TESTS_DIR = Path(__file__).parent
TESTS_DATA_DIR = TESTS_DIR / "data"
CONFIG_DIR = TESTS_DATA_DIR / "config"

SAMPLE_CONFIG_FILE = CONFIG_DIR / "sample_config.toml"
ALTERNATIVE_CONFIG_FILE = CONFIG_DIR / "alternative_config.toml"

AFFETTO_CONFIG_FILE = CONFIG_DIR / "affetto.toml"
