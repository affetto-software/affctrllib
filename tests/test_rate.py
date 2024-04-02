"""Tests for `rate` module."""

from __future__ import annotations

import pytest

from affctrllib.rate import Rate


class TestRate:
    def test_init(self) -> None:
        rate = Rate(10)
        assert rate.frequency == 10

    @pytest.mark.parametrize("invalid_freq", [1e-12, 0, -10])
    def test_init_raise_error_when_given_freq_is_zero_or_negative(self, invalid_freq) -> None:
        with pytest.raises(ValueError):
            _ = Rate(invalid_freq)
