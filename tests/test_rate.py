"""Tests for `rate` module."""

from __future__ import annotations

from time import sleep

import pytest

from affctrllib.rate import Rate

TEST_RATE_TOL = 3.0e-1


@pytest.fixture
def default_rate() -> Rate:
    return Rate(10)


class TestRate:
    def test_init(self, default_rate: Rate) -> None:
        rate = default_rate
        assert rate.frequency == 10
        assert rate.cycle_time() == 0

    @pytest.mark.parametrize("invalid_freq", [1e-12, 0, -10])
    def test_init_raise_error_when_given_freq_is_zero_or_negative(self, invalid_freq) -> None:
        with pytest.raises(ValueError):
            _ = Rate(invalid_freq)

    @pytest.mark.parametrize("duration,expected", [(1, 1), (0.1, 10), (0.01, 100)])
    def test_create_from_secs(self, duration: float, expected: float) -> None:
        rate = Rate.from_secs(duration)
        assert rate.frequency == expected

    @pytest.mark.parametrize("duration,expected", [(1, 1000), (10, 100), (100, 10)])
    def test_create_from_msecs(self, duration: int, expected: float) -> None:
        rate = Rate.from_msecs(duration)
        assert rate.frequency == expected

    @pytest.mark.parametrize("freq,expected", [(1, 1), (10, 0.1), (25, 0.04)])
    def test_expected_cycle_time(self, freq: float, expected: float) -> None:
        rate = Rate(freq)
        assert rate.expected_cycle_time() == expected

    def test_now(self, default_rate: Rate) -> None:
        rate = default_rate
        t1 = rate.now()
        sleep(0.001)
        t2 = rate.now()
        assert t1 < t2

    def test_start(self, default_rate: Rate) -> None:
        rate = default_rate
        sleep(0.001)
        t = rate.now()
        assert t > 0.001
        # Restart the timer
        rate.start()
        sleep(0.001)
        t = rate.now()
        assert t == pytest.approx(0.001, rel=TEST_RATE_TOL)

    def test_reset(self) -> None:
        rate = Rate(100)
        sleep(0.02)  # took longer than expected
        rate.reset()  # reset the timer
        sleep(0.01)  # took within expected time
        rate.sleep()  # calculate actual cycle time
        cycle_time = rate.cycle_time()
        t = rate.now()
        assert cycle_time == pytest.approx(0.01, rel=TEST_RATE_TOL)
        # rate.reset does not affect the elapsed time
        assert t == pytest.approx(0.03, rel=TEST_RATE_TOL)

    def test_sleep(self) -> None:
        rate = Rate(100)
        sleep(0.005)
        rate.sleep()
        t = rate.now()
        assert t == pytest.approx(0.01, rel=TEST_RATE_TOL)
        sleep(0.003)
        rate.sleep()
        t = rate.now()
        assert t == pytest.approx(0.02, rel=TEST_RATE_TOL)
