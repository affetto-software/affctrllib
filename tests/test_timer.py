import time
from typing import Callable

import pytest
from affctrllib.timer import Timer

TOL = 5e-2


class TestTimer:
    def test_init(self) -> None:
        timer = Timer()
        assert timer._rate is None
        assert timer._period is None

    def test_error_when_rate_is_not_set(self) -> None:
        timer = Timer()
        with pytest.raises(ValueError) as excinfo:
            _ = timer.rate
        assert "rate is not set" in str(excinfo.value)

    @pytest.mark.parametrize("rate", [10, 30, 100])
    def test_set_rate(self, rate) -> None:
        timer = Timer()
        timer.set_rate(rate)
        assert timer.rate == rate
        assert timer.period == 1.0 / rate

    @pytest.mark.parametrize("rate", [20, 50, 100])
    def test_rate_setter(self, rate) -> None:
        timer = Timer()
        timer.rate = rate
        assert timer.rate == rate
        assert timer.period == 1.0 / rate

    @pytest.mark.parametrize("rate", [30, 100, 300])
    def test_set_rate_init(self, rate):
        timer = Timer(rate=rate)
        assert timer.rate == rate
        assert timer.period == 1.0 / rate

    def test_error_when_period_is_not_set(self) -> None:
        timer = Timer()
        with pytest.raises(ValueError) as excinfo:
            _ = timer.period
        assert "period is not set" in str(excinfo.value)

    @pytest.mark.parametrize("period", [0.1, 0.01, 0.025])
    def test_set_period(self, period) -> None:
        timer = Timer()
        timer.set_period(period)
        assert timer.period == period
        assert timer.rate == 1.0 / period

    @pytest.mark.parametrize("period", [0.2, 0.02, 0.005])
    def test_period_setter(self, period) -> None:
        timer = Timer()
        timer.period = period
        assert timer.period == period
        assert timer.rate == 1.0 / period

    @pytest.mark.parametrize("period", [0.3, 0.03, 0.003])
    def test_set_period_init(self, period):
        timer = Timer(period=period)
        assert timer.period == period
        assert timer.rate == 1.0 / period

    @pytest.mark.parametrize("period", [0.1, 0.01, 0.025])
    def test_period_ns(self, period) -> None:
        timer = Timer()
        timer.set_period(period)
        assert timer.period_ns == int(period * 1e9)

    @pytest.mark.parametrize("rate", [30, 60, 90])
    def test_period_ns_by_rate(self, rate) -> None:
        timer = Timer()
        timer.set_rate(rate)
        assert timer.period_ns == int(1e9 / rate)

    def test_elapsed_time(self) -> None:
        timer = Timer()
        timer.start()
        time.sleep(0.01)
        t = timer.elapsed_time()
        assert t == pytest.approx(0.01, rel=TOL)

    def test_elapsed_time2(self) -> None:
        timer = Timer()
        timer.start()
        time.sleep(0.01)
        t = timer.elapsed_time()
        assert t == pytest.approx(0.01, rel=TOL)
        time.sleep(0.01)
        t = timer.elapsed_time()
        assert t == pytest.approx(0.02, rel=TOL)

    def test_block(self) -> None:
        timer = Timer(period=0.01)
        timer.start()
        time.sleep(0.005)
        timer.block()
        t = timer.elapsed_time()
        assert t == pytest.approx(0.01, rel=TOL)

    def test_block2(self) -> None:
        timer = Timer(period=0.01)
        timer.start()
        time.sleep(0.005)
        timer.block()
        t = timer.elapsed_time()
        assert t == pytest.approx(0.01, rel=TOL)
        time.sleep(0.005)
        timer.block()
        t = timer.elapsed_time()
        assert t == pytest.approx(0.02, rel=TOL)

    def test_block_n_times(self) -> None:
        timer = Timer(period=0.01)
        timer.start()
        n = 10
        for i in range(1, n + 1):
            time.sleep(0.005)
            timer.block()
            t = timer.elapsed_time()
            assert t == pytest.approx(0.01 * i, rel=TOL)

    def test_error_block_without_start(self) -> None:
        timer = Timer(period=0.01)
        time.sleep(0.001)
        with pytest.raises(RuntimeError) as excinfo:
            timer.block()
        assert "Timer.start() must be called before Timer.block()" in str(excinfo.value)

    @pytest.mark.parametrize(
        "time_ns_func",
        [
            time.time_ns,
            time.monotonic_ns,
            time.perf_counter_ns,
            # time.process_time_ns, # these functions are useless in this case
            # time.thread_time_ns,
        ],
    )
    def test_alternative_time_ns_function(
        self, time_ns_func: Callable[[], int]
    ) -> None:
        timer = Timer(rate=100)
        timer._time_ns_func = time_ns_func
        timer.start()
        time.sleep(0.005)
        timer.block()
        t = timer.elapsed_time()
        # assert t == pytest.approx(0.01)  # uncomment this to see differences
        assert t == pytest.approx(0.01, rel=TOL)

    def test_warn_block_has_no_effect(self) -> None:
        timer = Timer(period=0.01)
        timer.start()
        with pytest.warns(RuntimeWarning) as record:
            for _ in range(3):
                time.sleep(0.01)
                timer.block()
        assert len(record) == 3
        for i in range(3):
            assert str(record[i].message).startswith(
                "It took longer than specified period at t="
            )
