import pytest
from affctrllib._periodic_runner import PeriodicRunner


class TestPeriodicRunner:
    def test_init(self) -> None:
        runner = PeriodicRunner()
        assert not hasattr(runner, "dt")
        assert not hasattr(runner, "freq")
        assert runner.n_steps == 0

    @pytest.mark.parametrize("dt,freq", [(0.01, 100), (0.001, 1000), (0.02, 50)])
    def test_init_specify_dt(self, dt, freq) -> None:
        runner = PeriodicRunner(dt=dt)
        assert runner.dt == dt
        assert runner.freq == freq

    @pytest.mark.parametrize("freq,dt", [(100, 0.01), (1000, 0.001), (30, 1.0 / 30)])
    def test_init_specify_freq(self, freq, dt) -> None:
        runner = PeriodicRunner(freq=freq)
        assert runner.dt == dt
        assert runner.freq == freq

    def test_init_error_both_of_dt_freq_specified(self) -> None:
        dt = 0.01
        freq = 100
        with pytest.raises(ValueError) as excinfo:
            _ = PeriodicRunner(dt=dt, freq=freq)
        assert "Unable to specify DT and FREQ simultaneously" in str(excinfo.value)

    @pytest.mark.parametrize("dt,freq", [(0.01, 100), (0.001, 1000), (0.02, 50)])
    def test_dt_setter(self, dt, freq) -> None:
        runner = PeriodicRunner(dt=0.01)
        runner.dt = dt
        assert runner.dt == dt
        assert runner.freq == freq

    @pytest.mark.parametrize("freq,dt", [(100, 0.01), (1000, 0.001), (30, 1.0 / 30)])
    def test_freq_setter(self, dt, freq) -> None:
        runner = PeriodicRunner(dt=0.01)
        runner.freq = freq
        assert runner.dt == dt
        assert runner.freq == freq

    def test_update(self) -> None:
        runner = PeriodicRunner()
        assert runner.n_steps == 0
        runner.update()
        assert runner.n_steps == 1
        runner.update()
        assert runner.n_steps == 2
        runner.update()
        runner.update()
        runner.update()
        assert runner.n_steps == 5
