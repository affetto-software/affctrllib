import os

import pytest
from affctrllib.affstate import AffState, AffStateThread
from numpy.testing import assert_array_equal

CONFIG_DIR_PATH = os.path.join(os.path.dirname(__file__), "config")


@pytest.mark.filterwarnings("ignore:Sensor frequency is not provided")
class TestAffState:
    def test_init(self) -> None:
        state = AffState()
        assert state.dt == 0.01
        assert state.freq == 100
        assert state.n_steps == 0

    @pytest.mark.parametrize("dt,freq", [(0.01, 100), (0.001, 1000), (0.02, 50)])
    def test_init_specify_dt(self, dt, freq) -> None:
        state = AffState(dt=dt)
        assert state.dt == dt
        assert state.freq == freq

    @pytest.mark.parametrize("freq,dt", [(100, 0.01), (1000, 0.001), (30, 1.0 / 30)])
    def test_init_specify_freq(self, freq, dt) -> None:
        state = AffState(freq=freq)
        assert state.dt == dt
        assert state.freq == freq

    def test_init_error_both_of_dt_freq_specified(self) -> None:
        dt = 0.01
        freq = 100
        with pytest.raises(ValueError) as excinfo:
            _ = AffState(dt=dt, freq=freq)
        assert "Unable to specify DT and FREQ simultaneously" in str(excinfo.value)

    @pytest.mark.parametrize("dt,freq", [(0.01, 100), (0.001, 1000), (0.02, 50)])
    def test_dt_setter(self, dt, freq):
        state = AffState(dt=0.01)
        state.dt = dt
        assert state.dt == dt
        assert state.freq == freq

    @pytest.mark.parametrize("freq,dt", [(100, 0.01), (1000, 0.001), (30, 1.0 / 30)])
    def test_freq_setter(self, dt, freq):
        state = AffState(dt=0.01)
        state.freq = freq
        assert state.dt == dt
        assert state.freq == freq

    def test_init_config(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "default.toml")
        state = AffState(config)
        assert str(state.config_path) == config
        assert state.dof == 13
        assert state.freq == 30

    def test_init_config_alternative(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "alternative.toml")
        state = AffState(config)
        assert str(state.config_path) == config
        assert state.dof == 14
        assert state.freq == 100

    def test_init_config_empty(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "empty.toml")
        with pytest.warns() as record:
            state = AffState(config)
        assert len(record) == 2
        assert str(record[0].message) == "'chain' field is not defined"
        assert (
            str(record[1].message)
            == "Sensor frequency is not provided, set to default: 100"
        )
        assert state.freq == 100

    def test_update_raw_data(self) -> None:
        state = AffState(dt=0.01)
        data = list(range(15))
        state.update(data)
        assert state.raw_data == list(range(15))
        assert state.n_steps == 1

    def test_update_split_data_no_filtering(self) -> None:
        state = AffState(dt=0.01)
        data = list(range(15))
        state._filter_list = [None, None, None]
        state.update(data)
        assert_array_equal(state.q, [0, 3, 6, 9, 12])
        assert_array_equal(state.pa, [1, 4, 7, 10, 13])
        assert_array_equal(state.pb, [2, 5, 8, 11, 14])

    def test_update_split_data_with_filter(self) -> None:
        state = AffState(dt=0.01)
        # update 1
        data = [0] * 6
        expected = [0, 0]
        state.update(data)
        assert_array_equal(state.q, expected)
        assert_array_equal(state.pa, expected)
        assert_array_equal(state.pb, expected)
        # update 2
        data = [1] * 6
        expected = [0.2, 0.2]
        state.update(data)
        assert_array_equal(state.q, expected)
        assert_array_equal(state.pa, expected)
        assert_array_equal(state.pb, expected)
        # update 3
        data = [2] * 6
        expected = [0.6, 0.6]
        state.update(data)
        assert_array_equal(state.q, expected)
        assert_array_equal(state.pa, expected)
        assert_array_equal(state.pb, expected)
        # update 4
        data = [3] * 6
        expected = [1.2, 1.2]
        state.update(data)
        assert_array_equal(state.q, expected)
        assert_array_equal(state.pa, expected)
        assert_array_equal(state.pb, expected)
        # update 5
        data = [4] * 6
        expected = [2, 2]
        state.update(data)
        assert_array_equal(state.q, expected)
        assert_array_equal(state.pa, expected)
        assert_array_equal(state.pb, expected)
        assert state.n_steps == 5

    def test_update_calc_dq(self) -> None:
        state = AffState(dt=0.01)
        state._filter_list = [None, None, None]
        # update 1
        data = [0, 1] * 3
        expected = [0, 0]
        state.update(data)
        assert_array_equal(state.dq, expected)
        # update 2
        data = [1, 3] * 3
        expected = [100, 200]
        state.update(data)
        assert_array_equal(state.dq, expected)
        # update 3
        data = [2, 5] * 3
        expected = [100, 200]
        state.update(data)
        assert_array_equal(state.dq, expected)
        assert state.n_steps == 3


@pytest.mark.filterwarnings("ignore:Sensor frequency is not provided")
class TestAffStateThread:
    def test_init_config(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "default.toml")
        state = AffStateThread(config)
        assert state.freq == 30

    def test_init_alternative_freq(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "default.toml")
        state = AffStateThread(config, freq=50)
        assert state.freq == 50

    def test_set_freq(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "default.toml")
        state = AffStateThread(config)
        assert state.freq == 30
        state.freq = 50
        assert state.freq == 50
