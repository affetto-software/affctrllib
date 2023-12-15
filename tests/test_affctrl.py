import os

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from affctrllib.affctrl import AffCtrl, AffCtrlThread

CONFIG_DIR_PATH = os.path.join(os.path.dirname(__file__), "config")


@pytest.mark.filterwarnings("ignore:Control frequency is not provided")
class TestAffCtrl:
    def test_init(self) -> None:
        ctrl = AffCtrl()
        assert isinstance(ctrl, AffCtrl)
        assert ctrl.freq == 30
        assert ctrl.dt == 1.0 / 30
        assert ctrl.n_steps == 0
        assert ctrl.inactive_joints.shape == (0, 3)

    @pytest.mark.parametrize("dt,freq", [(0.01, 100), (0.001, 1000), (0.02, 50)])
    def test_init_specify_dt(self, dt, freq) -> None:
        ctrl = AffCtrl(dt=dt)
        assert ctrl.dt == dt
        assert ctrl.freq == freq

    @pytest.mark.parametrize("freq,dt", [(100, 0.01), (1000, 0.001), (30, 1.0 / 30)])
    def test_init_specify_freq(self, freq, dt) -> None:
        ctrl = AffCtrl(freq=freq)
        assert ctrl.dt == dt
        assert ctrl.freq == freq

    def test_init_error_both_of_dt_freq_specified(self) -> None:
        dt = 0.01
        freq = 100
        with pytest.raises(ValueError) as excinfo:
            _ = AffCtrl(dt=dt, freq=freq)
        assert "Unable to specify DT and FREQ simultaneously" in str(excinfo.value)

    @pytest.mark.parametrize("dt,freq", [(0.01, 100), (0.001, 1000), (0.02, 50)])
    def test_dt_setter(self, dt, freq):
        ctrl = AffCtrl(dt=0.01)
        ctrl.dt = dt
        assert ctrl.dt == dt
        assert ctrl.freq == freq

    @pytest.mark.parametrize("freq,dt", [(100, 0.01), (1000, 0.001), (30, 1.0 / 30)])
    def test_freq_setter(self, dt, freq):
        ctrl = AffCtrl(dt=0.01)
        ctrl.freq = freq
        assert ctrl.dt == dt
        assert ctrl.freq == freq

    def test_init_config(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "default.toml")
        ctrl = AffCtrl(config)
        assert str(ctrl.config_path) == config
        assert ctrl.dof == 13
        assert ctrl.freq == 30
        assert ctrl.scale_gain == 255 / 600
        assert_array_equal(
            ctrl.inactive_joints,
            [
                [1, 0, 0],
                [7, 100, 100],
                [8, 100, 100],
                [9, 100, 100],
                [10, 100, 100],
                [11, 100, 100],
                [12, 100, 100],
            ],
        )

    def test_init_config_alternative(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "alternative.toml")
        ctrl = AffCtrl(config)
        assert str(ctrl.config_path) == config
        assert ctrl.dof == 14
        assert ctrl.freq == 50
        assert ctrl.scale_gain == 255 / 400
        assert_array_equal(
            ctrl.inactive_joints,
            [
                [1, 400, 400],
                [3, 400, 400],
                [5, 400, 400],
                [7, 400, 400],
                [8, 200, 200],
                [10, 200, 200],
                [11, 200, 200],
                [12, 200, 200],
            ],
        )

    def test_init_config_empty(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "empty.toml")
        with pytest.warns() as record:
            ctrl = AffCtrl(config)
        assert len(record) == 2
        assert str(record[0].message) == "'chain' field is not defined"
        assert str(record[1].message) == "Control frequency is not provided, set to default: 30"
        assert ctrl.freq == 30

    def test_load_inactive_joints(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "default.toml")
        ctrl = AffCtrl(config)
        inactive_joints = {"inactive_joints": [{"index": 3}, {"index": "9,10-11", "pressure": 400}]}
        ctrl.reset_inactive_joints()
        ctrl.load_inactive_joints(inactive_joints)
        assert_array_equal(
            ctrl.inactive_joints,
            [[3, 0, 0], [9, 400, 400], [10, 400, 400], [11, 400, 400]],
        )

    @pytest.mark.parametrize(
        "input_range,expected",
        [
            ((0, 600), 255 / 600),
            ((100, 500), 255 / 400),
        ],
    )
    def test_set_input_range(self, input_range, expected) -> None:
        ctrl = AffCtrl()
        ctrl.set_input_range(input_range)
        assert ctrl.input_range == input_range
        assert ctrl.scale_gain == expected

    @pytest.mark.parametrize(
        "i,p",
        [
            (1, 10),
            (3, 30),
            ("5", 50),
            ("8", 80),
        ],
    )
    def test_set_inactive_joints(self, i, p):
        ctrl = AffCtrl()
        ctrl.set_inactive_joints(i, p)
        assert_array_equal(ctrl.inactive_joints, [[int(i), p, p]])

    @pytest.mark.parametrize(
        "seq,p",
        [
            ([0, 1, 2], 10),
            ([3, 7, 12], 30),
            ((4, 6), 50),
            ((10,), 80),
        ],
    )
    def test_set_inactive_joints_sequence(self, seq, p):
        ctrl = AffCtrl()
        ctrl.set_inactive_joints(seq, p)
        expected = [[int(i), p, p] for i in seq]
        assert_array_equal(ctrl.inactive_joints, expected)

    def test_set_inactive_joints_default_press(self):
        ctrl = AffCtrl()
        ctrl.set_inactive_joints(3)
        assert_array_equal(ctrl.inactive_joints, [[3, 0, 0]])

    @pytest.mark.parametrize(
        "pattern,pressure,expected",
        [
            (1, 10, [[1, 10, 10]]),
            ("2", 20, [[2, 20, 20]]),
            ("1-3", 30, [[1, 30, 30], [2, 30, 30], [3, 30, 30]]),
            ("2-5", 40, [[2, 40, 40], [3, 40, 40], [4, 40, 40], [5, 40, 40]]),
            ("3-3", 50, [[3, 50, 50]]),
            ("3-2", 60, np.empty(shape=(0, 3))),
            ("-3", 70, [[0, 70, 70], [1, 70, 70], [2, 70, 70], [3, 70, 70]]),
            ("10-", 80, [[10, 80, 80], [11, 80, 80], [12, 80, 80]]),
            ("1,3,5", 90, [[1, 90, 90], [3, 90, 90], [5, 90, 90]]),
            (
                "2,3-5,8",
                100,
                [
                    [2, 100, 100],
                    [3, 100, 100],
                    [4, 100, 100],
                    [5, 100, 100],
                    [8, 100, 100],
                ],
            ),
            (
                "0,10-",
                110,
                [[0, 110, 110], [10, 110, 110], [11, 110, 110], [12, 110, 110]],
            ),
            (
                "-",
                120,
                [
                    [0, 120, 120],
                    [1, 120, 120],
                    [2, 120, 120],
                    [3, 120, 120],
                    [4, 120, 120],
                    [5, 120, 120],
                    [6, 120, 120],
                    [7, 120, 120],
                    [8, 120, 120],
                    [9, 120, 120],
                    [10, 120, 120],
                    [11, 120, 120],
                    [12, 120, 120],
                ],
            ),
        ],
    )
    def test_set_inactive_joints_pattern(self, pattern, pressure, expected):
        ctrl = AffCtrl()
        ctrl.set_inactive_joints(pattern, pressure)
        assert_array_equal(ctrl.inactive_joints, expected)

    def test_set_inactive_joints_overwrite(self):
        ctrl = AffCtrl()
        ctrl.set_inactive_joints(1, 100)
        assert_array_equal(ctrl.inactive_joints, [[1, 100, 100]])
        ctrl.set_inactive_joints(2, 200)
        assert_array_equal(ctrl.inactive_joints, [[2, 200, 200]])

    @pytest.mark.parametrize(
        "pattern,p",
        [
            ("", 10),
            (",", 20),
            ("4-2", 30),
            ("a", 40),
            ([], 50),
        ],
    )
    def test_set_inactive_joints_do_nothing(self, pattern, p):
        ctrl = AffCtrl()
        ctrl.set_inactive_joints(pattern, p)
        expected = np.empty(shape=(0, 3))
        assert_array_equal(ctrl.inactive_joints, expected)

    def test_add_inactive_joints(self):
        ctrl = AffCtrl()
        ctrl.add_inactive_joints(1)
        ctrl.add_inactive_joints("7-12", 100)
        assert_array_equal(
            ctrl.inactive_joints,
            [
                [1, 0, 0],
                [7, 100, 100],
                [8, 100, 100],
                [9, 100, 100],
                [10, 100, 100],
                [11, 100, 100],
                [12, 100, 100],
            ],
        )

    @pytest.mark.parametrize(
        "seq1,seq2,p",
        [
            ([0, 1, 2], (4, 6), 10),
            ([3, 7, 12], (10,), 30),
        ],
    )
    def test_add_inactive_joints_sequence(self, seq1, seq2, p):
        ctrl = AffCtrl()
        ctrl.add_inactive_joints(seq1, p)
        ctrl.add_inactive_joints(seq2, p)
        expected = [[int(i), p, p] for i in seq1]
        expected.extend([[int(i), p, p] for i in seq2])
        assert_array_equal(ctrl.inactive_joints, expected)

    def test_add_inactive_joints_overwrite(self):
        ctrl = AffCtrl()
        ctrl.add_inactive_joints(1)
        ctrl.add_inactive_joints("7-12", 100)
        ctrl.add_inactive_joints([10, 11], 200)
        ctrl.add_inactive_joints(7, 300)
        assert_array_equal(
            ctrl.inactive_joints,
            [
                [1, 0, 0],
                [8, 100, 100],
                [9, 100, 100],
                [12, 100, 100],
                [10, 200, 200],
                [11, 200, 200],
                [7, 300, 300],
            ],
        )

    def test_reset_inactive_joints(self):
        ctrl = AffCtrl()
        ctrl.set_inactive_joints(1)
        assert_array_equal(ctrl.inactive_joints, [[1, 0, 0]])
        ctrl.reset_inactive_joints()
        assert_array_equal(ctrl.inactive_joints, np.empty(shape=(0, 3)))

    def test_get_inactive_joints_index(self):
        ctrl = AffCtrl()
        ctrl.set_inactive_joints(1)
        assert ctrl.inactive_joints_index == [1]
        assert ctrl.active_joints_index == [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        ctrl.set_inactive_joints([10, 11, 12])
        assert ctrl.inactive_joints_index == [10, 11, 12]
        assert ctrl.active_joints_index == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    @pytest.mark.parametrize(
        "i,p",
        [
            (1, 10),
            (3, 30),
            ("5", 50),
            ("8", 80),
        ],
    )
    def test_set_active_joints(self, i, p):
        ctrl = AffCtrl()
        ctrl.set_active_joints(i, p)
        inactive_index = list(range(13))
        inactive_index.pop(int(i))
        expected = [[i, p, p] for i in inactive_index]
        assert_array_equal(ctrl.inactive_joints, expected)

    def test_set_active_joints_default_press(self):
        ctrl = AffCtrl()
        ctrl.set_active_joints(3)
        inactive_index = list(range(13))
        inactive_index.pop(int(3))
        expected = [[i, 0, 0] for i in inactive_index]
        assert_array_equal(ctrl.inactive_joints, expected)

    @pytest.mark.parametrize(
        "seq,p",
        [
            (list(range(10)), 10),
            ((1, 2, 3, 5, 7, 9, 11, 12), 30),
        ],
    )
    def test_set_active_joints_sequence(self, seq, p):
        ctrl = AffCtrl()
        ctrl.set_active_joints(seq, p)
        index = list(range(13))
        for i in seq:
            index.remove(i)
        expected = [[i, p, p] for i in index]
        assert_array_equal(ctrl.inactive_joints, expected)

    @pytest.mark.parametrize(
        "pattern,p",
        [
            ("", 10),
            (",", 20),
            ("4-2", 30),
            ("a", 40),
            ([], 50),
            (None, 60),
        ],
    )
    def test_set_active_joints_inactivate_all(self, pattern, p):
        ctrl = AffCtrl()
        ctrl.set_active_joints(pattern, p)
        expected = [[i, p, p] for i in range(13)]
        assert_array_equal(ctrl.inactive_joints, expected)

    @pytest.mark.parametrize(
        "pattern,pressure,expected",
        [
            ("1-10", 30, [[0, 30, 30], [11, 30, 30], [12, 30, 30]]),
            (
                "3-3",
                50,
                [
                    [0, 50, 50],
                    [1, 50, 50],
                    [2, 50, 50],
                    [4, 50, 50],
                    [5, 50, 50],
                    [6, 50, 50],
                    [7, 50, 50],
                    [8, 50, 50],
                    [9, 50, 50],
                    [10, 50, 50],
                    [11, 50, 50],
                    [12, 50, 50],
                ],
            ),
            ("3-2", 60, [[i, 60, 60] for i in range(13)]),
            ("-8", 70, [[9, 70, 70], [10, 70, 70], [11, 70, 70], [12, 70, 70]]),
            ("3-", 80, [[0, 80, 80], [1, 80, 80], [2, 80, 80]]),
            (
                "1,3,5,7,9",
                90,
                [
                    [0, 90, 90],
                    [2, 90, 90],
                    [4, 90, 90],
                    [6, 90, 90],
                    [8, 90, 90],
                    [10, 90, 90],
                    [11, 90, 90],
                    [12, 90, 90],
                ],
            ),
            (
                "2,3-8,11",
                100,
                [
                    [0, 100, 100],
                    [1, 100, 100],
                    [9, 100, 100],
                    [10, 100, 100],
                    [12, 100, 100],
                ],
            ),
            (
                "-3,8-",
                110,
                [[4, 110, 110], [5, 110, 110], [6, 110, 110], [7, 110, 110]],
            ),
            ("-", 120, np.empty(shape=(0, 3))),
        ],
    )
    def test_set_active_joints_pattern(self, pattern, pressure, expected):
        ctrl = AffCtrl()
        ctrl.set_active_joints(pattern, pressure)
        assert_array_equal(ctrl.inactive_joints, expected)

    def test_set_active_joints_overwrite(self):
        ctrl = AffCtrl()
        ctrl.set_active_joints("10-", 100)
        assert_array_equal(
            ctrl.inactive_joints,
            [
                [0, 100, 100],
                [1, 100, 100],
                [2, 100, 100],
                [3, 100, 100],
                [4, 100, 100],
                [5, 100, 100],
                [6, 100, 100],
                [7, 100, 100],
                [8, 100, 100],
                [9, 100, 100],
            ],
        )
        ctrl.set_active_joints("-10", 200)
        assert_array_equal(ctrl.inactive_joints, [[11, 200, 200], [12, 200, 200]])

    def test_add_active_joints(self):
        ctrl = AffCtrl()
        ctrl.set_active_joints(1, 100)
        ctrl.add_active_joints("3-7")
        ctrl.add_active_joints("9-")
        assert_array_equal(
            ctrl.inactive_joints,
            [
                [0, 100, 100],
                [2, 100, 100],
                [8, 100, 100],
            ],
        )

    def test_add_active_joints_do_nothing(self):
        ctrl = AffCtrl()
        ctrl.add_active_joints(1)
        assert_array_equal(ctrl.inactive_joints, np.empty(shape=(0, 3)))

    def test_get_active_joints(self):
        ctrl = AffCtrl()
        ctrl.set_active_joints(1)
        assert ctrl.active_joints_index == [1]
        assert ctrl.inactive_joints_index == [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        ctrl.set_active_joints([10, 11, 12])
        assert ctrl.active_joints_index == [10, 11, 12]
        assert ctrl.inactive_joints_index == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_mask(self):
        config = os.path.join(CONFIG_DIR_PATH, "default.toml")
        ctrl = AffCtrl(config)
        u1 = np.ones((13,))
        u2 = np.ones((13,))
        u1, u2 = ctrl.mask(u1, u2)
        expected = np.ones((13,))
        expected[1] = 0
        expected[7:] = 100
        assert_array_equal(u1, expected)
        assert_array_equal(u2, expected)

    def test_masked_ctrl_input(self):
        u = np.ones((13,)) * 150
        config = os.path.join(CONFIG_DIR_PATH, "default.toml")
        ctrl = AffCtrl(config)
        u1, u2 = ctrl.update(0, u, u)
        expected = np.full((13,), 150 * 255 / 600)
        expected[1] = 0
        expected[7:] = 100 * 255 / 600
        assert_array_equal(u1, expected)
        assert_array_equal(u2, expected)
        assert ctrl.n_steps == 1


@pytest.mark.filterwarnings("ignore:Control frequency is not provided")
class TestAffCtrlThread:
    def test_init_config(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "default.toml")
        ctrl = AffCtrlThread(config=config)
        assert ctrl.freq == 30

    def test_init_alternative_freq(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "default.toml")
        ctrl = AffCtrlThread(config=config, freq=50)
        assert ctrl.freq == 50

    def test_set_freq(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "default.toml")
        ctrl = AffCtrlThread(config=config)
        assert ctrl.freq == 30
        ctrl.freq = 50
        assert ctrl.freq == 50

    def test_get_current_time(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "default.toml")
        ctrl = AffCtrlThread(config=config)
        assert ctrl.current_time == 0
