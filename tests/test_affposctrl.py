import os

import numpy as np
import pytest
from affctrllib.affposctrl import (
    AffPosCtrl,
    AffPosCtrlThread,
    FeedbackPID,
    FeedbackPIDF,
)
from numpy.testing import assert_array_equal

CONFIG_DIR_PATH = os.path.join(os.path.dirname(__file__), "config")


class TestFeedbackPID:
    def test_init(self) -> None:
        pid = FeedbackPID()
        assert pid.scheme_name == "pid"

    @pytest.mark.parametrize("kP", [50, [10, 20], np.array([10, 20, 30])])
    def test_init_with_kP(self, kP) -> None:
        pid = FeedbackPID(kP=kP)
        if isinstance(kP, np.ndarray):
            assert_array_equal(pid.kP, kP)
        else:
            assert pid.kP == kP

    @pytest.mark.parametrize("kD", [500, [100, 200], np.array([100, 200, 300])])
    def test_init_with_kD(self, kD) -> None:
        pid = FeedbackPID(kD=kD)
        if isinstance(kD, np.ndarray):
            assert_array_equal(pid.kD, kD)
        else:
            assert pid.kD == kD

    @pytest.mark.parametrize("kI", [5, [1, 2], np.array([1, 2, 3])])
    def test_init_with_kI(self, kI) -> None:
        pid = FeedbackPID(kI=kI)
        if isinstance(kI, np.ndarray):
            assert_array_equal(pid.kI, kI)
        else:
            assert pid.kI == kI

    @pytest.mark.parametrize("stiff", [127, [150, 150], np.array([200, 200, 200])])
    def test_init_with_stiff(self, stiff) -> None:
        pid = FeedbackPID(stiff=stiff)
        if isinstance(stiff, np.ndarray):
            assert_array_equal(pid.stiff, stiff)
        else:
            assert pid.stiff == stiff

    @pytest.mark.parametrize("kP", [50, [10, 20], np.array([10, 20, 30])])
    def test_kP_setter(self, kP) -> None:
        pid = FeedbackPID()
        pid.kP = kP
        if isinstance(kP, np.ndarray):
            assert_array_equal(pid.kP, kP)
        else:
            assert pid.kP == kP

    @pytest.mark.parametrize("kD", [500, [100, 200], np.array([100, 200, 300])])
    def test_kD_setter(self, kD) -> None:
        pid = FeedbackPID()
        pid.kD = kD
        if isinstance(kD, np.ndarray):
            assert_array_equal(pid.kD, kD)
        else:
            assert pid.kD == kD

    @pytest.mark.parametrize("kI", [5, [1, 2], np.array([1, 2, 3])])
    def test_kI_setter(self, kI) -> None:
        pid = FeedbackPID()
        pid.kI = kI
        if isinstance(kI, np.ndarray):
            assert_array_equal(pid.kI, kI)
        else:
            assert pid.kI == kI

    @pytest.mark.parametrize("stiff", [127, [150, 150], np.array([200, 200, 200])])
    def test_stiff_setter(self, stiff) -> None:
        pid = FeedbackPID()
        pid.stiff = stiff
        if isinstance(stiff, np.ndarray):
            assert_array_equal(pid.stiff, stiff)
        else:
            assert pid.stiff == stiff

    @pytest.mark.parametrize(
        "t,q,dq,qdes,dqdes,expected",
        [
            (0, 0, 0, 5, 0, 5),
            (1, 2, 4, 5, 0, 3),
            (2, 4, 2, 5, 0, 1),
            (3, 6, -2, 5, 0, -1),
            (4, 8, -4, 5, 0, -3),
        ],
    )
    def test_positional_feedback_kP(self, t, q, dq, qdes, dqdes, expected) -> None:
        pid = FeedbackPID(kP=1, kD=0, kI=0)
        u = pid.positional_feedback(t, q, dq, qdes, dqdes)
        assert u == expected

    @pytest.mark.parametrize(
        "t,q,dq,qdes,dqdes,expected",
        [
            (0, 0, 0, 5, 0, 5),
            (1, 2, 4, 5, 0, -1),
            (2, 4, 2, 5, 0, -1),
            (3, 6, -2, 5, 0, 1),
            (4, 8, -4, 5, 0, 1),
        ],
    )
    def test_positional_feedback_kP_kD(self, t, q, dq, qdes, dqdes, expected) -> None:
        pid = FeedbackPID(kP=1, kD=1, kI=0)
        u = pid.positional_feedback(t, q, dq, qdes, dqdes)
        assert u == expected

    def test_positional_feedback_kP_kD_kI(self) -> None:
        pid = FeedbackPID(kP=1, kD=1, kI=1)
        q = [0, 2, 4, 6, 8]
        dq = [0, 4, 2, -2, -4]
        qdes = [5, 5, 5, 5, 5]
        dqdes = [0, 0, 0, 0, 0]
        expected = [10, 7, 8, 9, 6]
        for t in range(5):
            u = pid.positional_feedback(t, q[t], dq[t], qdes[t], dqdes[t])
            assert u == expected[t]

    def test_update(self) -> None:
        pid = FeedbackPID(kP=1, kD=1, kI=1, stiff=2)
        q = [0, 2, 4, 6, 8]
        dq = [0, 4, 2, -2, -4]
        qdes = [5, 5, 5, 5, 5]
        dqdes = [0, 0, 0, 0, 0]
        expected = [(12, -8), (9, -5), (10, -6), (11, -7), (8, -4)]
        for t in range(5):
            ua, ub = pid.update(t, 0, 0, q[t], dq[t], qdes[t], dqdes[t])
            assert ua, ub == expected[t]


class TestFeedbackPIDF:
    def test_init(self) -> None:
        pidf = FeedbackPIDF()
        assert pidf.scheme_name == "pidf"

    @pytest.mark.parametrize("press_gain", [0.1, [0.2, 0.2], np.array([0.3, 0.3, 0.3])])
    def test_init_with_stiff(self, press_gain) -> None:
        pidf = FeedbackPIDF(press_gain=press_gain)
        if isinstance(press_gain, np.ndarray):
            assert_array_equal(pidf.press_gain, press_gain)
        else:
            assert pidf.press_gain == press_gain

    @pytest.mark.parametrize("press_gain", [127, [150, 150], np.array([200, 200, 200])])
    def test_press_gain_setter(self, press_gain) -> None:
        pidf = FeedbackPIDF()
        pidf.press_gain = press_gain
        if isinstance(press_gain, np.ndarray):
            assert_array_equal(pidf.press_gain, press_gain)
        else:
            assert pidf.press_gain == press_gain


@pytest.mark.filterwarnings("ignore:Control frequency is not provided")
class TestAffPosCtrl:
    def test_init(self) -> None:
        ctrl = AffPosCtrl()
        assert isinstance(ctrl, AffPosCtrl)
        assert ctrl.freq == 30
        assert ctrl.dt == 1.0 / 30
        assert ctrl.inactive_joints.shape == (0, 3)

    @pytest.mark.parametrize("dt,freq", [(0.01, 100), (0.001, 1000), (0.02, 50)])
    def test_init_specify_dt(self, dt, freq) -> None:
        ctrl = AffPosCtrl(dt=dt)
        assert ctrl.dt == dt
        assert ctrl.freq == freq

    @pytest.mark.parametrize("freq,dt", [(100, 0.01), (1000, 0.001), (30, 1.0 / 30)])
    def test_init_specify_freq(self, freq, dt) -> None:
        ctrl = AffPosCtrl(freq=freq)
        assert ctrl.dt == dt
        assert ctrl.freq == freq

    def test_init_error_both_of_dt_freq_specified(self) -> None:
        dt = 0.01
        freq = 100
        with pytest.raises(ValueError) as excinfo:
            _ = AffPosCtrl(dt=dt, freq=freq)
        assert "Unable to specify DT and FREQ simultaneously" in str(excinfo.value)

    @pytest.mark.parametrize("dt,freq", [(0.01, 100), (0.001, 1000), (0.02, 50)])
    def test_dt_setter(self, dt, freq):
        ctrl = AffPosCtrl(dt=0.01)
        ctrl.dt = dt
        assert ctrl.dt == dt
        assert ctrl.freq == freq

    @pytest.mark.parametrize("freq,dt", [(100, 0.01), (1000, 0.001), (30, 1.0 / 30)])
    def test_freq_setter(self, dt, freq):
        ctrl = AffPosCtrl(dt=0.01)
        ctrl.freq = freq
        assert ctrl.dt == dt
        assert ctrl.freq == freq

    def test_init_config(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "default.toml")
        ctrl = AffPosCtrl(config)
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
        assert isinstance(ctrl.feedback_scheme, FeedbackPID)
        assert ctrl.feedback_scheme.kP.dtype == float
        assert ctrl.feedback_scheme.kD.dtype == float
        assert ctrl.feedback_scheme.kI.dtype == float
        assert ctrl.feedback_scheme.stiff.dtype == float
        assert_array_equal(ctrl.feedback_scheme.kP, np.array([20] * 13))
        assert_array_equal(ctrl.feedback_scheme.kD, np.array([200] * 13))
        assert_array_equal(ctrl.feedback_scheme.kI, np.array([2] * 13))
        assert_array_equal(ctrl.feedback_scheme.stiff, np.array([150] * 13))

    def test_init_config_alternative(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "alternative.toml")
        ctrl = AffPosCtrl(config)
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
        assert isinstance(ctrl.feedback_scheme, FeedbackPIDF)
        assert ctrl.feedback_scheme.kP.dtype == float
        assert ctrl.feedback_scheme.kD.dtype == float
        assert ctrl.feedback_scheme.kI.dtype == float
        assert ctrl.feedback_scheme.stiff.dtype == float
        assert ctrl.feedback_scheme.press_gain.dtype == float
        assert_array_equal(ctrl.feedback_scheme.kP, np.array([3] * 13))
        assert_array_equal(ctrl.feedback_scheme.kD, np.array([30] * 13))
        assert_array_equal(ctrl.feedback_scheme.kI, np.array([0.3] * 13))
        assert_array_equal(ctrl.feedback_scheme.stiff, np.array([180] * 13))
        assert_array_equal(ctrl.feedback_scheme.press_gain, np.array([0.1] * 13))

    def test_init_config_empty(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "empty.toml")
        with pytest.warns() as record:
            ctrl = AffPosCtrl(config)
        assert len(record) == 2
        assert str(record[0].message) == "'chain' field is not defined"
        assert (
            str(record[1].message)
            == "Control frequency is not provided, set to default: 30"
        )
        assert ctrl.freq == 30

    def test_load_inactive_joints(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "default.toml")
        ctrl = AffPosCtrl(config)
        inactive_joints = {
            "inactive_joints": [{"index": 3}, {"index": "9,10-11", "pressure": 400}]
        }
        ctrl.reset_inactive_joints()
        ctrl.load_inactive_joints(inactive_joints)
        assert_array_equal(
            ctrl.inactive_joints,
            [[3, 0, 0], [9, 400, 400], [10, 400, 400], [11, 400, 400]],
        )

    def test_load_feedback_scheme(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "default.toml")
        ctrl = AffPosCtrl(config)
        ctrl.load_feedback_scheme("pidf")
        assert isinstance(ctrl.feedback_scheme, FeedbackPIDF)
        assert_array_equal(ctrl.feedback_scheme.kP, np.array([10] * 13))
        assert_array_equal(ctrl.feedback_scheme.kD, np.array([100] * 13))
        assert_array_equal(ctrl.feedback_scheme.kI, np.array([1] * 13))
        assert_array_equal(ctrl.feedback_scheme.stiff, np.array([120] * 13))
        assert_array_equal(ctrl.feedback_scheme.press_gain, np.array([1] * 13))

    @pytest.mark.parametrize(
        "input_range,expected",
        [
            ((0, 600), 255 / 600),
            ((100, 500), 255 / 400),
        ],
    )
    def test_set_input_range(self, input_range, expected) -> None:
        ctrl = AffPosCtrl()
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
        ctrl = AffPosCtrl()
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
        ctrl = AffPosCtrl()
        ctrl.set_inactive_joints(seq, p)
        expected = [[int(i), p, p] for i in seq]
        assert_array_equal(ctrl.inactive_joints, expected)

    def test_set_inactive_joints_default_press(self):
        ctrl = AffPosCtrl()
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
        ctrl = AffPosCtrl()
        ctrl.set_inactive_joints(pattern, pressure)
        assert_array_equal(ctrl.inactive_joints, expected)

    def test_set_inactive_joints_overwrite(self):
        ctrl = AffPosCtrl()
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
        ctrl = AffPosCtrl()
        ctrl.set_inactive_joints(pattern, p)
        expected = np.empty(shape=(0, 3))
        assert_array_equal(ctrl.inactive_joints, expected)

    def test_add_inactive_joints(self):
        ctrl = AffPosCtrl()
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
        ctrl = AffPosCtrl()
        ctrl.add_inactive_joints(seq1, p)
        ctrl.add_inactive_joints(seq2, p)
        expected = [[int(i), p, p] for i in seq1]
        expected.extend([[int(i), p, p] for i in seq2])
        assert_array_equal(ctrl.inactive_joints, expected)

    def test_reset_inactive_joints(self):
        ctrl = AffPosCtrl()
        ctrl.set_inactive_joints(1)
        assert_array_equal(ctrl.inactive_joints, [[1, 0, 0]])
        ctrl.reset_inactive_joints()
        assert_array_equal(ctrl.inactive_joints, np.empty(shape=(0, 3)))

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
        ctrl = AffPosCtrl()
        ctrl.set_active_joints(i, p)
        inactive_index = list(range(13))
        inactive_index.pop(int(i))
        expected = [[i, p, p] for i in inactive_index]
        assert_array_equal(ctrl.inactive_joints, expected)

    def test_set_active_joints_default_press(self):
        ctrl = AffPosCtrl()
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
        ctrl = AffPosCtrl()
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
        ctrl = AffPosCtrl()
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
        ctrl = AffPosCtrl()
        ctrl.set_active_joints(pattern, pressure)
        assert_array_equal(ctrl.inactive_joints, expected)

    def test_set_active_joints_overwrite(self):
        ctrl = AffPosCtrl()
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
        ctrl = AffPosCtrl()
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
        ctrl = AffPosCtrl()
        ctrl.add_active_joints(1)
        assert_array_equal(ctrl.inactive_joints, np.empty(shape=(0, 3)))

    def test_mask(self):
        config = os.path.join(CONFIG_DIR_PATH, "default.toml")
        ctrl = AffPosCtrl(config)
        u1 = np.ones((13,))
        u2 = np.ones((13,))
        u1, u2 = ctrl.mask(u1, u2)
        expected = np.ones((13,))
        expected[1] = 0
        expected[7:] = 100
        assert_array_equal(u1, expected)
        assert_array_equal(u2, expected)

    def test_masked_ctrl_input(self):
        z = np.zeros((13,))
        config = os.path.join(CONFIG_DIR_PATH, "default.toml")
        ctrl = AffPosCtrl(config)
        u1, u2 = ctrl.update(0, z, z, z, z, z, z)
        expected = np.full((13,), 150 * 255 / 600)
        expected[1] = 0
        expected[7:] = 100 * 255 / 600
        assert_array_equal(u1, expected)
        assert_array_equal(u2, expected)


@pytest.mark.filterwarnings("ignore:Control frequency is not provided")
class TestAffPosCtrlThread:
    def test_init_config(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "default.toml")
        ctrl = AffPosCtrlThread(config=config)
        assert ctrl.freq == 30

    def test_init_alternative_freq(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "default.toml")
        ctrl = AffPosCtrlThread(config=config, freq=50)
        assert ctrl.freq == 50

    def test_set_freq(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "default.toml")
        ctrl = AffPosCtrlThread(config=config)
        assert ctrl.freq == 30
        ctrl.freq = 50
        assert ctrl.freq == 50

    def test_get_current_time(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "default.toml")
        ctrl = AffPosCtrlThread(config=config)
        assert ctrl.current_time == 0

    def test_reset_trajectory(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "default.toml")
        ctrl = AffPosCtrlThread(config=config)
        ctrl.reset_trajectory()
        assert len(ctrl._qdes_func(0)) == ctrl.dof
        assert_array_equal(ctrl._qdes_func(0), 0)
        assert len(ctrl._dqdes_func(0)) == ctrl.dof
        assert_array_equal(ctrl._dqdes_func(0), 0)

    def test_reset_trajectory_float(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "default.toml")
        ctrl = AffPosCtrlThread(config=config)
        ctrl.reset_trajectory(50)
        assert len(ctrl._qdes_func(0)) == ctrl.dof
        assert_array_equal(ctrl._qdes_func(0), 50)
        assert len(ctrl._dqdes_func(0)) == ctrl.dof
        assert_array_equal(ctrl._dqdes_func(0), 0)

    def test_reset_trajectory_ndarray(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "default.toml")
        ctrl = AffPosCtrlThread(config=config)
        q0 = np.arange(ctrl.dof) * 10
        ctrl.reset_trajectory(q0)
        assert len(ctrl._qdes_func(0)) == ctrl.dof
        assert_array_equal(ctrl._qdes_func(0), q0)
        assert len(ctrl._dqdes_func(0)) == ctrl.dof
        assert_array_equal(ctrl._dqdes_func(0), 0)
