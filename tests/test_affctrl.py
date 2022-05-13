import os

import numpy as np
import pytest
from affctrllib.affctrl import AffCtrl, AffCtrlThread, FeedbackPID, FeedbackPIDF
from numpy.testing import assert_array_equal

CONFIG_DIR_PATH = os.path.join(os.path.dirname(__file__), "config")


class TestFeedbackPID:
    def test_init(self) -> None:
        _ = FeedbackPID()
        assert True

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
        _ = FeedbackPIDF()
        assert True

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
class TestAffCtrl:
    def test_init(self) -> None:
        ctrl = AffCtrl()
        assert isinstance(ctrl, AffCtrl)
        assert ctrl.freq == 30
        assert ctrl.dt == 1.0 / 30
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
        assert isinstance(ctrl.feedback_scheme, FeedbackPID)
        assert_array_equal(ctrl.feedback_scheme.kP, np.array([20] * 13))
        assert_array_equal(ctrl.feedback_scheme.kD, np.array([200] * 13))
        assert_array_equal(ctrl.feedback_scheme.kI, np.array([2] * 13))
        assert_array_equal(ctrl.feedback_scheme.stiff, np.array([150] * 13))

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
        assert isinstance(ctrl.feedback_scheme, FeedbackPIDF)
        assert_array_equal(ctrl.feedback_scheme.kP, np.array([3] * 13))
        assert_array_equal(ctrl.feedback_scheme.kD, np.array([30] * 13))
        assert_array_equal(ctrl.feedback_scheme.kI, np.array([0.3] * 13))
        assert_array_equal(ctrl.feedback_scheme.stiff, np.array([180] * 13))

    def test_init_config_empty(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "empty.toml")
        with pytest.warns() as record:
            ctrl = AffCtrl(config)
        assert len(record) == 2
        assert str(record[0].message) == "'chain' field is not defined"
        assert (
            str(record[1].message)
            == "Control frequency is not provided, set to default: 30"
        )
        assert ctrl.freq == 30

    def test_load_inactive_joints(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "default.toml")
        ctrl = AffCtrl(config)
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
        ctrl = AffCtrl(config)
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

    def test_reset_inactive_joints(self):
        ctrl = AffCtrl()
        ctrl.set_inactive_joints(1)
        assert_array_equal(ctrl.inactive_joints, [[1, 0, 0]])
        ctrl.reset_inactive_joints()
        assert_array_equal(ctrl.inactive_joints, np.empty(shape=(0, 3)))

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
        z = np.zeros((13,))
        config = os.path.join(CONFIG_DIR_PATH, "default.toml")
        ctrl = AffCtrl(config)
        u1, u2 = ctrl.update(0, z, z, z, z, z, z)
        expected = np.full((13,), 150 * 255 / 600)
        expected[1] = 0
        expected[7:] = 100 * 255 / 600
        assert_array_equal(u1, expected)
        assert_array_equal(u2, expected)


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
