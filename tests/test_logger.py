import glob
import os
from pathlib import Path

import numpy as np
import pytest
from affctrllib.logger import Logger

OUTPUT_DIR_PATH = os.path.join(os.path.dirname(__file__), "output")


class TestLogger:
    def test_init(self) -> None:
        logger = Logger()
        assert logger._fpath is None
        assert logger.sep == ","
        assert logger.eol == "\n"
        assert logger._labels == []
        assert logger._rawdata == []

    @pytest.mark.parametrize("sep", [",", ".", " ", "|"])
    def test_init_specify_sep(self, sep) -> None:
        logger = Logger(sep=sep)
        assert logger.sep == sep

    @pytest.mark.parametrize("eol", ["\n", "\r\n", "\r", "\0"])
    def test_init_specify_eol(self, eol) -> None:
        logger = Logger(eol=eol)
        assert logger.eol == eol

    @pytest.mark.parametrize(
        "labels,sep",
        [
            (["t"], None),
            (["t", "x", "y"], None),
            (["a", "b", "c", "d"], None),
            (["t"], " "),
            (["t", "x", "y"], "|"),
        ],
    )
    def test_set_labels(self, labels, sep) -> None:
        if sep is None:
            logger = Logger()
        else:
            logger = Logger(sep=sep)
        logger.set_labels(labels)
        if sep is None:
            assert logger.get_header() == ",".join(labels)
        else:
            assert logger.get_header() == sep.join(labels)

    def test_set_labels_tuple(self) -> None:
        logger = Logger()
        labels = ("t", "x", "y")
        logger.set_labels(labels)
        assert logger.get_header() == "t,x,y"

    def test_set_labels_dict_keys(self) -> None:
        logger = Logger()
        labels = {"t": 0.0, "x": 1.0, "y": 2.0}.keys()
        logger.set_labels(labels)
        assert logger.get_header() == "t,x,y"

    def test_set_labels_multiple_args(self) -> None:
        logger = Logger()
        label1 = "t"
        label2 = [f"q{i}" for i in range(3)]
        label3 = [f"dq{i}" for i in range(3)]
        label4 = "rq0"
        logger.set_labels(label1, label2, label3, label4)
        assert logger.get_header() == "t,q0,q1,q2,dq0,dq1,dq2,rq0"

    def test_set_labels_string_labels(self) -> None:
        logger = Logger()
        logger.set_labels("rq0", "rq1")
        assert logger.get_header() == "rq0,rq1"

    @pytest.mark.parametrize(
        "labels,additional",
        [
            (["t"], ["x", "y"]),
            (["a", "b"], ["c", "d"]),
        ],
    )
    def test_extend_labels(self, labels, additional) -> None:
        logger = Logger()
        logger.set_labels(labels)
        logger.extend_labels(additional)
        assert logger.get_header() == ",".join(labels + additional)

    def test_extend_labels_tuple(self) -> None:
        logger = Logger()
        logger.set_labels(["t"])
        logger.extend_labels(("x", "y"))
        assert logger.get_header() == ",".join(["t", "x", "y"])

    def test_extend_labels_dict_keys(self) -> None:
        logger = Logger()
        logger.set_labels(["t"])
        logger.extend_labels({"x": 1.0, "y": 2.0}.keys())
        assert logger.get_header() == ",".join(["t", "x", "y"])

    def test_extend_labels_before_set_labels(self) -> None:
        logger = Logger()
        logger.extend_labels(["t", "x", "y"])
        assert logger.get_header() == ",".join(["t", "x", "y"])

    def test_extend_labels_multiple_args(self) -> None:
        logger = Logger()
        logger.set_labels("t")
        label1 = [f"q{i}" for i in range(3)]
        label2 = [f"dq{i}" for i in range(3)]
        logger.extend_labels(label1, label2, "rq0")
        assert logger.get_header() == ",".join(
            ["t", "q0", "q1", "q2", "dq0", "dq1", "dq2", "rq0"]
        )

    def test_extend_labels_string(self) -> None:
        logger = Logger()
        logger.set_labels(["t"])
        logger.extend_labels("rq0", "rq1")
        assert logger.get_header() == ",".join(["t", "rq0", "rq1"])

    def test_get_labels_return_None_when_nothing(self) -> None:
        logger = Logger()
        assert logger.get_header() == ""

    @pytest.mark.parametrize("data", [[0, 1, 2], [4, 5, 6], ["a", "b", "c"]])
    def test_store_data(self, data) -> None:
        logger = Logger()
        logger.store_data(data)
        assert logger.get_data() == [data]

    @pytest.mark.parametrize(
        "line1,line2,line3",
        [
            ([0, 1, 2], [4, 5, 6], ["a", "b", "c"]),
            ([0, 1], [2, 3], [4, 5, 6]),
        ],
    )
    def test_store_multi_lines_data(self, line1, line2, line3) -> None:
        logger = Logger()
        logger.store_data(line1)
        logger.store_data(line2)
        logger.store_data(line3)
        assert logger.get_data() == [line1, line2, line3]

    def test_extend_data(self) -> None:
        logger = Logger()
        logger.store_data([0, 1, 2])
        logger.extend_data([3, 4])
        assert logger.get_data() == [[0, 1, 2, 3, 4]]

    def test_extend_data_2(self) -> None:
        logger = Logger()
        logger.store_data([0, 1, 2, 3, 4])
        logger.store_data([5, 6, 7])
        logger.extend_data([8, 9])
        assert logger.get_data() == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

    def test_extend_data_when_no_data_stored(self) -> None:
        logger = Logger()
        logger.extend_data([0, 1, 2])
        assert logger.get_data() == [[0, 1, 2]]

    def test_store(self) -> None:
        logger = Logger()
        arr1 = [0.1]
        arr2 = [12, 34, 56]
        arr3 = ["a", "b", "c"]
        arr4 = np.arange(1, 4) * 0.1 + 1.0
        logger.store(arr1, arr2, arr3, arr4)
        expected = [[0.1, 12, 34, 56, "a", "b", "c", 1.1, 1.2, 1.3]]
        assert logger.get_data() == expected

    def test_store_float(self) -> None:
        logger = Logger()
        t = 0.001
        arr1 = np.array([1, 2, 3])
        arr2 = [0.1, 0.2, 0.3]
        t2 = 10
        logger.store(t, arr1, arr2, t2)
        expected = [[0.001, 1, 2, 3, 0.1, 0.2, 0.3, 10]]
        assert logger.get_data() == expected

    @pytest.mark.parametrize(
        "labels,data",
        [
            (["t", "x", "y"], [[0.0, 10, 10], [0.1, 20, 30], [0.2, 30, 50]]),
            (["a", "b"], [[1, 10], [2, 20], [3, 30], [4, 40]]),
        ],
    )
    def test_dump(self, labels, data) -> None:
        output_filename = os.path.join(OUTPUT_DIR_PATH, "output.csv")
        if os.path.exists(output_filename):
            os.remove(output_filename)

        expected = ",".join(labels) + "\n"
        for d in data:
            expected += ",".join([str(x) for x in d]) + "\n"

        logger = Logger()
        logger.set_labels(labels)
        for d in data:
            logger.store_data(d)
        logger.dump(output_filename, quiet=True)

        with open(output_filename, "r") as f:
            assert f.read() == expected

    @pytest.mark.parametrize(
        "data",
        [
            ([[0.0, 10, 10], [0.1, 20, 30], [0.2, 30, 50]]),
            ([[1, 10], [2, 20], [3, 30], [4, 40]]),
        ],
    )
    def test_dump_no_header(self, data) -> None:
        output_filename = os.path.join(OUTPUT_DIR_PATH, "output.csv")
        if os.path.exists(output_filename):
            os.remove(output_filename)

        expected = ""
        for d in data:
            expected += ",".join([str(x) for x in d]) + "\n"

        logger = Logger()
        for d in data:
            logger.store_data(d)
        logger.dump(output_filename, quiet=True)

        with open(output_filename, "r") as f:
            assert f.read() == expected

    def test_dump_no_data(self) -> None:
        output_filename = os.path.join(OUTPUT_DIR_PATH, "output.csv")
        if os.path.exists(output_filename):
            os.remove(output_filename)

        logger = Logger()
        logger.dump(output_filename, quiet=True)

        with open(output_filename, "r") as f:
            assert f.read() == ""

    @pytest.mark.parametrize("sep", [",", " ", "|"])
    def test_dump_specify_sep(self, sep) -> None:
        output_filename = os.path.join(OUTPUT_DIR_PATH, "output.csv")
        if os.path.exists(output_filename):
            os.remove(output_filename)

        labels = ["t", "x", "y"]
        data = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        expected = sep.join(labels) + "\n"
        for d in data:
            expected += sep.join([str(x) for x in d]) + "\n"

        logger = Logger(sep=sep)
        logger.set_labels(labels)
        for d in data:
            logger.store_data(d)
        logger.dump(output_filename, quiet=True)

        with open(output_filename, "r") as f:
            assert f.read() == expected

    @pytest.mark.parametrize("eol", ["\n", "\r\n", "\0"])
    def test_dump_specify_eol(self, eol) -> None:
        output_filename = os.path.join(OUTPUT_DIR_PATH, "output.csv")
        if os.path.exists(output_filename):
            os.remove(output_filename)

        sep = ","
        labels = ["t", "x", "y"]
        data = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        expected = sep.join(labels) + eol
        for d in data:
            expected += sep.join([str(x) for x in d]) + eol

        logger = Logger(eol=eol)
        logger.set_labels(labels)
        for d in data:
            logger.store_data(d)
        logger.dump(output_filename, quiet=True)

        with open(output_filename, mode="r", newline="\r\n") as f:
            assert f.read() == expected

    def test_fpath_setter(self) -> None:
        logger = Logger()
        assert logger._fpath is None
        output_filename = os.path.join(OUTPUT_DIR_PATH, "output.csv")
        logger.fpath = output_filename
        assert logger.fpath == output_filename

    def test_set_filename(self) -> None:
        logger = Logger()
        assert logger._fpath is None
        output_filename = os.path.join(OUTPUT_DIR_PATH, "output.csv")
        logger.set_filename(output_filename)
        assert logger.fpath == output_filename

    def test_dump_overwrite_false(self) -> None:
        for f in glob.glob(os.path.join(OUTPUT_DIR_PATH, "*.csv")):
            os.remove(f)
        output_filename = os.path.join(OUTPUT_DIR_PATH, "output.csv")
        Path(output_filename).touch()

        expected_filename = os.path.join(OUTPUT_DIR_PATH, "output.1.csv")
        assert not os.path.exists(expected_filename)

        logger = Logger()
        logger.dump(output_filename, overwrite=False, quiet=True)
        assert os.path.exists(expected_filename)

    def test_dump_overwrite_false_2(self) -> None:
        for f in glob.glob(os.path.join(OUTPUT_DIR_PATH, "*.csv")):
            os.remove(f)
        output_filename = os.path.join(OUTPUT_DIR_PATH, "output.csv")
        Path(output_filename).touch()
        for i in [1, 2, 3]:
            Path(os.path.join(OUTPUT_DIR_PATH, f"output.{i}.csv")).touch()

        expected_filename = os.path.join(OUTPUT_DIR_PATH, "output.4.csv")
        assert not os.path.exists(expected_filename)

        logger = Logger()
        logger.dump(output_filename, overwrite=False, quiet=True)
        assert os.path.exists(expected_filename)

    def test_dump_specify_no_fname(self, capsys) -> None:
        labels = ["t", "x", "y"]
        data = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        expected = ",".join(labels) + "\n"
        for d in data:
            expected += ",".join([str(x) for x in d]) + "\n"

        logger = Logger()
        logger.set_labels(labels)
        for d in data:
            logger.store_data(d)
        logger.dump(quiet=True)
        captured = capsys.readouterr()
        assert captured.out == expected

    def test_dump_specify_fname_in_init(self) -> None:
        output_filename = os.path.join(OUTPUT_DIR_PATH, "output.csv")
        if os.path.exists(output_filename):
            os.remove(output_filename)

        labels = ["t", "x", "y"]
        data = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        expected = ",".join(labels) + "\n"
        for d in data:
            expected += ",".join([str(x) for x in d]) + "\n"

        logger = Logger(output_filename)
        logger.set_labels(labels)
        for d in data:
            logger.store_data(d)
        logger.dump(quiet=True)

        with open(output_filename, "r") as f:
            assert f.read() == expected

    def test_dump_print_result(self, capsys) -> None:
        output_filename = os.path.join(OUTPUT_DIR_PATH, "output.csv")
        if os.path.exists(output_filename):
            os.remove(output_filename)
        expected = f"Saving data in <{str(output_filename)}>... done.\n"

        labels = ["t", "x", "y"]
        data = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        logger = Logger()
        logger.set_labels(labels)
        for d in data:
            logger.store_data(d)
        logger.dump(output_filename)
        captured = capsys.readouterr()
        assert captured.out == expected

    def test_dump_quiet(self, capsys) -> None:
        output_filename = os.path.join(OUTPUT_DIR_PATH, "output.csv")
        if os.path.exists(output_filename):
            os.remove(output_filename)
        expected = ""

        labels = ["t", "x", "y"]
        data = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        logger = Logger()
        logger.set_labels(labels)
        for d in data:
            logger.store_data(d)
        logger.dump(output_filename, quiet=True)
        captured = capsys.readouterr()
        assert captured.out == expected
