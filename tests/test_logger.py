import glob
import os
from pathlib import Path

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

    def test_set_labels_string(self) -> None:
        logger = Logger()
        labels = "txy"
        logger.set_labels(labels)
        assert logger.get_header() == "t,x,y"

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
        logger.dump(output_filename)

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
        logger.dump(output_filename)

        with open(output_filename, "r") as f:
            assert f.read() == expected

    def test_dump_no_data(self) -> None:
        output_filename = os.path.join(OUTPUT_DIR_PATH, "output.csv")
        if os.path.exists(output_filename):
            os.remove(output_filename)

        logger = Logger()
        logger.dump(output_filename)

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
        logger.dump(output_filename)

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
        logger.dump(output_filename)

        with open(output_filename, mode="r", newline="\r\n") as f:
            assert f.read() == expected

    def test_dump_overwrite_false(self) -> None:
        for f in glob.glob(os.path.join(OUTPUT_DIR_PATH, "*.csv")):
            os.remove(f)
        output_filename = os.path.join(OUTPUT_DIR_PATH, "output.csv")
        Path(output_filename).touch()

        expected_filename = os.path.join(OUTPUT_DIR_PATH, "output.1.csv")
        assert not os.path.exists(expected_filename)

        logger = Logger()
        logger.dump(output_filename, overwrite=False)
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
        logger.dump(output_filename, overwrite=False)
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
        logger.dump()
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
        logger.dump()

        with open(output_filename, "r") as f:
            assert f.read() == expected