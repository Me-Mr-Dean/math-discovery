import types
from src.utils import prefix_suffix_utils


class MockIndex(list):
    def __init__(self, data):
        super().__init__(data)
        self.name = None


class MockDataFrame:
    def __init__(self, fill, index, columns):
        self.index = MockIndex(index)
        self.columns = MockIndex(columns)
        self.data = {r: {c: fill for c in columns} for r in index}

    class Loc:
        def __init__(self, df):
            self.df = df

        def __setitem__(self, key, value):
            row, col = key
            self.df.data[row][col] = value

        def __getitem__(self, key):
            row, col = key
            return self.df.data[row][col]

    @property
    def loc(self):
        return MockDataFrame.Loc(self)


class MockPandas(types.SimpleNamespace):
    DataFrame = MockDataFrame


def patch_pandas(monkeypatch):
    monkeypatch.setattr(prefix_suffix_utils, "pd", MockPandas)


def test_prefix_suffix_matrix_simple(monkeypatch):
    patch_pandas(monkeypatch)

    df = prefix_suffix_utils.generate_prefix_suffix_matrix([123, 145, 267], 1, 1)

    assert list(df.index) == [1, 2]
    assert df.index.name == "prefix_1d"
    assert list(df.columns) == ["3", "5", "7"]
    assert df.columns.name == "suffix_1d"
    assert df.data == {
        1: {"3": 1, "5": 1, "7": 0},
        2: {"3": 0, "5": 0, "7": 1},
    }


def test_prefix_suffix_matrix_multi_digits(monkeypatch):
    patch_pandas(monkeypatch)

    df = prefix_suffix_utils.generate_prefix_suffix_matrix([101, 201, 202], 2, 1)

    assert list(df.index) == [10, 20]
    assert list(df.columns) == ["1", "2"]
    assert df.data[10] == {"1": 1, "2": 0}
    assert df.data[20] == {"1": 1, "2": 1}
