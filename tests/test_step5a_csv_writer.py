import csv

from experiments.run_step5a_mutual_hil_smoke import _write_dict_rows_csv


def test_write_dict_rows_csv_accepts_heterogeneous_rows(tmp_path):
    path = tmp_path / "heterogeneous.csv"
    rows = [{"a": 1}, {"a": 2, "b": 3}, {"c": 4}]

    _write_dict_rows_csv(path, rows)

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        written_rows = list(reader)

    assert reader.fieldnames == ["a", "b", "c"]
    assert written_rows == [
        {"a": "1", "b": "", "c": ""},
        {"a": "2", "b": "3", "c": ""},
        {"a": "", "b": "", "c": "4"},
    ]
