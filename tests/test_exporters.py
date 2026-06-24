"""Test result exporters."""

from cross_bench.results.exporters import to_csv, to_json, to_markdown_table


def test_to_json():
    data = {"key": "value", "number": 42}
    result = to_json(data)
    assert '"key": "value"' in result


def test_to_csv():
    rows = [
        {"name": "run1", "pass_rate": 0.85},
        {"name": "run2", "pass_rate": 0.92},
    ]
    csv_str = to_csv(rows)
    assert "name,pass_rate" in csv_str
    assert "run1,0.85" in csv_str


def test_to_csv_empty():
    assert to_csv([]) == ""


def test_to_markdown_table():
    rows = [
        {"model": "gpt-4o", "pass_rate": 0.85},
        {"model": "claude", "pass_rate": 0.92},
    ]
    md = to_markdown_table(rows)
    assert "| model | pass_rate |" in md
    assert "| gpt-4o |" in md
    assert "0.9200" in md


def test_to_markdown_empty():
    assert to_markdown_table([]) == "*No results*"
