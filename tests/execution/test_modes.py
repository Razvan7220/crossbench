"""Test execution mode helper functions."""

from cross_bench.execution.modes import _extract_bash_commands, _extract_code_block


def test_extract_bash_commands():
    text = '''Let me check the files.
```bash
ls -la
```
Then run tests.
```bash
python -m pytest
```
'''
    commands = _extract_bash_commands(text)
    assert len(commands) == 2
    assert "ls -la" in commands[0]
    assert "python -m pytest" in commands[1]


def test_extract_bash_commands_empty():
    assert _extract_bash_commands("No commands here.") == []


def test_extract_code_block():
    text = '''Here's the solution:
```python
def add(a, b):
    return a + b
```
'''
    code = _extract_code_block(text)
    assert "def add(a, b):" in code
    assert "return a + b" in code


def test_extract_code_block_last():
    text = '''First attempt:
```python
x = 1
```
Better version:
```python
x = 2
```
'''
    code = _extract_code_block(text)
    assert "x = 2" in code
