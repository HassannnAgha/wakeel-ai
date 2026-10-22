"""Runs the calculator checks in calculators_test.js (inheritance shares, limitation dates).

The calculators run in the browser, so their tests are JavaScript; this file lets pytest
run them. Skipped when Node.js is not installed.

Run:  .venv/bin/python -m pytest tests/ -v
 or:  node tests/calculators_test.js
"""
import os
import shutil
import subprocess


def test_calculators_against_worked_cases():
    node = shutil.which("node")
    if not node:
        import pytest
        pytest.skip("Node.js is not installed")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calculators_test.js")
    run = subprocess.run([node, script], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "FAIL" not in run.stdout
