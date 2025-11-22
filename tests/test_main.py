"""
Tests for main module.
"""
from led_detection.main import add

def test_add():
    """
    Test add function.
    """
    assert add(1, 2) == 3
