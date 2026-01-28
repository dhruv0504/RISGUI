# tests/test_services.py
from app.services.farfield_core import compute_farfield_pattern
from app.services.nearfield_range_beams_core import compute_nearfield_pattern

def test_farfield_basic():
    res = compute_farfield_pattern({"azi": 0})
    assert "figure" in res and "pattern" in res

def test_nearfield_basic():
    res = compute_nearfield_pattern({"param": 1})
    assert "figure" in res and "z" in res
