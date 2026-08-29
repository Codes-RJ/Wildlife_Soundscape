import math

from environment import calculate_speed_of_sound_mps


def test_speed_of_sound_increases_with_temperature():
    cold = calculate_speed_of_sound_mps(10.0, 50.0, 1013.25)
    warm = calculate_speed_of_sound_mps(30.0, 50.0, 1013.25)
    assert warm > cold
    assert 330.0 < cold < 350.0
    assert 345.0 < warm < 360.0


def test_humidity_has_small_positive_effect():
    dry = calculate_speed_of_sound_mps(25.0, 10.0, 1013.25)
    humid = calculate_speed_of_sound_mps(25.0, 90.0, 1013.25)
    assert humid > dry
    assert humid - dry < 3.0
