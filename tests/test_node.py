from node import classify_sequence


def test_sequence_normal():
    result = classify_sequence(10, 11)
    assert result.gap == 0
    assert not result.duplicate_or_old


def test_sequence_gap():
    result = classify_sequence(10, 14)
    assert result.gap == 3


def test_sequence_wrap():
    result = classify_sequence(0xFFFFFFFF, 0)
    assert result.gap == 0
