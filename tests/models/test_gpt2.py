from gradling.models.gpt2.chat import right_pad


def test_right_pad():
    assert right_pad([1, 2, 3], 5, 0) == [1, 2, 3, 0, 0]
    assert right_pad([1, 2, 3], 6, 0) == [1, 2, 3, 0, 0, 0]
    assert right_pad([1, 2, 3], 3, 0) == [1, 2, 3]
    assert right_pad([1, 2, 3], 2, 0) == [1, 2, 3]
