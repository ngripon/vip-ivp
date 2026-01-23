import numpy as np
import pytest

from src.vip_ivp.domain.variables import TemporalVar


@pytest.mark.parametrize(
    "test_input,expected",
    [
        # Scalars
        (5.2, 5.2),
        (None, None),
        # Collections
        (np.array([1.1, 2, 3]), np.array([1.1, 2, 3])),
        ([1.1, 2, 3], np.array([1.1, 2, 3])),
        ({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2, "c": 3}),
        # Recursive collections
        (np.arange(6).reshape(2, 3), np.arange(6).reshape(2, 3)),
        ([[0, 1, 2], [3, 4, 5]], np.arange(6).reshape(2, 3)),
        (
                {"a": 1, "b": {"c": 2, "d": 5.5}},
                {"a": 1, "b": {"c": 2, "d": 5.5}}
        ),
        # Function
        (lambda t: 5 * t + 1, 1),
        (lambda t, y: 5 * t + 1 + y[0] + y[1], 4),
    ]
)
def test_output_dimension_with_scalar_time(test_input, expected, timestamp=0, y=np.array([1, 2])):
    sut = TemporalVar(test_input)

    output = sut(timestamp, y)

    assert_equality(output, expected)

@pytest.mark.parametrize(
    "test_input,expected",
    [
        # Scalars
        (5.2, np.array([5.2, 5.2, 5.2])),
        (None, np.array([None, None, None])),
        # Collections
        (np.array([10, 20]), np.array([[10, 10, 10], [20, 20, 20]])),
        (
                np.arange(6).reshape(2, 3),
                np.array([[10, 10, 10], [20, 20, 20]])),  # TODO: Find what is the right result
        ({"a": 1, "b": 2}, {"a": np.array([1, 1, 1]), "b": np.array([2, 2, 2])}),

        # Function
        (lambda t: t + 1, np.array([1, 2, 3])),
        (lambda t, y: t + y[0] + y[1], np.array([5, 8, 11])),
    ]
)
def test_output_dimension_with_vector_time(
        test_input, expected, time_vector=np.array([0, 1, 2]), y=np.array([[1, 2, 3], [4, 5, 6]])
):
    sut = TemporalVar(test_input)

    output = sut(time_vector, y)

    assert_equality(output, expected)


def assert_equality(a, b):
    if isinstance(a, dict):
        if not isinstance(b, dict) or list(a.keys()) != list(b.keys()):
            assert False
        [assert_equality(xa, xb) for xa, xb in zip(a.values(), b.values())]
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    else:
        assert a == b
