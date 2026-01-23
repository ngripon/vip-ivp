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

    if isinstance(output, np.ndarray):
        np.testing.assert_array_equal(output, expected)
    else:
        assert output == expected


def test_output_dimension_with_vector_time(
        test_input, expected, time_vector=np.array([0, 1, 2]), y=np.array([[1, 2], [3, 4], [5, 6]])
):
    sut = TemporalVar(test_input)

    output = sut(time_vector, y)

    assert np.array_equal(output, expected)
