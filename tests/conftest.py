import os

import pytest

import vip_ivp as vip


@pytest.fixture(autouse=True)
def clear_solver_before_tests():
    vip.clear()


@pytest.fixture(scope="session", autouse=True)
def set_base_directory():
    # Store the current working directory
    original_directory = os.getcwd()

    # Change to the base directory (replace with your project root path)
    base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(base_directory)

    # Yield to allow tests to run in the base directory
    yield

    # Change back to the original directory after tests
    os.chdir(original_directory)
