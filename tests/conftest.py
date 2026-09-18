import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: statistical validation by simulation; run with -m slow")


@pytest.fixture(scope="session")
def sim():
    from sublift.datasets import simulate_experiment

    return simulate_experiment(n=12_000, seed=11, treatment_discount=0.3)
