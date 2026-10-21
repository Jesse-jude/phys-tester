from __future__ import annotations

import pytest

from phys_tester.demo import generate_demo
from phys_tester.io import load_items


@pytest.fixture(scope="session")
def demo_records():
    return generate_demo(240, seed=7)


@pytest.fixture(scope="session")
def demo_items(demo_records):
    return load_items(demo_records)
