import numpy as np
import pytest

import vvtk_fastcsv as fc


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture(scope="session")
def build_info():
    info = fc.build_info()
    assert info["cpp_extension"], "the C++ extension must be built to run the tests"
    return info
