import pytest
import tests.utils.debug_data as debug_data
import cnn_from_scratch.convolver as convolver
import cnn_from_scratch.cnn as cnn

@pytest.fixture
def dbg():
    return debug_data.Debug_data()

@pytest.fixture
def convolver_obj():
    return convolver.Convolver()

@pytest.fixture
def cnn_obj(convolver_obj):
    return cnn.CNN(convolver_obj)

@pytest.fixture
def epsilon():
    return 1e-6