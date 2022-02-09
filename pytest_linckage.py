from linckage import *

def test_linckage_desequilibrium():
    assert linckage_desequilibrium[np.array([1, 0, 1, 0]),\
     np.array([1, 0, 1, 0])] == 0.2, "test1"
