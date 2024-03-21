import numpy as np
import pytest


def f():
    raise SystemExit(1)
def test_mytest():
    with pytest.raises(SystemExit):
        f()

class TestVanilla(object):
    def test_sanity(self):
        assert True

class TestMean(object):
    def test_sanity(self):
        assert True

class TestMedian(object):
    def test_sanity(self):
        return True

class TestInterquartile(object):
    def test_sanity(self):
        return True

class TestWindsor(object):
    def test_sanity(self):
        return True
