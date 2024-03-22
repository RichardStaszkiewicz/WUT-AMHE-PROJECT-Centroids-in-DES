import numpy as np
import pytest
from modules.centroids import (
    mean_centroid,
    median_centroid,
    vanila_centroid,
    windsor_centroid,
    interquartile_centroid
)


def f():
    raise SystemExit(1)


def test_mytest():
    with pytest.raises(SystemExit):
        f()


class TestVanilla(object):
    def test_one_point(self):
        p = np.array([
            [1.5, 6.8]
        ])
        w = np.array([
            1.0
        ])
        res = vanila_centroid(p, [w])

        assert res == pytest.approx([1.5, 6.8])

    def test_simple_equal_weights(self):
        p = np.array([
            [1.2, 2.5],
            [5.8, 6.1],
            [8.4, 7.4],
            [1.2, 1.6],
        ])
        w = np.array([
            1, 1, 1, 1
        ]).T
        res = vanila_centroid(p, [w])

        assert res == pytest.approx([16.6, 17.6])

    def test_simple_weighted(self):
        p = np.array([
            [1.2, 2.5],
            [5.8, 6.1],
            [8.4, 7.4],
            [1.2, 1.6],
        ])
        w = np.array([
            2, 1, 0.5, 2
        ]).T
        res = vanila_centroid(p, [w])

        assert res == pytest.approx([14.8, 18.])

    def test_zero_point(self):
        p = np.array([
            [0.0, 0.0]
        ])
        w = np.array([
            1.0
        ])
        res = vanila_centroid(p, [w])

        assert res == pytest.approx([0.0, 0.0])

    def test_duplicates(self):
        p = np.array([
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1]
        ])
        w = np.array([
            1.0,
            1.0,
            1.0,
            1.0
        ])
        res = vanila_centroid(p, [w])

        assert res == pytest.approx([4.0, 4.0])

    def test_zero_weight(self):
        p = np.array([
            [1.5, 6.8]
        ])
        w = np.array([
            0.0
        ])
        res = vanila_centroid(p, [w])

        assert res == pytest.approx([0.0, 0.0])


class TestMean(object):
    def test_one_point(self):
        p = np.array([
            [1.5, 6.8]
        ])
        res = mean_centroid(p)

        assert res == pytest.approx([1.5, 6.8])

    def test_simple(self):
        p = np.array([
            [1.2, 2.5],
            [5.8, 6.1],
            [8.4, 7.4],
            [1.2, 1.6],
        ])
        res = mean_centroid(p)

        assert res == pytest.approx([4.15, 4.4])

    def test_zero_point(self):
        p = np.array([
            [0.0, 0.0]
        ])
        res = mean_centroid(p)

        assert res == pytest.approx([0.0, 0.0])

    def test_duplicates(self):
        p = np.array([
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1]
        ])
        res = mean_centroid(p)

        assert res == pytest.approx([1.0, 1.0])


class TestMedian(object):
    def test_one_point(self):
        p = np.array([
            [1.5, 6.8]
        ])
        res = median_centroid(p)

        assert res == pytest.approx([1.5, 6.8])

    def test_simple(self):
        p = np.array([
            [1.2, 2.5],
            [5.8, 6.1],
            [8.4, 7.4],
            [1.2, 1.6],
        ])
        res = median_centroid(p)

        assert res == pytest.approx([3.5, 4.3])

    def test_zero_point(self):
        p = np.array([
            [0.0, 0.0]
        ])
        res = median_centroid(p)

        assert res == pytest.approx([0.0, 0.0])

    def test_duplicates(self):
        p = np.array([
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1]
        ])
        res = median_centroid(p)

        assert res == pytest.approx([1.0, 1.0])


class TestInterquartile(object):
    def test_one_point(self):
        p = np.array([
            [1.5, 6.8]
        ])
        res = interquartile_centroid(p, [50])

        assert res == pytest.approx([1.5, 6.8])

    def test_simple(self):
        p = np.array([
            [1.2, 2.5],
            [5.8, 6.1],
            [8.4, 7.4],
            [1.2, 1.6],
        ])
        res = interquartile_centroid(p, [10])
        assert res == pytest.approx([4.41, 4.44])
        res = interquartile_centroid(p, [20])
        assert res == pytest.approx([4.02, 4.38])
        res = interquartile_centroid(p, [50])
        assert res == pytest.approx([3.5, 4.3])

    def test_zero_point(self):
        p = np.array([
            [0.0, 0.0]
        ])
        res = interquartile_centroid(p, [50])

        assert res == pytest.approx([0.0, 0.0])

    def test_duplicates(self):
        p = np.array([
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1]
        ])
        res = interquartile_centroid(p, [10])
        assert res == pytest.approx([1, 1])
        res = interquartile_centroid(p, [20])
        assert res == pytest.approx([1, 1])
        res = interquartile_centroid(p, [50])
        assert res == pytest.approx([1, 1])


class TestWindsor(object):
    def test_one_point(self):
        p = np.array([
            [1.5, 6.8]
        ])
        res = windsor_centroid(p, [10])

        assert res == pytest.approx([1.5, 6.8])

    def test_simple(self):
        p = np.array([
            [1.2, 2.5],
            [5.8, 6.1],
            [8.4, 7.4],
            [1.2, 1.6],
        ])
        res = windsor_centroid(p, [10])
        assert res == pytest.approx([3.955, 4.37])
        res = windsor_centroid(p, [20])
        assert res == pytest.approx([3.682, 4.328])
        res = windsor_centroid(p, [50])
        assert res == pytest.approx([3.5, 4.3])

    def test_zero_point(self):
        p = np.array([
            [0.0, 0.0]
        ])
        res = windsor_centroid(p, [10])

        assert res == pytest.approx([0.0, 0.0])

    def test_duplicates(self):
        p = np.array([
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1]
        ])
        res = windsor_centroid(p, [10])

        assert res == pytest.approx([1.0, 1.0])
