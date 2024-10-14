from point_metric import calculate_sum_assigment_cost, distance, calculate_cost_matrix, point_metric
from math import isclose
import numpy as np
from itertools import permutations
from typing import Tuple
TEST_POINTS = [
    {"pred": [(30, 50), (44, 70)], "gt": [(31, 51), (43, 70)]},
    {"pred": [(44, 70)], "gt": [(31, 51), (43, 70)]},
    {"pred": [(30, 50), (44, 70), (10, 10)], "gt": [(31, 51), (43, 70)]},
    {"pred": [(30, 50), (44, 70), (31, 51), (30, 51)], "gt": [(31, 51), (43, 70)]},
    {"pred": [], "gt": []},
]


def test_distance():
    assert isclose(distance((0, 0), (0, 0)), 0)
    assert isclose(distance((1, 0), (0, 0)), 1)
    assert isclose(distance((0, 0), (1, 0)), 1)
    assert isclose(distance((0, 0), (3, 4)), 5)
    assert isclose(distance((44, 70), (43, 70)), 1)


def test_calculate_cost_matrix_shape():
    for t in TEST_POINTS:
        cm = calculate_cost_matrix(t["pred"], t["gt"])
        assert cm.shape == (len(t["pred"]), len(t["gt"]))


def test_calculate_cost_matrix_equal():
    for t in TEST_POINTS:
        cm = calculate_cost_matrix(t["pred"], t["pred"])
        assert np.array_equal(cm, cm.T)


def test_calculate_sum_assigment_cost_equal():
    for t in TEST_POINTS:
        cm = calculate_cost_matrix(t["pred"], t["pred"])
        cost = calculate_sum_assigment_cost(cm)
        assert isclose(0, cost)

def test_calculate_sum_assigment_cost_permute():
    for t in TEST_POINTS:
        p,g = t["pred"], t["pred"]
        cm1 = calculate_cost_matrix(p,g)
        cost1 = calculate_sum_assigment_cost(cm1)
        for p_perm in permutations(p):
            for g_perm in permutations(g):
                cm_p = calculate_cost_matrix(p_perm,g_perm) # pyright: ignore
                cost_p = calculate_sum_assigment_cost(cm_p)
            
                assert(isclose(cost_p, cost1))

def test_calculate_point_metric_permute():
    for t in TEST_POINTS:
        p,g = t["pred"], t["pred"]
        m1 = point_metric(p,g)
        for p_perm in permutations(p):
            for g_perm in permutations(g):
                m_perm = point_metric(p_perm,g_perm) # pyright: ignore
                assert(isclose(m1, m_perm))


if __name__ == '__main__':
    import pytest

    pytest.main()