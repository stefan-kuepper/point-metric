from typing import Tuple, List
from math import sqrt
import numpy as np
from scipy.optimize import linear_sum_assignment

Point = Tuple[float, float]


def distance_sq(p1: Point, p2: Point):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def distance(p1: Point, p2: Point):
    return sqrt(distance_sq(p1, p2))


def calculate_cost_matrix(pred: List[Point], gt: List[Point]):
    pred_np = np.array(pred)
    gt_np = np.array(gt)

    # dist_mat = np.sqrt(((pred_np[:, None] - gt_np[None, :])**2).sum(-1))

    m = np.zeros([len(pred), len(gt)])

    for p in range(len(pred)):
        for g in range(len(gt)):
            m[p, g] = distance(pred[p], gt[g])

    return m


def calculate_sum_assigment_cost(cost_matrix: np.ndarray) -> float:
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    return float(cost_matrix[row_ind, col_ind].sum())


def count_extra_or_missing(pred: List[Point], gt: List[Point]):
    lp = len(pred)
    lg = len(gt)

    return abs(lp - lg)


def point_metric(pred: List[Point], gt: List[Point], k=100):
    """
    Calculates a point metric using the Hungarian algorithm with a penalty for extra or missing points.

    Args:
        pred (List[Point]): A list of predicted points.
        gt (List[Point]): A list of ground truth points.
        k (float, optional): The penalty for extra or missing points. Defaults to 100.

    Returns:
        float: The point metric.
    """
    cm = calculate_cost_matrix(pred, gt)
    cost_displacement = calculate_sum_assigment_cost(cm)
    cost_extra_missing = count_extra_or_missing(pred, gt)

    return float(cost_displacement + k * cost_extra_missing)
