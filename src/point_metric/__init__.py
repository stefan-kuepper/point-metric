from typing import Tuple, List
from math import sqrt
import numpy as np
from scipy.optimize import linear_sum_assignment

Point = Tuple[float, float]


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points of arbitrary dimension"""
    return np.sqrt(np.sum((p1 - p2) ** 2))


def calculate_cost_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Calculate cost matrix between two sets of points.

    Args:
        pred: Array of shape (N, D) where N is number of points and D is dimension
        gt: Array of shape (M, D) where M is number of points and D is dimension
    """
    # Verify inputs have same dimension
    if pred.shape[1] != gt.shape[1]:
        raise ValueError(
            f"Point dimensions must match. Got {pred.shape[1]} and {gt.shape[1]}"
        )

    # More efficient vectorized version
    return np.sqrt(((pred[:, None] - gt[None, :]) ** 2).sum(-1))


def calculate_sum_assigment_cost(cost_matrix: np.ndarray) -> float:
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return float(cost_matrix[row_ind, col_ind].sum())


def count_extra_or_missing(pred: np.ndarray, gt: np.ndarray) -> int:
    return abs(len(pred) - len(gt))


def point_metric(pred: np.ndarray, gt: np.ndarray, k: float = 100) -> float:
    """
    Calculates a point metric using the Hungarian algorithm with a penalty for extra or missing points.

    Args:
        pred: Array of shape (N, D) where N is number of points and D is dimension
        gt: Array of shape (M, D) where M is number of points and D is dimension
        k: The penalty for extra or missing points. Defaults to 100.

    Returns:
        float: The point metric.
    """
    # Input validation
    if not isinstance(pred, np.ndarray) or not isinstance(gt, np.ndarray):
        pred = np.array(pred)
        gt = np.array(gt)

    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError("Input arrays must be 2-dimensional")

    if pred.shape[1] != gt.shape[1]:
        raise ValueError(
            f"Point dimensions must match. Got {pred.shape[1]} and {gt.shape[1]}"
        )

    cm = calculate_cost_matrix(pred, gt)
    cost_displacement = calculate_sum_assigment_cost(cm)
    cost_extra_missing = count_extra_or_missing(pred, gt)

    return float(cost_displacement + k * cost_extra_missing)
