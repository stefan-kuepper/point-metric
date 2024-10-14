# %% 
from point_metric import calculate_sum_assigment_cost, distance, calculate_cost_matrix, point_metric
from math import isclose
import numpy as np
from scipy.optimize import linear_sum_assignment
TEST_POINTS = [
    {"pred": [(30, 50), (44, 70)], "gt": [(31, 51), (43, 70)]},
    {"pred": [(30, 50), (44, 70)], "gt": [(30, 50), (44, 70)]},
    {"pred": [(44, 70)], "gt": [(31, 51), (43, 70)]},
    {"pred": [(30, 50), (44, 70), (10, 10)], "gt": [(31, 51), (43, 70)]},
    {"pred": [(30, 50), (44, 70), (31, 51), (30, 51)], "gt": [(31, 51), (43, 70)]},
    {"pred": [], "gt": []},
]

# %%

cms = []

for t in TEST_POINTS:
    cm = calculate_cost_matrix(t['pred'], t['gt'])
    c = calculate_sum_assigment_cost(cm)
    print (cm)
    cms.append(cm)

# %%

for cm in cms:
    r,c = linear_sum_assignment(cm)
    print((r,c))
# %%
for t in TEST_POINTS:
    p, g = t['pred'], t['gt']
    print(f"Predicted: {p}\nGT       :{g}")
    m = point_metric(t['pred'], t['gt'], k=10)
    print(f"metric: {m}")
    print()
# %%
