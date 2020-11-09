import numpy as np
import matplotlib.pyplot as plt


def check_clockwise_polygon(polygon):
    """
    sum over edges: sum (x2 − x1)(y2 + y1)
    :param polygon:
    :return:
    """
    n_vertices = polygon.shape[0]
    sum_edges = 0
    for i in range(n_vertices):
        x1 = polygon[i % n_vertices, 0]
        y1 = polygon[i % n_vertices, 1]
        x2 = polygon[(i + 1) % n_vertices, 0]
        y2 = polygon[(i + 1) % n_vertices, 1]

        sum_edges += (x2 - x1) * (y2 + y1)

    if sum_edges > 0:
        return True
    else:
        return False

def cross_product(p1, p2):
    """
    :param p1: point_1 (x1, y1)
    :param p2: point_2 (x2, y2)
    :return: float number
    """
    return p1[0] * p2[1] - p2[0] * p1[1]


def direction(p1, p2, p3):
    """
    :param p1:
    :param p2:
    :param p3:
    :return: returns the cross product of vector p1p3 and p1p2
    If p1p3 is clockwise from p1p2 it returns positive value;
    if p1p3 is counter-clockwise from p1p2 it returns negative value
    if p1 p2 and p3 are collinear it returns 0
    """
    p1p3 = (p3[0] - p1[0], p3[1] - p1[1])
    p1p2 = (p2[0] - p1[0], p2[1] - p1[1])

    return cross_product(p1p3, p1p2)


def on_segment(p1, p2, p):
    """
    :param p1:
    :param p2:
    :param p:
    :return: return if p on line segment p1p2
    """
    return min(p1[0], p2[0]) <= p[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= p[1] <= max(p1[1], p2[1])


def intersect(p1, p2, p3, p4):
    """
    :param p1:
    :param p2:
    :param p3:
    :param p4:
    :return: check if ling segment p1p2 intersects p3p4
    """
    d1 = direction(p3, p4, p1)
    d2 = direction(p3, p4, p2)
    d3 = direction(p1, p2, p3)
    d4 = direction(p1, p2, p4)

    if d1 * d2 < 0 and d3 * d4 < 0:
        return True
    elif d1 == 0 and on_segment(p3, p4, p1):
        return True
    elif d2 == 0 and on_segment(p3, p4, p2):
        return True
    elif d3 == 0 and on_segment(p1, p2, p3):
        return True
    elif d4 == 0 and on_segment(p1, p2, p4):
        return True
    else:
        return False




