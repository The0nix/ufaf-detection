from typing import List, Tuple

import numpy as np


def bbox_to_coordinates(bbox: np.ndarray, rot: bool = True):
    """
    Get vertices from bounding box parametrized with it's center_y, center_x, width, length, sin(a) and
    cos(a).
    :param bbox: array of ground truth bounding box geometrical parameters (center_y, center_x,
    width, length, sin(a), cos(a))
    :param rot: perform rotation
    :return: list of 4 tuples of (y, x) coordinates: [left_top, right_top, right_bottom, left_bottom]
    """
    y, x = np.asarray(bbox[:2])
    width, length = bbox[2:4]  # converts borders calculus to center

    left_top = [y - width / 2, x - length / 2]
    right_top = [y - width / 2, x + length / 2]
    right_bottom = [y + width / 2, x + length / 2]
    left_bottom = [y + width / 2, x - length / 2]
    vertices = [left_top, right_top, right_bottom, left_bottom]
    if rot:
        sin, cos = bbox[4:]
        rotation = np.asarray([[-sin, cos], [cos, sin]])
        vertices_centered = [np.asarray([vertex[0] - y, vertex[1] - x]) for vertex in vertices]
        vertices_centered_rotated = [tuple(rotation @ vertex.reshape(-1, 1))
                                     for vertex in vertices_centered]
        vertices = [(vertex[0] + y, vertex[1] + x) for vertex in vertices_centered_rotated]
        vertices = np.array(vertices).squeeze(2)

    vertices = np.asarray(vertices)
    return vertices


def calc_iou(box1: List[Tuple[int, int]], box2: List[Tuple[int, int]]) -> float:
    """
    Calculate IoU of two boxes presented by their corners' coordinates
    :param box1: list of 4 tuples of (x, y) coordinates: [left_top, right_top, right_bottom, left_bottom]
    :param box2: list of 4 tuples of (x, y) coordinates: [left_top, right_top, right_bottom, left_bottom]
    :return: intersection over union score
    """

    left_top_x = max(box1[0][1], box2[0][1])  # left top angle of the intersection
    left_top_y = max(box1[0][0], box2[0][0])  #
    right_bottom_x = min(box1[2][1], box2[2][1])  # right bottom angle of the intersection
    right_bottom_y = min(box1[2][0], box2[2][0])  #

    box1_area = (box1[0][1] - box1[2][1]) * (box1[0][0] - box1[2][0])  # width multiplied by height
    box2_area = (box2[0][1] - box2[2][1]) * (box2[0][0] - box2[2][0])  #
    intersection = (
        (right_bottom_x - left_top_x) * (right_bottom_y - left_top_y)
        if (right_bottom_y > left_top_y and right_bottom_x > left_top_x)
        else 0
    )
    union = box1_area + box2_area - intersection
    return intersection / union
