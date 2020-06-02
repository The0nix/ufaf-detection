import numpy as np


def bbox_to_coordinates(bbox: np.ndarray, rot: bool = True):
    """
    Get vertices from bounding box parametrized with it's center_y, center_x, width, length, sin(a) and
    cos(a).
    :param bbox: array of ground truth bounding box geometrical parameters (center_y, center_x,
    width, length, sin(a), cos(a))
    :param rot: perform rotation
    :return: list of 4 tuples of (x, y) coordinates: [left_top, right_top, right_bottom, left_bottom]
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
