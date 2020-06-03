from typing import List, Tuple

import torch


def bbox_to_coordinates(bboxes: torch.Tensor, rot: bool = False) -> torch.Tensor:
    """
    Get vertices from bounding box parametrized with it's center_y, center_x, width, length, sin(a) and
    cos(a).
    :param bboxes: torch.Tensor of shape [n_boxes, 6] with
    ground truth bounding boxes' geometrical parameters [center_y, center_x,
    width, length, sin(a), cos(a)]
    :param rot: perform rotation
    :return: tensor of shape [n_boxes, 4, 2] of boxes of 4 corners of (y, x) coordinates:
    [left_top, right_top, right_bottom, left_bottom]
    """
    bboxes = bboxes.float()
    y, x = bboxes[:, :2].t()
    width, length = bboxes[:, 2:4].t()

    left_top = torch.stack([y - width / 2, x - length / 2])
    right_top = torch.stack([y - width / 2, x + length / 2])
    right_bottom = torch.stack([y + width / 2, x + length / 2])
    left_bottom = torch.stack([y + width / 2, x - length / 2])
    vertices = [left_top, right_top, right_bottom, left_bottom]
    if rot:
        raise NotImplementedError("Rotations are not yet supported")
        sin, cos = bboxes[:, 4:].t()
        rotation = torch.tensor([[-sin, cos], [cos, sin]])
        vertices_centered = [torch.tensor([vertex[0] - y, vertex[1] - x]) for vertex in vertices]
        vertices_centered_rotated = [tuple(rotation @ vertex.reshape(-1, 1))
                                     for vertex in vertices_centered]
        vertices = [(vertex[0] + y, vertex[1] + x) for vertex in vertices_centered_rotated]
        vertices = torch.tensor(vertices)
    vertices = torch.stack(vertices).permute(2, 0, 1)  # make n_boxes first
    return vertices


def calc_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU of two boxes presented by their corners' coordinates
    :param boxes1: torch.Tensor of size (n_boxes_1, 4, 2) of boxes' coordinates:
    [left_top, right_top, right_bottom, left_bottom]
    :param boxes2: torch.Tensor of size (n_boxes_2, 4, 2) of boxes' coordinates:
    [left_top, right_top, right_bottom, left_bottom]
    :return: torch.Tensor of size (n_boxes_1, n_boxes_2)  of intersection over union scores between each pair
    """
    size_1 = boxes1.shape[0]
    size_2 = boxes2.shape[0]

    # Calculate intersection corners for each pair
    left_top_x = torch.max(boxes1[:, 0, 1].unsqueeze(1).expand(size_1, size_2),  # left top angle of the intersection
                           boxes2[:, 0, 1].unsqueeze(0).expand(size_1, size_2))  #
    left_top_y = torch.max(boxes1[:, 0, 0].unsqueeze(1).expand(size_1, size_2),  #
                           boxes2[:, 0, 0].unsqueeze(0).expand(size_1, size_2))  #
    right_bottom_x = torch.min(boxes1[:, 2, 1].unsqueeze(1).expand(size_1, size_2),  # right bottom angle of the
                               boxes2[:, 2, 1].unsqueeze(0).expand(size_1, size_2))  # intersection
    right_bottom_y = torch.min(boxes1[:, 2, 0].unsqueeze(1).expand(size_1, size_2),  #
                               boxes2[:, 2, 0].unsqueeze(0).expand(size_1, size_2))  #

    # Calculate intersection from corners as width * height
    intersection = (
        (right_bottom_x - left_top_x) * (right_bottom_y - left_top_y)
        * ((right_bottom_y > left_top_y) & (right_bottom_x > left_top_x))
    )

    # Calculate areas as width * length and expand to matrix
    box1_area = ((boxes1[:, 0, 1] - boxes1[:, 2, 1]) * (boxes1[:, 0, 0] - boxes1[:, 2, 0]))
    box1_area = box1_area.unsqueeze(1).expand_as(intersection)
    box2_area = ((boxes2[:, 0, 1] - boxes2[:, 2, 1]) * (boxes2[:, 0, 0] - boxes2[:, 2, 0]))
    box2_area = box2_area.unsqueeze(0).expand_as(intersection)

    # Calculate union and return iou
    union = box1_area + box2_area - intersection
    return intersection / union
