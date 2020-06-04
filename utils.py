from typing import List, Tuple

import torch
import numpy as np


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
    boxes1_area = ((boxes1[:, 0, 1] - boxes1[:, 2, 1]) * (boxes1[:, 0, 0] - boxes1[:, 2, 0]))
    boxes1_area = boxes1_area.unsqueeze(1).expand_as(intersection)
    boxes2_area = ((boxes2[:, 0, 1] - boxes2[:, 2, 1]) * (boxes2[:, 0, 0] - boxes2[:, 2, 0]))
    boxes2_area = boxes2_area.unsqueeze(0).expand_as(intersection)

    # Calculate union and return iou
    union = boxes1_area + boxes2_area - intersection
    return intersection / union


class Bbox_getter:
    """
    Extracts bounding boxes from model prediction.
    :param threshold: float, threshold probablity of classification (<=threshold - no bbox, >threshold - bbox)
    :param scaling_factor: int, 2**(num_maxpool) scaling factor to map output from model prediction to original "image"
    :param voxels_per_meter: number of voxels per meter in the frames
    :param car_size: size of the car in meters
    """

    def __init__(self, threshold: float, scaling_factor: int = 1, voxels_per_meter: int = 5, car_size: int = 5):
        self.threshold = threshold
        self.scaling_factor = scaling_factor
        self.voxels_per_meter = voxels_per_meter
        self.car_size = car_size
        self.bbox_scaling = car_size * voxels_per_meter
        self.predefined_bboxes = [[1, 1], [1, 2], [2, 1], [1, 6], [6, 1], [2, 2]]

    def __call__(self, image: torch.Tensor) -> np.ndarray:
        """
        :param image: torch.Tensor feature map shape: (batch,img_width, img_length, depth), output of detector
        :return: np.ndarray of bboxes
        """

        bboxes = np.zeros([6, 0])
        batch, depth, w, l = image.shape
        image.permute(0, 2, 3, 1)
        image = image.reshape(batch, w, l, depth // 6, 6)
        classification_index = depth // 6 - 1

        # find bboxes indexes
        cars_indeces = (image[:, :, :, classification_index, :] > self.threshold).nonzero().numpy()

        # extract and scale parameters for all found bboxes
        for car_idx in cars_indeces:
            bbox_center_y = car_idx[1] * self.scaling_factor
            predicted_displacement_y = image[car_idx[0], car_idx[1], car_idx[2], car_idx[3], 0] \
                * self.predefined_bboxes[car_idx[3]][0]
            y = bbox_center_y - predicted_displacement_y

            bbox_center_x = car_idx[2] * self.scaling_factor
            predicted_displacement_x = image[car_idx[0], car_idx[1], car_idx[2], car_idx[3], 1] \
                * self.predefined_bboxes[car_idx[3]][1]
            x = bbox_center_x - predicted_displacement_x

            predicted_w = image[car_idx[0], car_idx[1], car_idx[2], car_idx[3], 2]
            w = torch.exp(predicted_w * self.bbox_scaling * self.predefined_bboxes[car_idx[3]][1])

            h = torch.exp(image[car_idx[0], car_idx[1], car_idx[2], car_idx[3], 3]
                          * self.bbox_scaling * self.predefined_bboxes[car_idx[3]][1])

            angle1 = image[car_idx[0], car_idx[1], car_idx[2], car_idx[3], 4]
            angle2 = image[car_idx[0], car_idx[1], car_idx[2], car_idx[3], 5]

            bbox = np.array([y, x, w, h, angle1, angle2]).reshape(6, 1)
            bboxes = np.append(bboxes, bbox, axis=1)

        return bboxes


