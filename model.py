import cProfile
from collections import defaultdict
from math import sqrt
from time import time
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn
from shapely.geometry import Polygon


class EarlyFusion(nn.Module):
    """
    Early fusion feature extraction model from Fast & Furious paper. Extracts information from several lidar frames.
    """
    def __init__(self, img_depth, n_base_channels=32, n_time_steps=5):
        super().__init__()
        self.n_base_channels = n_base_channels
        self.out_channels = n_base_channels * 8
        self.conv1d = nn.Conv1d(n_time_steps, 1, kernel_size=1, bias=False)
        self.vgg_modules = nn.Sequential(
            nn.Conv2d(img_depth, n_base_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_base_channels, n_base_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(n_base_channels, n_base_channels * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_base_channels * 2, n_base_channels * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(n_base_channels * 2, n_base_channels * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_base_channels * 4, n_base_channels * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_base_channels * 4, n_base_channels * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(n_base_channels * 4, n_base_channels * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_base_channels * 8, n_base_channels * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_base_channels * 8, n_base_channels * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        :param frames: set of frames for several time steps (default 5),
        expected shape is (batch_size, time_steps, img_depth, img_length, img_width)
        :return: feature map
        """
        batch_size, n_time_steps, *pic_size = frames.shape
        frames = torch.reshape(frames, (batch_size, n_time_steps, pic_size[0] * pic_size[1] * pic_size[2]))
        fused = self.conv1d(frames)
        fused = torch.reshape(fused, (batch_size, *pic_size))
        vgg_out = self.vgg_modules(fused)
        return vgg_out


class Detector(nn.Module):
    """
    :param img_depth: int, discretized height of the image from BEV
    :param n_predefined_boxes: int, number of bounding boxed corresponding to each cell of the feature map

    First `6 * n_predefined_boxes` depth levels of final_conv output correspond to BB location parameters
    on the original img, last `n_predefined_boxes` depth levels correspond to classification probabilities of
    BB containing a vehicle
    """
    def __init__(self, img_depth, n_predefined_boxes=6):
        super().__init__()
        self.feature_extractor = EarlyFusion(img_depth)

        self.final_conv = nn.Conv2d(self.feature_extractor.out_channels, 7 * n_predefined_boxes,
                                    kernel_size=3, padding=1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        :param frames: set of frames for several time steps (default 5),
        expected shape is (batch_size, time_steps, img_depth, img_length, img_width)
        :return: predictions for bounding boxes positions and probabilities of a vehicle being in the bounding box
        """
        feature_map = self.feature_extractor(frames)
        output = self.final_conv(feature_map)

        return output


class GroundTruthFormer:
    """
    :param gt_frame_size: tuple of the length and width of the frame (expected to be equal among all frames)
    :param gt_bboxes: list of lists of ground truth bounding boxes parameters, which are torch.Tensors of 6 numbers:
    center coordinates, length, width, sin(a) and cos(a)
    :param detector_output: 4D tensor, output of the detector
    :param n_pools: number of pooling layers in the feature extractor
    """
    def __init__(self, gt_frame_size: Tuple[int, int], gt_bboxes: List[torch.Tensor], detector_output: torch.Tensor,
                 n_pools: int = 4) -> None:
        self.gt_frame_width, self.gt_frame_length = gt_frame_size
        self.gt_bboxes = gt_bboxes
        self.batch_size = detector_output.shape[0]
        self.detector_out_width, self.detector_out_length = detector_output.shape[-2:]
        self.detector_output = detector_output
        self.n_pools = n_pools
        bbox_scaling = 1  # TODO: get from OnixinO
        predefined_bboxes = [[1, 1], [1, 2], [2, 1], [1, 6], [6, 1], [2, 2]]
        self.predefined_bboxes = [[bbox_scaling * dim for dim in box] for box in predefined_bboxes]

    def __call__(self):
        return self.form_gt()

    def form_gt(self, iou_threshold: int = 0.4) -> torch.Tensor:
        """
        Function builds 4D torch.Tensor with a shape of the detector output for the batch of frames.
        The built tensor will be then used as ground truth data to calculate loss for the model.
        :return: 4D tensor of ground truth data
        """
        gt_result = torch.zeros_like(self.detector_output)
        for n in range(self.batch_size):
            gt_with_candidate_matches = defaultdict(list)
            used_boxes = set()
            for gt_box in self.gt_bboxes[n]:
                current_max_iou, current_max_box = 0, None
                # TODO: speed up using set of (i, j, k) boxes
                for i in range(self.detector_out_width):
                    for j in range(self.detector_out_length):
                        for k, candidate_box in enumerate(self.predefined_bboxes):
                            if (i, j, k) in used_boxes:
                                continue
                            candidate_box_parametrized = self._project_predefined_bbox_to_img([i, j, *candidate_box])
                            iou = self.calc_iou_from_polygons(self._get_polygon(gt_box.numpy()),
                                                              self._get_polygon(candidate_box_parametrized, rot=False))
                            if iou > iou_threshold:
                                used_boxes.add((i, j, k))
                                gt_with_candidate_matches[gt_box].append((i, j, k))
                                gt_result[n, k * 6:(k + 1) * 6, i, j] = gt_box  # add bbox coordinates
                                gt_result[n, len(self.predefined_bboxes) * 6 + k, i, j] = 1  # assign true class label
                            else:
                                if iou > current_max_iou:
                                    used_boxes.add((i, j, k))
                                    if current_max_box is not None:
                                        used_boxes.remove(current_max_box)
                                    current_max_iou, current_max_box = iou, (i, j, k)
                if gt_box not in gt_with_candidate_matches:
                    # TODO: check that current_max_box is not None with real data
                    # next line handles current_max_box being None while testing, remove it while applying to real data
                    current_max_box = (1, 1, 1)
                    gt_with_candidate_matches[gt_box] = list(current_max_box)
                    i, j, k = current_max_box
                    gt_result[n, k * 6:(k + 1) * 6, i, j] = gt_box  # add bbox coordinates
                    gt_result[n, len(self.predefined_bboxes) * 6 + k, i, j] = 1  # assign true class label

        return gt_result

    def _project_predefined_bbox_to_img(self, params: List[int]) -> np.ndarray:
        """
        Retrieve projection of the candidate box to the original image
        :param params: list or tuple of cell indices, width and length of the predefined bbox
        :return: tensor of cell indices, width, length, sin(0) and cos(0) of the predefined bbox projection
        on the original image
        """
        return np.concatenate((np.array([elem * 2 ** self.n_pools for elem in params]),
                               np.array([0, 1])))

    @staticmethod
    def _get_polygon(parametrized_box: np.ndarray, rot: bool = True) -> Polygon:
        """
        Get Polygon object from bounding box parametrized with it's center_y, center_x, width, length, sin(a) and
        cos(a). Center is considered to be a "right-bottom center" (matters when image dimensions are even).
        :param parametrized_box: array of ground truth bounding box geometrical parameters (center_y, center_x,
        width, length, sin(a), cos(a))
        :return: Polygon object, initialized by vertices of the GT bbox polygon on original image scale
        """
        i, j = parametrized_box[:2]
        width, length = parametrized_box[2:4] + np.array([1, 1])  # converts borders calculus to center

        left_top = [j - length // 2, i - width // 2]
        right_top = [j + length // 2 + length % 2 - 1, i - width // 2]
        right_bottom = [j + length // 2 + length % 2 - 1, i + width // 2 + width % 2 - 1]
        left_bottom = [j - length // 2, i + width // 2 + width % 2 - 1]
        vertices = [left_top, right_top, right_bottom, left_bottom]
        if rot:
            sin, cos = parametrized_box[4:]
            rotation = np.array([[cos, -sin], [sin, cos]])
            return Polygon([tuple(rotation @ np.array(vertex).reshape(-1, 1)) for vertex in vertices])
        return Polygon(vertices)

    @staticmethod
    def calc_iou_from_polygons(gt_box: Polygon, candidate_box: Polygon) -> float:
        return gt_box.intersection(candidate_box).area / gt_box.union(candidate_box).area


# sanity checks: model forward pass and
# batch_size, time_steps, depth, width, length = 8, 5, 20, 128, 128
# frames = torch.randn((batch_size, time_steps, depth, width, length)).cuda()
# gt_bboxes = [[torch.randn(6) for j in range(20)] for i in range(batch_size)]
#
# net = Detector(depth).cuda()
# begin = time()
# model_out = net(frames)
# end = time()
# print(model_out.shape, f'Detector forward pass time taken: {(end - begin):.4f} seconds', sep='\n')
#
# cProfile.run("GroundTruthFormer((128, 128), gt_bboxes, model_out)()")
#
# gt_former = GroundTruthFormer((128, 128), gt_bboxes, model_out)
# begin = time()
# gt = gt_former()
# end = time()
# print(gt.shape, f'Ground truth former time taken: {(end - begin):.2f} seconds', sep='\n', end='\n\n')
#
# # sanity check: rectangle area must not change after rotation
# gt_boxes = [torch.tensor([1, 2, 5, 5, 1, 0]),
#             torch.tensor([1, 2, 5, 5, 0, 1]),
#             torch.tensor([1, 2, 5, 5, sqrt(2) / 2, sqrt(2) / 2])]
# for gt_box in gt_boxes:
#     print(f'True area: {gt_box[2] * gt_box[3]}', end='')
#     print(f', area after rotation: {GroundTruthFormer._get_polygon(gt_box).area}')
