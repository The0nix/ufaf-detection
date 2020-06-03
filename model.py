import cProfile
from collections import defaultdict
from math import sqrt
from time import time
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

import utils


class EarlyFusion(nn.Module):
    """
    Early fusion feature extraction model from Fast & Furious paper. Extracts information from several lidar frames.
    :param img_depth: int, discretized height of the image from BEV
    :param n_base_channels: int, number of channels in the first convolution layer of VGG16
    :param n_time_steps: int, number of frames to be processed
    """
    def __init__(self, img_depth, n_base_channels=32, n_time_steps=5):
        super().__init__()
        self.n_base_channels = n_base_channels
        self.out_channels = n_base_channels * 8
        self.conv1d = nn.Conv1d(n_time_steps, 1, kernel_size=1, bias=False)
        self.vgg_modules = nn.Sequential(
            nn.Conv2d(img_depth, n_base_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Dropout(.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_base_channels, n_base_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Dropout(.2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(n_base_channels, n_base_channels * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Dropout(.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_base_channels * 2, n_base_channels * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Dropout(.2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(n_base_channels * 2, n_base_channels * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Dropout(.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_base_channels * 4, n_base_channels * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Dropout(.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_base_channels * 4, n_base_channels * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Dropout(.2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(n_base_channels * 4, n_base_channels * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Dropout(.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_base_channels * 8, n_base_channels * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Dropout(.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_base_channels * 8, n_base_channels * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Dropout(.2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        :param frames: set of frames for several time steps (default 5),
        expected shape is (batch_size, time_steps, img_depth, img_width, img_length)
        :return: 4D torch.Tensor feature map
        """
        batch_size, n_time_steps, *pic_size = frames.shape
        frames = torch.reshape(frames, (batch_size, n_time_steps, pic_size[0] * pic_size[1] * pic_size[2]))
        fused = self.conv1d(frames)
        fused = torch.reshape(fused, (batch_size, *pic_size))
        vgg_out = self.vgg_modules(fused)
        return vgg_out


class Detector(nn.Module):
    """
    Predicts 2D bounding boxes for cars objects in provided frames.
    :param img_depth: int, discretized height of the image from BEV
    :param n_time_steps: int, number of frames to be processed
    :param n_predefined_boxes: int, number of bounding boxed corresponding to each cell of the feature map

    First `6 * n_predefined_boxes` depth levels of final_conv output correspond to BB location parameters
    on the original img, last `n_predefined_boxes` depth levels correspond to classification probabilities of
    BB containing a vehicle
    """
    def __init__(self, img_depth: int, n_time_steps: int = 1, n_predefined_boxes: int = 6) -> None:
        super().__init__()
        self.feature_extractor = EarlyFusion(img_depth, n_time_steps=n_time_steps)

        self.final_conv = nn.Conv2d(self.feature_extractor.out_channels, 7 * n_predefined_boxes,
                                    kernel_size=3, padding=1)
        # TODO: extract automatically from architecture
        self.n_pools = 4
        self.out_channels = 7 * n_predefined_boxes

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        :param frames: set of frames for several time steps (default 5),
        expected shape is (batch_size, time_steps, img_depth, img_length, img_width)
        :return: 4D torch.Tensor, predictions for bounding boxes positions and probabilities of a vehicle being
        in the bounding box
        """
        feature_map = self.feature_extractor(frames)
        output = self.final_conv(feature_map)
        return output


class GroundTruthFormer:
    """
    Forms tensor of ground truth data to calculate loss.
    :param gt_frame_size: tuple of the length and width of the frame (expected to be equal among all frames)
    :param detector_output_size: Tuple(int, int, int, int), shape of the detector's output
    :param voxels_per_meter: number of voxels per meter in the frames
    :param car_size: size of the car in meters
    :param n_pools: number of pooling layers in the feature extractor
    :param iou_threshold: threshold above which box is considered match to ground truth
    :param n_bbox_params: number of regression numbers for each bounding box
    """
    def __init__(self, gt_frame_size: Tuple[int, int], detector_output_size: Tuple[int, int, int, int],
                 voxels_per_meter: int = 5, car_size: int = 5, n_pools: int = 4, iou_threshold: int = 0.4,
                 n_bbox_params: int = 6) -> None:
        self.gt_frame_width, self.gt_frame_length = gt_frame_size
        self.batch_size = detector_output_size[0]
        self.detector_out_depth, self.detector_out_width, self.detector_out_length = detector_output_size[1:]
        self.n_pools = n_pools
        self.iou_threshold = iou_threshold
        self.n_bbox_params = n_bbox_params
        self.bbox_scaling = car_size * voxels_per_meter
        predefined_bboxes = [[1, 1], [1, 2], [2, 1], [1, 6], [6, 1], [2, 2]]
        self.predefined_bboxes = [[dim for dim in box] for box in predefined_bboxes]

    def __call__(self, gt_bboxes: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        :param gt_bboxes: list of lists of ground truth bounding boxes parameters, which are torch.Tensors of 6 numbers:
        center coordinates, length, width, sin(a) and cos(a)
        :return: ground truth data
        """
        return self.form_gt(gt_bboxes)

    def form_gt(self, gt_bboxes: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Builds 4D torch.Tensor with a shape of the detector output for the batch of frames.
        :param gt_bboxes: list of lists of ground truth bounding boxes parameters, which are torch.Tensors of 6 numbers:
        center coordinates, length, width, sin(a) and cos(a)
        :return: 4D torch.Tensor of ground truth data
        """
        gt_result = torch.zeros(self.batch_size, self.detector_out_depth, self.detector_out_width,
                                self.detector_out_length)
        for n in range(self.batch_size):
            gt_with_candidate_matches = defaultdict(list)
            used_boxes = set()
            for gt_box in gt_bboxes[n]:
                current_max_iou, current_max_box = 0, None
                # TODO: speed up using set of (i, j, k) boxes
                for i in range(self.detector_out_width):
                    for j in range(self.detector_out_length):
                        for k, candidate_box in enumerate(self.predefined_bboxes):
                            if (i, j, k) in used_boxes:
                                continue
                            candidate_box_parametrized = self._project_predefined_bbox_to_img([i, j, *candidate_box])
                            iou = utils.calc_iou(utils.bbox_to_coordinates(gt_box.numpy(), rot=False),
                                                 utils.bbox_to_coordinates(candidate_box_parametrized, rot=False))
                            if iou > self.iou_threshold:
                                used_boxes.add((i, j, k))
                                gt_with_candidate_matches[gt_box].append((i, j, k))
                                gt_result[n, k * self.n_bbox_params:(k + 1) * self.n_bbox_params,
                                          i, j] = self._normalize_gt(gt_box, candidate_box_parametrized)  #add bbox coordinates
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
                    #current_max_box = (1, 1, 1)
                    gt_with_candidate_matches[gt_box] = list(current_max_box)
                    i, j, k = current_max_box
                    candidate_box_parametrized = self._project_predefined_bbox_to_img([i, j, *(self.predefined_bboxes[k])])
                    gt_result[n, k * 6:(k + 1) * 6, i, j] = self._normalize_gt(gt_box, candidate_box_parametrized)  # add bbox coordinates
                    gt_result[n, len(self.predefined_bboxes) * 6 + k, i, j] = 1  # assign true class label
        return gt_result

    def _normalize_gt(self, gt_box, candidate_box) -> torch.Tensor:
        """
        Get normalization of ground truth
        :param gt_box: ground truth box, torch.Tensor of 6 numbers: center coordinates, length, width, sin(a) and cos(a)
        :param candidate_box: predicted box, torch.Tensor of 6 numbers: center coordinates, length, width, sin(a) and cos(a)
        :return: normalized gt_box, torch.Tensor of 6 numbers
        """
        coords_ix, sizes_ix = 2, 4
        candidate_box = torch.as_tensor(candidate_box).float()
        gt_box[:coords_ix] = torch.div(candidate_box[:coords_ix] - gt_box[:coords_ix], gt_box[coords_ix:sizes_ix])
        gt_box[coords_ix:sizes_ix] = torch.log(torch.div(candidate_box[coords_ix:sizes_ix], gt_box[coords_ix:sizes_ix]))
        return gt_box

    def _project_predefined_bbox_to_img(self, params: List[int]) -> np.ndarray:
        """
        Retrieve projection of the candidate box to the original image
        :param params: list or tuple of cell indices, width and length of the predefined bbox
        :return: tensor of cell indices, width, length, sin(0) and cos(0) of the predefined bbox projection
        on the original image
        """
        y, x, width, length = params
        return np.asarray([y * 2 ** self.n_pools,      # find projection through pooling layers
                           x * 2 ** self.n_pools,
                           width * self.bbox_scaling,  # scale to real world cars size
                           length * self.bbox_scaling,
                           0, 1])                      # zero rotation


# sanity checks: model forward pass and ground truth forming
#batch_size, time_steps, depth, width, length = 8, 1, 20, 128, 128
#frames = torch.randn((batch_size, time_steps, depth, width, length)) #.cuda()
#gt_bboxes = [[torch.randn(6) for j in range(20)] for i in range(batch_size)]
#print('gt_bboxes', gt_bboxes[0][0].size(), len(gt_bboxes), len(gt_bboxes[0]))
#
#net = Detector(depth) #.cuda()
#begin = time()
#model_out = net(frames)
#end = time()
#print(model_out.shape, f'Detector forward pass time taken: {(end - begin):.4f} seconds', sep='\n')
#
#cProfile.run("GroundTruthFormer((128, 128), model_out.shape)(gt_bboxes)")
#
#print('model_out', model_out.size())
#gt_former = GroundTruthFormer((128, 128), model_out.shape)
#begin = time()
#gt = gt_former(gt_bboxes)
#end = time()
#print(gt.shape, f'Ground truth former time taken: {(end - begin):.2f} seconds', sep='\n', end='\n\n')
#
# # sanity check: rectangle area must not change after rotation
#gt_boxes = [torch.tensor([1, 2, 5, 5, 1, 0]),
#             torch.tensor([1, 2, 5, 5, 0, 1]),
#             torch.tensor([1, 2, 5, 5, sqrt(2) / 2, sqrt(2) / 2])]
#for gt_box in gt_boxes:
#    print(f'True area: {gt_box[2] * gt_box[3]}', end='')
#    print(f', area after rotation: {utils.bbox_to_coordinates(gt_box.numpy())}')
