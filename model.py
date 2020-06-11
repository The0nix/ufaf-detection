import cProfile
from collections import defaultdict
from math import sqrt
from time import time
from typing import List, Tuple, Optional, Union

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
    def __init__(self, img_depth, n_base_channels=32, n_time_steps=5) -> None:
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
    :param n_bbox_params: int, number of parameters in bbox. Default is 6: [y, x, width, length, sin(a), cos(a)]
    """
    def __init__(self, img_depth: int, n_time_steps: int = 1,
                 n_predefined_boxes: int = 6, n_bbox_params: int = 6) -> None:
        super().__init__()
        self.feature_extractor = EarlyFusion(img_depth, n_time_steps=n_time_steps)
        self.n_predefined_boxes = n_predefined_boxes
        self.n_bbox_params = n_bbox_params

        self.final_conv = nn.Conv2d(self.feature_extractor.out_channels, (n_bbox_params + 1) * n_predefined_boxes,
                                    kernel_size=3, padding=1)
        # TODO: extract automatically from architecture
        self.n_pools = 4
        self.out_channels = (n_bbox_params + 1) * n_predefined_boxes

    def forward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param frames: set of frames for several time steps (default 5),
        expected shape is [batch_size, time_steps, img_depth, img_width, img_length]
        :return: Tuple of two 4D torch.Tensor of predicted data:
        First tensor is of shape [batch_size, detector_out_width, detector_out_length, n_predefined_boxes]
        and represents classification target with ones for boxes with associated GT boxes
        Second tensor is of shape
        [batch_size, detector_out_width, detector_out_length, n_predefined_boxes, n_bbox_params]
        and represents regression target
        """
        feature_map = self.feature_extractor(frames)
        output = self.final_conv(feature_map)

        output = output.permute(0, 2, 3, 1)  # move channels dimension to end
        classification_output = output[:, :, :, :self.n_predefined_boxes]
        regression_output = output[:, :, :, self.n_predefined_boxes:]
        regression_output = regression_output.reshape(
            regression_output.shape[:-1] + (self.n_predefined_boxes, self.n_bbox_params)
        )

        return classification_output, regression_output


class DetectionLoss(nn.modules.loss._Loss):
    """
    Combination of losses for both regression and classification targets
    :param prediction_units_per_cell: number of predefined bounding boxes per feature map cell
    :param regression_values_per_unit: number of regression values per bounding box
    :param classification_values_per_unit: number of classes for classification problem
    :param regression_base_loss: loss function to be used for regression targets
    :param classification_base_loss: loss function to be used for classification targets
    :param negative_positive_ratio: ratio of negative samples to positive samples for hard negative mining
    """
    def __init__(self, prediction_units_per_cell: int = 6, regression_values_per_unit: int = 6,
                 classification_values_per_unit: int = 1, negative_positive_ratio: int = 3,
                 regression_base_loss: Optional[nn.modules.loss._Loss] = None,
                 classification_base_loss: Optional[nn.modules.loss._Loss] = None) -> None:
        super().__init__()
        self.prediction_units_per_cell = prediction_units_per_cell
        self.regression_values_per_unit = regression_values_per_unit
        self.classification_values_per_unit = classification_values_per_unit
        self.negative_positive_ratio = negative_positive_ratio
        self.regression_base_loss = regression_base_loss or nn.SmoothL1Loss()
        self.classification_base_loss = classification_base_loss or nn.BCEWithLogitsLoss()

    # noinspection PyUnresolvedReferences
    def __call__(self, predictions: Tuple[torch.Tensor, torch.Tensor],
                 ground_truth: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Compute loss
        :param predictions: model output
        :param ground_truth: ground truth data
        Both are tuples of two 4D torch.Tensor of data:
        First tensor is of shape [batch_size, detector_out_width, detector_out_length, n_predefined_boxes]
        and represents classification target with ones for boxes with associated GT boxes
        Second tensor is of shape
        [batch_size, detector_out_width, detector_out_length, n_predefined_boxes, n_bbox_params]
        and represents regression target
        :return: one element torch.Tensor loss
        """
        pred_classification, pred_regression = predictions
        gt_classification, gt_regression = ground_truth
        pred_regression *= gt_classification.unsqueeze(-1)
        gt_regression *= gt_classification.unsqueeze(-1)  # may be redundant

        # perform hard negative mining
        n_positive = gt_classification.sum()
        n_values_to_eliminate = int((torch.numel(gt_classification) - (self.negative_positive_ratio + 1) * n_positive))

        # n_values_to_eliminate can be less than zero when classes on the provided frames are already well balanced:
        if n_values_to_eliminate > 0:
            negative_probs = pred_classification * (1 - gt_classification)
            negative_probs = negative_probs.flatten()
            negative_probs[torch.topk(negative_probs, k=n_values_to_eliminate,
                                      largest=False).indices] = -1e9  # filter low negative probabilities
            negative_probs = negative_probs.view_as(pred_classification)
            pred_classification *= gt_classification  # leave only positive predictions
            pred_classification += negative_probs     # add mined negative predictions
        return self.regression_base_loss(pred_regression, gt_regression) + \
            self.classification_base_loss(pred_classification, gt_classification)


class GroundTruthFormer:
    """
    Forms tensor of ground truth data to calculate loss.
    :param gt_frame_size: tuple of the length and width of the frame (expected to be equal among all frames)
    :param detector_output_size: Tuple(int, int, int, int), shape of the detector's output
    :param voxels_per_meter: number of voxels per meter in the frames
    :param car_size: size of the car in meters
    :param n_pools: number of pooling layers in the feature extractor
    :param iou_threshold: threshold above which box is considered match to ground truth
    :param n_bbox_params: number of parameters in bbox. Default is 6: [y, x, width, length, sin(a), cos(a)]
    :param device: torch device for initialization of tensors
    """
    def __init__(self, gt_frame_size: Tuple[int, int], detector_output_size: Tuple[int, int, int, int],
                 voxels_per_meter: int = 5, car_size: int = 5, n_pools: int = 4, iou_threshold: int = 0.4,
                 n_bbox_params: int = 6, device: Union[torch.device, str] = torch.device('cpu')) -> None:
        self.device = device
        self.gt_frame_width, self.gt_frame_length = gt_frame_size
        _, self.detector_out_width, self.detector_out_length = detector_output_size[1:]
        self.n_pools = n_pools
        self.iou_threshold = iou_threshold
        self.n_bbox_params = n_bbox_params
        self.bbox_scaling = car_size * voxels_per_meter
        self.prior_boxes_params = [[[] for j in range(self.detector_out_length)] for i in range(self.detector_out_width)]
        self.prior_boxes_coords = [[[] for j in range(self.detector_out_length)] for i in range(self.detector_out_width)]
        self.predefined_bboxes = [[1, 1], [1, 2], [2, 1], [1, 6], [6, 1], [2, 2]]
        for i in range(self.detector_out_width):
            for j in range(self.detector_out_length):
                for w, l in self.predefined_bboxes:
                    parametrized_box = self._project_predefined_bbox_to_img((i, j, w, l))
                    coordinates = utils.bbox_to_coordinates(parametrized_box.unsqueeze(0), rot=False)[0]
                    self.prior_boxes_params[i][j].append(parametrized_box)
                    self.prior_boxes_coords[i][j].append(coordinates)
        self.prior_boxes_params = torch.stack(
            [torch.stack([torch.stack(box) for box in boxes]) for boxes in self.prior_boxes_params]
        ).to(self.device)
        self.prior_boxes_coords = torch.stack(
            [torch.stack([torch.stack(box) for box in boxes]) for boxes in self.prior_boxes_coords]
        ).to(self.device)

    def __call__(self, gt_bboxes: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param gt_bboxes: list of torch.Tensors of shape [n_bboxes, n_bbox_params]
        of ground truth bounding boxes parameters which are center coordinates, length, width, sin(a) and cos(a)
        :return: ground truth data
        """
        return self.form_gt(gt_bboxes)

    def form_gt(self, gt_bboxes: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Builds 4D torch.Tensor with a shape of the detector output for the batch of frames.
        :param gt_bboxes: list of torch.Tensors of shape [n_bboxes, n_bbox_params]
        of ground truth bounding boxes parameters which are center coordinates, length, width, sin(a) and cos(a)
        :return: Tuple of two 4D torch.Tensor of ground truth data:
        First tensor is of shape [batch_size, detector_out_width, detector_out_length, n_predefined_boxes]
        and represents classification target with ones for boxes with associated GT boxes
        Second tensor is of shape
        [batch_size, detector_out_width, detector_out_length, n_predefined_boxes, n_bbox_params]
        and represents regression target
        """
        gt_bboxes_coords = [utils.bbox_to_coordinates(gt_bbox, rot=False).to(self.device) for gt_bbox in gt_bboxes]
        classification_target = []
        regression_target = []
        prior_boxes_params = self.prior_boxes_params.view(-1, self.n_bbox_params)
        prior_boxes_coords = self.prior_boxes_coords.view(-1, 4, 2)  # each box is 4 points of 2 coordinates)
        for gt_bbox, gt_bbox_coords in zip(gt_bboxes, gt_bboxes_coords):
            iou = utils.calc_iou(gt_bbox_coords, prior_boxes_coords)  # tensor of size [n_gt_boxes, n_prior_boxes]
            gt_has_match = (iou >= self.iou_threshold).sum(dim=1).bool()
            prior_has_match = (iou >= self.iou_threshold).sum(dim=0).bool()
            gt_best_match = iou.argmax(dim=1)
            prior_match = iou.argmax(dim=0)
            prior_match[gt_best_match[~gt_has_match]] = (~gt_has_match).nonzero().flatten()
            prior_has_match[gt_best_match[~gt_has_match]] = True
            prior_has_match = prior_has_match.view(
                self.detector_out_width, self.detector_out_length, len(self.predefined_bboxes)
            ).float()

            cur_regression_target = torch.zeros(self.detector_out_width * self.detector_out_length *
                                                len(self.predefined_bboxes), self.n_bbox_params, device=self.device)

            # Calculate regression target: y and x offset, width and length correction, sin and cos of angle
            cur_regression_target[:, 0] = \
                (prior_boxes_params[:, 0] - gt_bbox[prior_match, 0]) / gt_bbox[prior_match, 2]
            cur_regression_target[:, 1] = \
                (prior_boxes_params[:, 1] - gt_bbox[prior_match, 1]) / gt_bbox[prior_match, 3]
            cur_regression_target[:, 2] = torch.log(prior_boxes_params[:, 2] / gt_bbox[prior_match, 2])
            cur_regression_target[:, 3] = torch.log(prior_boxes_params[:, 3] / gt_bbox[prior_match, 3])
            cur_regression_target[:, 4] = gt_bbox[prior_match, 4]
            cur_regression_target[:, 5] = gt_bbox[prior_match, 5]

            cur_regression_target = \
                cur_regression_target.view(self.detector_out_width, self.detector_out_length,
                                           len(self.predefined_bboxes), self.n_bbox_params)
            cur_regression_target *= prior_has_match.unsqueeze(3)

            regression_target.append(cur_regression_target)
            classification_target.append(prior_has_match)
        return torch.stack(classification_target), torch.stack(regression_target)

    def _project_predefined_bbox_to_img(self, params: Tuple[int, int, int, int]) -> torch.Tensor:
        """
        Retrieve projection of the candidate box to the original image
        :param params: tuple of cell indices, width and length of the predefined bbox
        :return: tensor of shape [n_bbox_params] of cell indices, width, length, sin(0) and cos(0)
        of the predefined bbox projection on the original image
        """
        y, x, width, length = params
        return torch.tensor([
            y * 2 ** self.n_pools,      # find projection through pooling layers
            x * 2 ** self.n_pools,
            width * self.bbox_scaling,  # scale to real world cars size
            length * self.bbox_scaling,
            0, 1                        # zero rotation
        ])


# sanity checks: model forward pass and ground truth forming
# batch_size, time_steps, depth, width, length = 8, 1, 20, 128, 128
# frames = torch.randn((batch_size, time_steps, depth, width, length)).cuda()
# gt_bboxes = torch.stack([torch.stack([torch.randn(6) * 10 + 20 for j in range(20)]) for i in range(batch_size)]).cuda()
#
# net = Detector(depth).cuda()
# begin = time()
# model_out_class, model_out_reg = net(frames)
# end = time()
# print(model_out_class.shape, f'Detector forward pass time taken: {(end - begin):.4f} seconds', sep='\n')
#
# gt_former = GroundTruthFormer((128, 128), model_out_class.shape, device="cuda")
#
# cProfile.run("gt_former(gt_bboxes)")
#
# gt_former = GroundTruthFormer((128, 128), model_out.shape)
# begin = time()
# gt = gt_former(gt_bboxes)
# end = time()
# print(gt.shape, f'Ground truth former time taken: {(end - begin):.2f} seconds', sep='\n', end='\n\n')
#
# # sanity check: rectangle area must not change after rotation
# gt_boxes = [torch.tensor([1, 2, 5, 5, 1, 0]),
#             torch.tensor([1, 2, 5, 5, 0, 1]),
#             torch.tensor([1, 2, 5, 5, sqrt(2) / 2, sqrt(2) / 2])]
# for gt_box in gt_boxes:
#    print(f'True area: {gt_box[2] * gt_box[3]}', end='')
#    print(f', area after rotation: {utils.bbox_to_coordinates(gt_box.numpy())}')
