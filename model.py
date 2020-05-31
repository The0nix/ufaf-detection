from collections import defaultdict
from time import time

import torch
from torch import nn

from shapely.geometry import Polygon


class EarlyFusion(nn.Module):
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

    def forward(self, frames):
        """
        :param frames: set of frames for several time steps (default 5),
        expected shape is (batch_size, time_steps, img_depth, img_length, img_width)
        :return: feature map
        """
        batch_size, n_time_steps, pic_size = frames.shape[0], frames.shape[1], frames.shape[2:]
        frames = torch.reshape(frames, (batch_size, n_time_steps, pic_size[0] * pic_size[1] * pic_size[2]))
        fused = self.conv1d(frames)
        fused = torch.reshape(fused, (batch_size, *pic_size))
        vgg_out = self.vgg_modules(fused)
        return vgg_out


class Detector(nn.Module):
    def __init__(self, img_depth, n_predefined_boxes=6):
        """
        :param img_depth: int, discretized height of the image from BEV
        :param n_predefined_boxes: int, number of bounding boxed corresponding to each cell of the feature map

        First `6 * n_predefined_boxes` depth levels of final_conv output correspond to BB location parameters
        on the original img, last `n_predefined_boxes` depth levels correspond to classification probabilities of
        BB containing a vehicle
        """
        super().__init__()
        self.feature_extractor = EarlyFusion(img_depth)

        self.final_conv = nn.Conv2d(self.feature_extractor.out_channels, 7 * n_predefined_boxes,
                                    kernel_size=3, padding=1)

    def forward(self, frames):
        """
        :param frames: set of frames for several time steps (default 5),
        expected shape is (batch_size, time_steps, img_depth, img_length, img_width)
        :return: predictions for bounding boxes positions and probabilities of a vehicle being in the bounding box
        """
        feature_map = self.feature_extractor(frames)
        output = self.final_conv(feature_map)

        return output


class GroundTruthFormer:
    def __init__(self, gt_frame_size: tuple, gt_bboxes: list, detector_output: torch.Tensor, n_pools=4) -> None:
        """
        :param gt_frame_size: tuple of the legth and width of the frame (expected to be equal among all frames)
        :param gt_bboxes: list of lists of ground truth bounding boxes parameters, which are torch.Tensors of 6 numbers:
        center coordinates, length, width, sin(a) and cos(a)
        :param detector_output: 4D tensor, output of the detector
        :param n_pools: number of pooling layers in the feature extractor
        """
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

    def form_gt(self) -> torch.Tensor:
        """
        Function builds 4D torch.Tensor with a shape of the detector output for the batch of frames.
        The built tensor will be then used as ground truth data to calculate loss for the model.
        :return: 4D tensor of ground truth data
        """
        iou_threshold = 0.4
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
                            iou = self.calc_iou_from_polygons(self._get_polygon_from_gt(gt_box),
                                                              self._get_polygon_from_candidate((i, j, *candidate_box),
                                                                                               self.n_pools))
                            if iou > iou_threshold and (i, j, k) not in used_boxes:
                                used_boxes.add((i, j, k))
                                gt_with_candidate_matches[gt_box].append((i, j, k))
                                gt_result[n, k * 6:(k + 1) * 6, i, j] = gt_box  # add bbox coordinates
                                gt_result[n, len(self.predefined_bboxes) * 6 + k, i, j] = 1  # assign true class label
                            elif iou < iou_threshold and (i, j, k) not in used_boxes:
                                if iou > current_max_iou:
                                    used_boxes.add((i, j, k))
                                    if current_max_box is not None:
                                        used_boxes.remove(current_max_box)
                                    current_max_iou, current_max_box = iou, (i, j, k)
                                else:
                                    continue
                            else:
                                continue
                if gt_box not in gt_with_candidate_matches:
                    # TODO: check that current_max_box is not None with real data
                    current_max_box = (1, 1, 1)
                    gt_with_candidate_matches[gt_box] = list(current_max_box)
                    i, j, k = current_max_box
                    gt_result[n, k * 6:(k + 1) * 6, i, j] = gt_box  # add bbox coordinates
                    gt_result[n, len(self.predefined_bboxes) * 6 + k, i, j] = 1  # assign true class label

        return gt_result

    @staticmethod
    def _transform_predef_bbox_to_img(params: (list, tuple), n_pools: int) -> list:
        """
        Retrieve projection of the candidate box to the original image
        :param params: tuple of cell indices, width and length of the predefined bbox
        :param n_pools: number of pooling layers in the feature extractor
        :return: tuple of of cell indices, width and length of the predefined bbox projection on the original image
        """
        return [elem * 2 ** n_pools for elem in params]

    @staticmethod
    def _get_polygon_from_gt(gt_box: (list, tuple)) -> Polygon:
        """
        Get Polygon object from ground truth bounding box parametrization. Center is considered to be a
        "right-bottom center".
        :param gt_box: list or tuple of ground truth bounding box geometrical parameters
        :return: Polygon object, initialized by vertices of the GT bbox polygon on original image scale
        """
        return Polygon([[1, 1], [0, 1], [0, 0], [1, 0]])

    @staticmethod
    def _get_polygon_from_candidate(candidate_box: (list, tuple), n_pools: int) -> Polygon:
        """
        Get Polygon object from feature map predefined box parametrization. Center is considered to be a
        "right-bottom center".
        :param candidate_box: list or tuple of predefined box center coordinates and size params `i, j, width, length`
        :param n_pools: number of pooling layers in the feature extractor
        :return: Polygon object, initialized by vertices of the box polygon on original image scale
        """
        candidate_box = GroundTruthFormer._transform_predef_bbox_to_img(candidate_box, n_pools)
        i, j, width, length = candidate_box
        left_top = [j - length // 2, i - width // 2]
        right_top = [j + length // 2 + length % 2 - 1, i - width // 2]
        right_bottom = [j + length // 2 + length % 2 - 1, i + width // 2 + width % 2 - 1]
        left_bottom = [j - length // 2, i + width // 2 + width % 2 - 1]
        a = Polygon([left_top, right_top, right_bottom, left_bottom]).area
        return Polygon([left_top, right_top, right_bottom, left_bottom])

    @staticmethod
    def calc_iou_from_polygons(gt_box: Polygon, candidate_box: Polygon) -> float:
        return gt_box.intersection(candidate_box).area / gt_box.union(candidate_box).area


# little work and shape testing snippet
# batch_size, time_steps, depth, width, length = 8, 5, 20, 128, 128
# frames = torch.randn((batch_size, time_steps, depth, width, length)).cuda()
# gt_bboxes = [[torch.randn(6).cuda() for j in range(20)] for i in range(batch_size)]
#
# net = Detector(depth).cuda()
# begin = time()
# model_out = net(frames)
# end = time()
# print(model_out.shape, f'Time taken: {(end - begin):.4f} seconds', sep='\n')
#
# gt_former = GroundTruthFormer((128, 128), gt_bboxes, model_out)
# begin = time()
# gt = gt_former()
# end = time()
# print(gt.shape, f'Time taken: {(end - begin):.2f} seconds', sep='\n')
