import math
import os
from typing import Tuple

import matplotlib.pyplot as plt
import torch

from dataset import create_nuscenes, NuscenesBEVDataset
from model import Detector, GroundTruthFormer
from visualization import draw_bev_with_bboxes

torch.random.manual_seed(2)


class McProcessor:
    """
    Forms Monte Carlo based uncertanties and visualizes them
    :param data_path: relative path to data folder
    :param nuscenes_version: version of the dataset
    :param n_scenes: number of scenes in dataset
    :param threshold: threshold for choosing is bbox or not
    :return: Tuple[torch.tensor, np.ndarray] - first  - grid tensor, second - gt_bboxes
    """
    def __init__(self, data_path: str, n_scenes: int = 10, nuscenes_version: str = 'v1.0-mini',
                 threshold: int = 0.5, model_path: str = None) -> None:
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('Using device: GPU\n')
        else:
            self.device = torch.device('cpu')
            print('Using device: CPU\n')

        # init dataset
        self.version = nuscenes_version
        self.n_scenes = n_scenes
        self.nuscenes = create_nuscenes(data_path, nuscenes_version)
        self.dataset = NuscenesBEVDataset(nuscenes=self.nuscenes, n_scenes=n_scenes)

        # init model
        frame_depth, _, _ = self.dataset.grid_size
        self.model = Detector(img_depth=frame_depth)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.train()  # keeps dropouts active

        self.threshold = threshold

    def visualise_monte_carlo(self, frame_id: int = 0, n_samples: int = 10, batch_size: int = 4,
                              save_imgs: bool = False, saving_folder: str = "pics") -> None:
        """
        Visualize predictions obtained via Monte Carlo estimations and save plots if needed
        :param frame_id: number of frame (grid) in the dataset
        :param n_samples: number of samples for Monte Carlo approach
        :param batch_size: size of batch
        :param save_imgs: - flag, if true - save figs to folder pics
        :param saving_folder: - path to the folder, where images will be saved (creates new if there is none)
        """

        mean_class, _, mean_reg, sigma_reg = self.apply_monte_carlo(frame_id, n_samples, batch_size)
        fig, ax_gt, ax_pred = self._vis_mc(mean_class, mean_reg, sigma_reg, frame_id)

        if save_imgs:
            if not os.path.exists(saving_folder):
                os.makedirs(saving_folder)
            img_path = f'{saving_folder}/{self.version}_{self.n_scenes}_{frame_id}_full.png'
            fig.savefig(img_path)

            img_path = f'{saving_folder}/{self.version}_{self.n_scenes}_{frame_id}_gt.png'
            extent = ax_gt.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(img_path, bbox_inches=extent)

            img_path = f'{saving_folder}/{self.version}_{self.n_scenes}_{frame_id}_pred.png'
            extent_ped = ax_pred.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(img_path, bbox_inches=extent_ped)

        plt.show()

    def apply_monte_carlo(self, frame_id: int = 0, n_samples: int = 10, batch_size: int = 4) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply Monte Carlo dropout for representing model uncertainty
        :param frame_id: id of frame (grid) in the dataset
        :param n_samples: number of samples for Monte Carlo approach
        :param batch_size: batch size
        :return: tuple of four torch.Tensors:
            1st - mean values of class model prediction
            2nd - standard deviations of class model prediction
            3rd - mean values of regression model prediction
            4th - standard deviations of regression model prediction
        """
        assert n_samples > 1, "Need minimum 2 samples to calculate unbiased variance"

        grid, boxes = self.dataset[frame_id]
        class_output, reg_output = self.model(grid[None].to(self.device))
        samples_class, samples_reg = class_output, reg_output

        # TODO: append to single batch for speeding up
        for i in range(math.ceil((n_samples - 1) / batch_size)):
            current_batch_size = min((n_samples - 1) - (i * batch_size), batch_size)
            stacked_grid = torch.stack(current_batch_size * [grid])
            class_output, reg_output = self.model(stacked_grid.to(self.device))
            samples_class = torch.cat((samples_class, class_output))
            samples_reg = torch.cat((reg_output, samples_reg))

        # calculate stats from samples set
        samples_class, samples_reg = samples_class.detach(), samples_reg.detach()
        mean_reg = torch.mean(samples_reg, dim=0).unsqueeze(0)
        variance_reg = torch.var(samples_reg, dim=0)
        sigma_reg = torch.sqrt(variance_reg).unsqueeze(0)
        mean_class = torch.mean(samples_class, dim=0).unsqueeze(0)
        variance_class = torch.var(samples_class, dim=0)
        sigma_class = torch.sqrt(variance_class).unsqueeze(0)

        return mean_class, sigma_class, mean_reg, sigma_reg

    def _vis_mc(self, mean_class: torch.Tensor, mean_regr: torch.Tensor,
                sigma_regr: torch.Tensor, frame_id: int = 0) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
        """
        Visualize predictions obtained via Monte Carlo estimations
        :param mean_class: mean of classification predictions
        :param mean_regr: mean of regression predictions
        :param sigma_regr: standard deviation of regression predictions
        :param frame_id: number of frame (grid) in the dataset
        :return: tuple of three pyplot objects:
            1st - figure object
            2nd - plot with ground truth bounding boxes
            3rd - plt with predicted bounding boxes
        """
        grid, boxes = self.dataset[frame_id]
        grid, boxes = grid.cpu().squeeze(), boxes.cpu()
        frame_depth, frame_width, frame_length = self.dataset.grid_size
        detector_out_shape = (1, self.model.out_channels, frame_width // (2 ** self.model.n_pools),
                              frame_length // (2 ** self.model.n_pools))
        gt_former = GroundTruthFormer((frame_width, frame_length), detector_out_shape, device=self.device)

        fig = plt.figure(figsize=(12, 24))
        ax_gt = fig.add_subplot(2, 1, 1)
        ax_pred = fig.add_subplot(2, 1, 2)

        # plot gt bboxes
        ax_gt = draw_bev_with_bboxes(grid, boxes, edgecolor="red", ax=ax_gt)
        mapped_bb, mapped_bb_3sigma, mapped_bb_n3sigma = self.get_bbox_from_regression(mean_class,
                                                                                       mean_regr,
                                                                                       sigma_regr,
                                                                                       gt_former.prior_boxes_params)
        ax_pred = draw_bev_with_bboxes(grid, mapped_bb_3sigma.cpu(), edgecolor="red",
                                       label="model confidence 98%", ax=ax_pred)
        ax_pred = draw_bev_with_bboxes(grid, mapped_bb.cpu(), edgecolor="darkred", ax=ax_pred,
                                       label="model confidence 50%")
        ax_pred = draw_bev_with_bboxes(grid, mapped_bb_n3sigma.cpu(), edgecolor="lightcoral", ax=ax_pred,
                                       label="model confidence 2%")
        ax_pred.legend()

        return fig, ax_gt, ax_pred
    
    def get_bbox_from_regression(self, mean_class: torch.Tensor, mean_regr: torch.Tensor, sigma_regr: torch.Tensor,
                                 prior_boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate bboxes (of mean values, mean - sigma, mean + 3 sigma) from Monte Carlo estimations
        :param mean_class: mean of classification predictions
        :param mean_regr: mean of regression predictions
        :param sigma_regr: standard deviation of regression predictions
        :param prior_boxes: prior boxes from GroundTruthFormer
        :return: tuple of three torch.Tensors:
            1st - bbox with confidence 50% (from mean)
            2nd - bbox with confidence 98% (from mean + 3 * sigma)
            3rd - bbox with confidence 2%% (from mean - 3 * sigma)
        """

        prior_boxes = prior_boxes[(torch.sigmoid(mean_class) > self.threshold).squeeze()]
        unmapped_bb = mean_regr.squeeze()[(torch.sigmoid(mean_class.squeeze()) > self.threshold)]

        mapped_bb = torch.zeros_like(unmapped_bb)
        mapped_bb[:, 2:4] = prior_boxes[:, 2:4] / torch.clamp(torch.exp(unmapped_bb[:, 2:4]), min=1e-6)
        mapped_bb[:, 0:2] = prior_boxes[:, 0:2] - (unmapped_bb[:, 0:2] * mapped_bb[:, 2:4])
        mapped_bb[:, 4] = unmapped_bb[:, 4]
        mapped_bb[:, 5] = unmapped_bb[:, 5]

        mapped_bb_3sigma = mapped_bb.clone()
        
        # forward propagation of uncertainty for non-linear case:
        propagated_std = prior_boxes[:, 2:4] * (-torch.exp(-unmapped_bb[:, 2:4])) * \
            sigma_regr.squeeze()[(torch.sigmoid(mean_class.squeeze()) > self.threshold)][:, 2:4]
        mapped_bb_3sigma[:, 2:4] -= 3 * propagated_std
        
        mapped_bb_neg_3sigma = mapped_bb.clone()
        mapped_bb_neg_3sigma[:, 2:4] += 3 * propagated_std

        return mapped_bb, mapped_bb_3sigma, mapped_bb_neg_3sigma
