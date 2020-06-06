import os
import math
import matplotlib.pyplot as plt
import torch
from typing import Tuple

from dataset import create_nuscenes, NuscenesBEVDataset
from visualization import draw_bev_with_bboxes
from model import GroundTruthFormer, Detector



torch.random.manual_seed(2)



class McProcessor:
    """
        Forms Monte Carlo based uncertanties and visualizes them
         :param model: pytorch model
         :param data_path: relative path to data folder
         :param version: version of the dataset
         :param n_scenes: number of scenes in dataset
         :param threshold: threshold for choosing is bbox or not
         :return: Tuple[torch.tensor, np.ndarray] - first  - grid tensor, seond - gt_bboxes
    """

    def __init__(self, data_path: str, version: str = "v1.0-mini", n_scenes: int = 85,
                 threshold: int = 0.5, model_path: str = None, model: torch.nn.Module = None):

        self.version = version
        self.n_scenes = n_scenes
        self.nuscenes = create_nuscenes(data_path, version)
        self.dataset = NuscenesBEVDataset(nuscenes=self.nuscenes, n_scenes=n_scenes)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('Using device: GPU\n')
        else:
            self.device = torch.device('cpu')
            print('Using device: CPU\n')

        if model is not None:
            self.model = model
            self.model = self.model.to(self.device)

        else:
            assert (model_path is not None)
            frame_depth, _, _ = self.get_grid_size()
            self.model = Detector(img_depth=frame_depth)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.threshold = threshold
        

    def get_grid_size(self) -> Tuple[int, int, int]:
        """
              :return: Tuple[int, int, int] - first  - frame_depth, frame_width, frame_length
        """

        frame_depth, frame_width, frame_length = self.dataset.grid_size
        return frame_depth, frame_width, frame_length


    def visualise_montecarlo(self, data_number: int = 0, n_samples: int = 10, batch_size: int = 4,
                             save_imgs: bool = False, saving_folder: str = "pics/"):

        """
           :param data_number: id of data(grid) in scene
           :param n_samples: numper of samples for Monte Carlo approach
           :param batch_size: size of batch
           :param save_imgs: - flag, if true - save figs to folder pics
           :param saving_folder: - path to the folder, where images will be saved 
                                   (if folder doesnt exist, creates new)
            :return: Tuple[plt.figure.Figure, matplotlib.axes.Axes, matplotlib.axes.Axes] -
                                first - figure object
                                second - GT plot
                                third - prediction plot
        """

        mean_class, _, mean_reg, sigma_reg = self.apply_monte_carlo(data_number, n_samples, batch_size)

        fig, ax_gt, ax_pred = self._vis_mc(mean_class, mean_reg, sigma_reg, data_number)

        if save_imgs:

                if not os.path.exists(saving_folder):
                    os.makedirs(saving_folder)
                img_path = saving_folder + '{version}_{n_scenes}_{data_number}_full.png'.format(
                           version=self.version, n_scenes=self.n_scenes, data_number=data_number)
                
                fig.savefig(img_path)
                img_path = saving_folder + '{version}_{n_scenes}_{data_number}_gt.png'.format(
                            version=self.version, n_scenes=self.n_scenes, data_number=data_number)
                extent = ax_gt.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

                fig.savefig(img_path, bbox_inches=extent)

                img_path = saving_folder + '{version}_{n_scenes}_{data_number}_pred.png'.format(
                            version=self.version, n_scenes=self.n_scenes, data_number=data_number)
                extent_ped = ax_pred.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(img_path, bbox_inches=extent_ped)

        return fig, ax_gt, ax_pred


    def apply_monte_carlo(self, data_number: int = 0,
                          n_samples: int = 10, batch_size: int = 4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
           Apply Monte Carlo dropout  for representing model uncertainty

           :param data_number: id of data(grid) in scene
           :param n_samples: number of samples for Monte Carlo approach
           :param batch_size: = size of bach
           :return: Tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -
                                                    1st  - tensor of mean values of class model prediction,
                                                    2nd - tensor of standart deviations of class model prediction
                                                    3rd  - tensor of mean values of regression model prediction,
                                                    4th - tensor of standart deviations of regression model prediction
           """
        
        assert (n_samples>1)

        # set up computing device for pytorch

        self.model.to(self.device)
        # keep dropouts active
        self.model.train()

        grid, boxes = self.dataset[data_number]

        class_output, reg_output = self.model(grid[None].to(self.device))
        model_out_shape = list(reg_output.shape)
        class_output_shape = list(class_output.shape)
        
        samples_reg = reg_output #.unsqueeze(-1)      #torch.empty(model_out_shape + [0]).to(device)
        samples_class = class_output #.unsqueeze(-1)  #torch.empty(class_output_shape + [0]).to(device)
        
        # TODO: append to single batch for speeding up
        for i in range(0, math.ceil((n_samples - 1)/batch_size)):
            current_batch_size = min((n_samples - 1) - (i * batch_size), batch_size)
            stacked_grid = torch.stack(current_batch_size * [grid]) #.squeeze()
            class_output, reg_output = self.model(stacked_grid.to(self.device))
            samples_reg = torch.cat((reg_output, samples_reg))
            samples_class = torch.cat((samples_class, class_output))

        # calc mean sigma of regression
        mean_reg = torch.mean(samples_reg, dim=0).detach().unsqueeze(0)
        variance_reg = torch.var(samples_reg, dim=0)
        sigma_reg = torch.sqrt(variance_reg).detach().unsqueeze(0)
        # calc mean sigma of classification
        mean_class = torch.mean(samples_class, dim=0).detach().unsqueeze(0)
        variance_class = torch.var(samples_class, dim=0)
        sigma_class = torch.sqrt(variance_class).detach().unsqueeze(0)

        return mean_class, sigma_class, mean_reg, sigma_reg

    def _vis_mc(self, mean_class: torch.Tensor, mean_regr: torch.Tensor,
               sima_regr: torch.Tensor, data_number: int = 0):
        """
           visualization of predictions processed by self.apply_monte_carlo
           :param mean_class: relative path to data folder
           :param sigma_class: version of the dataset
           :param mean_regr: number of scenes in dataset
           :param sima_regr: id of data(grid) in scene
           :param data_number: id of data(grid) in scene
           :return: Tuple(matplotlib.Axes, matplotlib.Axes) -  gtplots
        """

        grid, boxes = self.dataset[data_number]
        frame_depth, frame_width, frame_length = self.get_grid_size()

        detector_out_shape = 1, self.model.out_channels, frame_width // (2 ** self.model.n_pools), \
                             frame_length // (2 ** self.model.n_pools)
        gt_former = GroundTruthFormer((frame_width, frame_length), detector_out_shape, device=self.device)

        is_bbox = torch.sigmoid(mean_class) >self.threshold
        unc_prior_bboxes = gt_former.prior_boxes_params[is_bbox.squeeze()]

        grid = grid.cpu().squeeze()

        fig = plt.figure(figsize=(12,24))
        ax_gt = fig.add_subplot(2, 1, 1)
        ax_pred = fig.add_subplot(2, 1, 2)

        # plot gt bboxes
        ax_gt = draw_bev_with_bboxes(grid.cpu(), boxes.cpu(), edgecolor="red", ax=ax_gt)

        mapped_bb, mapped_bb_3sigma, mapped_bb_n3sigma = self.get_bbox_from_regression(mean_regr.cpu(),
                                                                                      sima_regr.cpu(),
                                                                                      mean_class.cpu(),
                                                                                      gt_former.prior_boxes_params)

        ax_pred = draw_bev_with_bboxes(grid.cpu(), mapped_bb_3sigma.cpu(), edgecolor="red",
                                       label="model confidence 98%", ax=ax_pred)
        ax_pred = draw_bev_with_bboxes(grid.cpu(), mapped_bb.cpu(), edgecolor="darkred", ax=ax_pred,
                                       label="model confidence 50%")
        ax_pred = draw_bev_with_bboxes(grid.cpu(), mapped_bb_n3sigma.cpu(), edgecolor="lightcoral", ax=ax_pred,
                                       label="model confidence 2%")
        ax_pred.legend()

        return fig, ax_gt, ax_pred

    
    def get_bbox_from_regression(self, mean_regr: torch.Tensor, sigma_regr: torch.Tensor,
                                 mean_class: torch.Tensor,
                                 prior_boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
            Calculating of bboxes (of mean values, mean+-3sigma)from predictions processed by self.apply_monte_carlo
            
            :param mean_regr: mean values of predicted regressions
            :param sigma_regr: std of predicted regressions
            :param mean_class: mean values of predicted regressions
            :param prior_boxes: prior boxes from GroundTruthFormer
            :return:  Tuple[torch.Tensor, torch.Tensor, torch.Tensor], where:
                                                        first - bbox with confidence 50% (from mean)
                                                        second - bbox with confidence 98% (from mean+3sigma)
                                                        third - bbox with confidence 33% (from mean-sigma)

        """

        prior_boxes = prior_boxes[(torch.sigmoid(mean_class) > self.threshold).squeeze()].cpu()

        unmmaped_bb = mean_regr.squeeze()[(torch.sigmoid(mean_class.squeeze()) > self.threshold)]

        mapped_bb = torch.zeros_like(unmmaped_bb)
        mapped_bb[:, 2:4] = prior_boxes[:, 2:4] / torch.clamp(torch.exp(unmmaped_bb[:, 2:4]), min=1e-6)
        mapped_bb[:, 0:2] = prior_boxes[:, 0:2] - (unmmaped_bb[:, 0:2] * mapped_bb[:, 2:4])
        mapped_bb[:, 4] = unmmaped_bb[:, 4]
        mapped_bb[:, 5] = unmmaped_bb[:, 5]

        
        mapped_bb_3sigma = mapped_bb.clone()
        
        # forward propagation of uncertainty for non-linear case:
        propagated_std = prior_boxes[:, 2:4] * (-torch.exp(-unmmaped_bb[:, 2:4])) * sigma_regr.squeeze()[(torch.sigmoid(mean_class.squeeze()) > self.threshold)][:, 2:4]
        mapped_bb_3sigma[:, 2:4] -= 3 * propagated_std
        
        mapped_bb_neg_3sigma = mapped_bb.clone()
        mapped_bb_neg_3sigma[:, 2:4] += 3 * propagated_std

        return mapped_bb, mapped_bb_3sigma, mapped_bb_neg_3sigma

if __name__ == "__main__":
    # data_path = "/home/robot/repos/nuscenes/v1.0-mini"
    data_path = "C:\\repos\\v1.0-mini"
    n_scenes = 10

    # t = torch.rand(1,25,46,6)
    # t1 = torch.rand(4,25,46,6)
    # t2 = torch.cat((t1,t),dim=0)
    # print(t2.shape)
    # exit()
    # processing given model
    # model = Detector(27)
    # mc_processor = Mc_processor(data_path, version="v1.0-mini", n_scenes=n_scenes, data_number=20,
    #                             model=model)c
    # ax_gt, ax_pred = mc_processor.visualise_montecarlo(data_number=20, n_samples=10)
    # plt.show()
    # processing saved model

    mc_processor = McProcessor(data_path, version="v1.0-mini", n_scenes=n_scenes,
                                model_path="C:\\Users\\Alex\\Downloads\\Jun-06-2020-02_42_23.pth")
    fig, ax_gt, ax_pred = mc_processor.visualise_montecarlo(batch_size=2, data_number=21, n_samples=5, save_imgs=0)
    plt.show()

