import datetime
from typing import Optional, List, Tuple

from sklearn.metrics import auc, precision_recall_curve
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange, tqdm

from dataset import create_nuscenes, NuscenesBEVDataset
from model import Detector, GroundTruthFormer


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

    def __call__(self, predictions: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Compute loss
        :param predictions: model output
        :param ground_truth: ground truth data
        :return: one element torch.Tensor loss
        """
        gt_regression = ground_truth[:, :self.prediction_units_per_cell * self.regression_values_per_unit, :, :]
        gt_classification = ground_truth[:, self.prediction_units_per_cell * self.regression_values_per_unit:, :, :]
        pred_regression = predictions[:, :self.prediction_units_per_cell * self.regression_values_per_unit, :, :]
        pred_classification = predictions[:, self.prediction_units_per_cell * self.regression_values_per_unit:, :, :]
        positive_reg_mask = torch.repeat_interleave(gt_classification, self.regression_values_per_unit, dim=1)
        pred_regression *= positive_reg_mask
        gt_regression *= positive_reg_mask  # may be redundant

        # perform hard negative mining
        n_positive = gt_classification.sum()
        n_values_to_eliminate = int((torch.numel(gt_classification) - (self.negative_positive_ratio + 1) * n_positive))

        if n_values_to_eliminate < 0:
            # this happens when classes on the provided frames are already well balanced
            return self.regression_base_loss(pred_regression, gt_regression) + \
                   self.classification_base_loss(pred_classification, gt_classification)

        negative_probs = pred_classification * (1 - gt_classification)
        negative_probs_flat = negative_probs.view(-1, 1)
        negative_probs_flat[torch.topk(negative_probs_flat, k=n_values_to_eliminate,
                                       largest=False, dim=0).indices] = 0  # filter low negative probabilities
        negative_probs = negative_probs_flat.view(negative_probs.size())
        pred_classification *= gt_classification  # leave only positive predictions
        pred_classification += negative_probs     # add mined negative predictions
        return self.regression_base_loss(pred_regression, gt_regression) + \
            self.classification_base_loss(pred_classification, gt_classification)


def pr_auc(gt_classes: torch.Tensor, preds: torch.Tensor) -> float:
    """
    Compute area under precision-recall curve
    :param gt_classes: 4D torch.Tensor of of ground truth classes labels
    :param preds: 4D torch.Tensor of predicted class probabilities
    :return: float, Precision-Recall AUC
    """
    precision, recall, _ = precision_recall_curve(gt_classes.numpy().flatten(), preds.numpy().flatten())
    return auc(precision, recall)


def frames_bboxes_collate_fn(batch: List[Tuple[torch.Tensor, List[torch.Tensor]]]) \
        -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
    """
    Collate frames and bounding boxes into proper batch
    :param batch: list of tuples (frame tensor, list of bboxes)
    :return: tuple of (frames tensor, lists of lists of bboxes)
    """
    grid = torch.stack([b[0] for b in batch])
    bboxes = [b[1] for b in batch]
    return grid, bboxes


def run_epoch(model: torch.nn.Module, loader: DataLoader, criterion: nn.modules.loss._Loss, gt_former: GroundTruthFormer,
              epoch: int, mode: str = 'train', writer: SummaryWriter = None,
              optimizer: torch.optim.Optimizer = None, device: torch.device = torch.device('cuda')) -> None:
    """
    Run one epoch for model. Can be used for both training and validation.
    :param model: pytorch model to be trained or validated
    :param loader: data loader to run model on batches
    :param criterion: callable class to calculate loss
    :param gt_former: callable class to form ground truth data to compute loss
    :param epoch: number of current epoch
    :param mode: `train` or `eval', controls model parameters update need
    :param writer: tensorboard writer
    :param optimizer: pytorch model parameters optimizer
    :param device: device to be used for model related computations
    """
    if mode == 'train':
        model.train()
    elif mode == 'eval':
        model.eval()
        cumulative_loss = 0
    else:
        raise ValueError(f'Unknown mode: {mode}')

    for i, (frames, bboxes) in enumerate(tqdm(loader, desc="Batch", leave=False)):
        frames = frames.to(device)
        preds = model(frames)
        gt_data = gt_former.form_gt(bboxes).to(device)
        loss = criterion(preds, gt_data)
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if writer is not None:
                writer.add_scalar('Loss', loss.item(), epoch * len(loader) + i)
            # TODO: save model based on score
        else:
            cumulative_loss += loss.item()
    if mode != 'train':
        writer.add_scalar('Loss', loss.item(), epoch * len(loader) + loader.batch_size)


def train(data_path: str, tb_path: str = None, n_scenes: int = 85, version: str = 'v1.0-trainval',
          n_loader_workers: int = 8, batch_size: int = 32, n_epochs: int = 100) -> None:
    """
    Train model, log training statistics if tb_path is specified.
    :param data_path: relative path to data folder
    :param tb_path: name of the folder for tensorboard data to be store in
    :param n_scenes: number of scenes in dataset
    :param version: version of the dataset
    :param n_loader_workers: number of CPU workers for data loader processing
    :param batch_size: batch size
    :param n_epochs: total number of epochs to train the model
    """
    # set up computing device for pytorch
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using device: GPU\n')
    else:
        device = torch.device('cpu')
        print('Using device: CPU\n')

    # set up tensorboard writer
    if tb_path is not None:
        date = datetime.datetime.now().strftime('%b-%d-%Y-%H:%M:%S')
        train_writer = SummaryWriter(log_dir=f'{tb_path}/{date}/train')
        val_writer = SummaryWriter(log_dir=f'{tb_path}/{date}/val')
        print(f'Logging tensorboard data to directory: {tb_path}/{date}\n')
    else:
        train_writer, val_writer = None, None
        print(f'No tensorboard logging will be performed\n')

    # set up dataset and model
    nuscenes = create_nuscenes(data_path)
    dataset = NuscenesBEVDataset(nuscenes=nuscenes, n_scenes=n_scenes)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_loader_workers,
                              collate_fn=frames_bboxes_collate_fn, pin_memory=True)
    print('Loaders are ready.\n',
          f'Number of batches in train loader: {len(train_loader)}\n')
          # f'Number of bathces in validation loader: {len(val_loader)}')

    frame_depth, frame_width, frame_length = dataset.grid_size
    model = Detector(img_depth=frame_depth).to(device)
    criterion = DetectionLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, gamma=0.5, step_size=50)  # TODO: adjust step_size empirically
    detector_out_shape = batch_size, model.out_channels, frame_width // (2 ** model.n_pools), \
        frame_length // (2 ** model.n_pools)
    gt_former = GroundTruthFormer((frame_width, frame_length), detector_out_shape)

    for epoch in trange(n_epochs, desc="Epoch"):
        run_epoch(model, train_loader, criterion, gt_former, epoch, mode='train', writer=train_writer,
                  optimizer=optimizer)
        scheduler.step()
        # run_epoch(model, val_loader, criterion, epoch, mode='val', writer=val_writer)


def eval(data_path: str, model_path: str, **kwargs):  # TODO: remove **kwargs and add proper keyword arguments
    raise NotImplementedError
