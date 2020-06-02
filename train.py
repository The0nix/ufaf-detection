import datetime

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange

from dataset import create_nuscenes, NuscenesBEVDataset
from model import Detector, GroundTruthFormer


class DetectionLoss(nn.modules.loss._Loss):
    def __init__(self, prediction_units_per_cell: int = 6, regression_values_per_unit: int = 6,
                 classification_values_per_unit: int = 1) -> None:
        super().__init__()
        self.prediction_units_per_cell = prediction_units_per_cell
        self.regression_values_per_unit = regression_values_per_unit
        self.classification_values_per_unit = classification_values_per_unit

    def __call__(self, predictions: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        return self.forward(predictions, ground_truth)

    def forward(self, predictions: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        gt_regression = ground_truth[:, :self.prediction_units_per_cell * self.regression_values_per_unit, :, :]
        gt_classification = ground_truth[:, self.prediction_units_per_cell * self.regression_values_per_unit:, :, :]
        pred_regression = predictions[:, :self.prediction_units_per_cell * self.regression_values_per_unit, :, :]
        pred_classification = predictions[:, self.prediction_units_per_cell * self.regression_values_per_unit:, :, :]
        mask = torch.repeat_interleave(gt_classification, self.regression_values_per_unit, dim=1)
        pred_regression *= mask
        # TODO: add normalization
        return nn.SmoothL1Loss()(pred_regression, gt_regression) + \
            nn.BCEWithLogitsLoss()(pred_classification, gt_classification)


def auc_pr(preds: torch.Tensor, gt_classes: torch.Tensor) -> float:
    raise NotImplementedError


def run_epoch(model: torch.nn.Module, loader: DataLoader, criterion: nn.modules.loss._Loss, gt_former: GroundTruthFormer,
              epoch: int, mode: str = 'train', writer: SummaryWriter = None,
              optimizer: torch.optim.optimizer.Optimizer = None, device: torch.device = torch.device('cuda')) -> None:
    if mode == 'train':
        model.train()
    else:
        model.eval()
        cumulative_loss = 0
    for i, (frames, bboxes) in enumerate(loader):
        frames = frames.to(device)
        preds = model(frames)
        gt_data = gt_former.form_gt(bboxes)
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
          n_loader_workers: int = 8, batch_size: int = 32, n_epochs: int = 100):
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
                              pin_memory=True)
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

    for epoch in trange(n_epochs):
        run_epoch(model, train_loader, criterion, gt_former, epoch, mode='train', writer=train_writer,
                  optimizer=optimizer)
        scheduler.step()
        # run_epoch(model, val_loader, criterion, epoch, mode='val', writer=val_writer)

def eval(data_path: str, model_path: str, **kwargs):  # TODO: remove **kwargs and add proper keyword arguments
    raise NotImplementedError
