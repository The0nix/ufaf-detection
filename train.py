import os
import datetime
from typing import Optional, Union, List, Tuple

import sklearn.metrics as skmetrics
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange, tqdm

from dataset import create_nuscenes, NuscenesBEVDataset
from model import Detector, DetectionLoss, GroundTruthFormer


def pr_auc(gt_classes: torch.Tensor, preds: torch.Tensor) -> float:
    """
    Compute area under precision-recall curve
    :param gt_classes: 4D torch.Tensor of of ground truth classes labels
    :param preds: 4D torch.Tensor of predicted class probabilities
    :return: float, Precision-Recall AUC
    """
    score = skmetrics.average_precision_score(
        gt_classes.detach().flatten().cpu().numpy(),
        preds.detach().flatten().cpu().numpy(),
    )
    return score


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


# noinspection PyUnboundLocalVariable
def run_epoch(model: torch.nn.Module, loader: DataLoader, criterion: nn.modules.loss._Loss,
              gt_former: GroundTruthFormer, epoch: int, mode: str = 'train', writer: SummaryWriter = None,
              optimizer: Optimizer = None, n_dumps_per_epoch: int = 10, train_loader_size: int = None,
              device: Union[torch.device, str] = torch.device('cpu')) -> Optional[Tuple[float, float]]:
    """
    Run one epoch for model. Can be used for both training and validation.
    :param model: pytorch model to be trained or validated
    :param loader: data loader to run model on batches
    :param criterion: callable class to calculate loss
    :param gt_former: callable class to form ground truth data to compute loss
    :param epoch: number of current epoch
    :param mode: `train` or `val', controls model parameters update need
    :param writer: tensorboard writer
    :param optimizer: pytorch model parameters optimizer
    :param n_dumps_per_epoch: how many times per epoch to dump images to tensorboard (not implemented yet)
    :param train_loader_size: number of objects in the train loader, needed for plots scaling in val mode
    :param device: device to be used for model related computations
    :return: values for cumulative loss and score (only in 'val' mode)
    """
    if mode == 'train':
        model.train()
    elif mode == 'val':
        model.eval()
        cumulative_loss, cumulative_score = 0, 0
    else:
        raise ValueError(f'Unknown mode: {mode}')

    for i, (frames, bboxes) in enumerate(tqdm(loader, desc="Batch", leave=False)):
        frames = frames.to(device)
        bboxes = [bbox.to(device) for bbox in bboxes]
        preds = model(frames)
        gt_data = gt_former.form_gt(bboxes)
        loss = criterion(preds, gt_data)
        score = pr_auc(gt_data[0], preds[0])
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if writer is not None:
                writer.add_scalar('Loss', loss.item(), epoch * len(loader) + i)
                writer.add_scalar('Score', score, epoch * len(loader) + i)
        else:
            cumulative_loss += loss.item()
            cumulative_score += score
    if mode == 'val':
        if train_loader_size is not None:
            # scales val data to train data on the plots
            iterations = epoch * train_loader_size + loader.batch_size
        else:
            iterations = epoch * len(loader) + loader.batch_size
        cumulative_loss /= len(loader)
        cumulative_score /= len(loader)
        if writer is not None:
            writer.add_scalar('Loss', cumulative_loss, iterations)
            writer.add_scalar('Score', cumulative_score, iterations)
        return cumulative_loss, cumulative_score


def train(data_path: str, output_model_dir: str, input_model_path: Optional[str] = None, tb_path: str = None,
          n_scenes: int = 10, nuscenes_version: str = 'v1.0-mini', learning_rate: int = 1e-4,
          n_dumps_per_epoch: int = 10, n_loader_workers: int = 4, batch_size: int = 12, n_epochs: int = 50,
          device_id: List[int] = None) -> None:
    """
    Train model, log training statistics if tb_path is specified.
    :param data_path: relative path to data folder
    :param output_model_dir: path to directory to save model weights to
    :param input_model_path: path to model weights. If None, create new model
    :param tb_path: name of the folder for tensorboard data to be store in
    :param n_scenes: number of scenes in dataset
    :param nuscenes_version: version of the dataset
    :param learning_rate: learning rate for Adam
    :param n_dumps_per_epoch: how many times per epoch to dump images to tensorboard (not implemented yet)
    :param n_loader_workers: number of CPU workers for data loader processing
    :param batch_size: batch size
    :param n_epochs: total number of epochs to train the model
    :param device_id: list of gpu device ids to use, e.g [0, 1]
    """
    # create path for model save
    os.makedirs(output_model_dir, exist_ok=True)

    # set up computing device for pytorch
    if torch.cuda.is_available():
        if device_id is None:
            device_id = [0]
        if max(device_id) < torch.cuda.device_count():
            # device_id/s all exist on machine,
            # device is set as a root device
            device = torch.device(f'cuda:{device_id[0]}')
        else:
            # device_id is out of range, setting to defaults cuda:0
            print('Warning: specified number of gpu device_id is larger than available, using cuda:0.')
            device = torch.device('cuda:0')
        print('Using device: GPU\n')
    else:
        device = torch.device('cpu')
        print('Using device: CPU\n')

    date = datetime.datetime.now().strftime('%b-%d-%Y-%H:%M:%S')

    # set up tensorboard writer
    if tb_path is not None:
        train_writer = SummaryWriter(log_dir=f'{tb_path}/{date}/train')
        val_writer = SummaryWriter(log_dir=f'{tb_path}/{date}/val')
        print(f'Logging tensorboard data to directory: {tb_path}/{date}\n')
    else:
        train_writer, val_writer = None, None
        print(f'No tensorboard logging will be performed\n')

    # set up dataset and model
    nuscenes = create_nuscenes(data_path, nuscenes_version)
    train_dataset = NuscenesBEVDataset(nuscenes=nuscenes, n_scenes=n_scenes, mode='train')
    val_dataset = NuscenesBEVDataset(nuscenes=nuscenes, n_scenes=n_scenes, mode='val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_loader_workers,
                              collate_fn=frames_bboxes_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=n_loader_workers,
                            collate_fn=frames_bboxes_collate_fn, pin_memory=True)
    print('Loaders are ready.\n',
          f'Number of batches in train loader: {len(train_loader)}\n'
          f'Number of bathces in validation loader: {len(val_loader)}', sep='')

    frame_depth, frame_width, frame_length = train_dataset.grid_size
    model = Detector(img_depth=frame_depth)
    if input_model_path is not None:
        model.load_state_dict(torch.load(input_model_path, map_location="cpu"))
    model = model.to(device)
    criterion = DetectionLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, gamma=0.5, step_size=50)  # TODO: adjust step_size empirically
    detector_out_shape = (batch_size, model.out_channels, frame_width // (2 ** model.n_pools),
                          frame_length // (2 ** model.n_pools))
    gt_former = GroundTruthFormer((frame_width, frame_length), detector_out_shape, device=device)

    if len(device_id) > 1 and max(device_id) < torch.cuda.device_count():
        # if more than one device_id specified, use DataParallel
        model = nn.DataParallel(model, device_ids=device_id)
    model = model.to(device)

    best_val_score = float('-inf')
    for epoch in trange(n_epochs, desc="Epoch"):
        run_epoch(model, train_loader, criterion, gt_former, epoch, mode='train',
                  writer=train_writer, optimizer=optimizer, device=device)
        scheduler.step()
        val_loss, val_score = run_epoch(model, val_loader, criterion, gt_former, epoch,
                                        mode='val', train_loader_size=len(train_loader), writer=val_writer,
                                        device=device)
        # saving model weights in case validation loss AND score are better
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), f'{output_model_dir}/{date}.pth')
            print('\nModel checkpoint is saved.\n',
                  f'loss: {val_loss:.3f}, score: {val_score:.3f}\n')


def eval(data_path: str, model_path: str, n_scenes: int = 10, nuscenes_version: str = 'v1.0-mini',
         n_loader_workers: int = 4, batch_size: int = 12):
    """
    Evaluate model.
    :param data_path: relative path to data folder
    :param model_path: relative path to save model weights
    :param n_scenes: number of scenes in dataset
    :param nuscenes_version: version of the dataset
    :param n_loader_workers: number of CPU workers for data loader processing
    :param batch_size: batch size
    """
    # set up computing device for pytorch
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using device: GPU\n')
    else:
        device = torch.device('cpu')
        print('Using device: CPU\n')

    # set up dataset and model
    nuscenes = create_nuscenes(data_path, version=nuscenes_version)
    eval_dataset = NuscenesBEVDataset(nuscenes=nuscenes, n_scenes=n_scenes, mode='val')
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, num_workers=n_loader_workers,
                             collate_fn=frames_bboxes_collate_fn, pin_memory=True)

    print('Validation loader is ready.\n',
          f'Number of batches in eval loader: {len(eval_loader)}\n', sep='')

    frame_depth, frame_width, frame_length = eval_dataset.grid_size
    model = Detector(img_depth=frame_depth).to(device)
    # load model from checkpoint
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)
    criterion = DetectionLoss()
    detector_out_shape = (batch_size, model.out_channels, frame_width // (2 ** model.n_pools),
                          frame_length // (2 ** model.n_pools))
    gt_former = GroundTruthFormer((frame_width, frame_length), detector_out_shape, device=device)
    eval_loss, eval_score = run_epoch(model, eval_loader, criterion, gt_former, epoch=1, mode='val', device=device)
    return eval_loss, eval_score
