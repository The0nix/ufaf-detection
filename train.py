import datetime

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import create_nuscenes, NuscenesBEVDataset
from model import Detector


def train(data_path: str, tb_path: str = None, n_scenes: int = 85, version: str = 'v1.0-trainval',
          n_loader_workers: int = 8, batch_size: int = 32):
    # set up computing device for pytorch
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using device: GPU')
    else:
        device = torch.device('cpu')
        print('Using device: CPU')

    # set up tensorboard writer
    if tb_path is not None:
        date = datetime.datetime.now().strftime('%b-%d-%Y-%H:%M:%S')
        writer = SummaryWriter(log_dir=f'{tb_path}/{date}')
        print(f'Logging tensorboard data to directory: {tb_path}/{date}')
    else:
        print(f'No tensorboard logging will be performed')

    nuscenes = create_nuscenes(data_path)
    dataset = NuscenesBEVDataset(nuscenes=nuscenes, n_scenes=n_scenes)
    model = Detector(img_depth=20).to(device).train()
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, gamma=0.5, step_size=50)  # TODO: adjust step_size empirically


def eval(data_path: str, model_path: str, **kwargs):  # TODO: remove **kwargs and add proper keyword arguments
    raise NotImplementedError
