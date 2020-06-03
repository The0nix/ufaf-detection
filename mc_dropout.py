import torch
from torch import nn
from dataset import create_nuscenes, NuscenesBEVDataset
from typing import Tuple


torch.random.manual_seed(1)


def run_monte_carlo(model: torch.nn.Module, data_path: str, version: str = "v1.0-mini", n_scenes: int = 85, data_number: int=0,
                    n_samples: int=30) -> Tuple[torch.Tensor,torch.Tensor]:
    """
       Apply Monte Carlo dropout  for representing model uncertainty
       :param model: pytorch model to be trained or validated
       :param data_path: relative path to data folder
       :param version: version of the dataset
       :param n_scenes: number of scenes in dataset
       :param data_number: id of data(grid) in scene
       :param: n_samples: numper of samples for Monte Carlo approach
       :return: Tuple(torch.tensor, torch.tensor) - first  - tensor of mean values of model prediction, 
                                                    second - tensor of standart deviations of model prediction
       """

    # keep dropouts active
    model.eval()
    
    # set up computing device for pytorch
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using device: GPU\n')
    else:
        device = torch.device('cpu')
        print('Using device: CPU\n')

    model.to(device)

    nuscenes = create_nuscenes(data_path, version)
    dataset = NuscenesBEVDataset(nuscenes=nuscenes, n_scenes=n_scenes)
    grid, boxes = dataset[data_number]

    model_out_shape = list(model(grid.to(device)).shape)
    samples = torch.empty(model_out_shape + [0]).to(device)

    #TODO: append to single batch for speeding up
    for _ in range(n_samples):
        pred = model(grid.to(device))

        samples = torch.cat((samples, pred.reshape(model_out_shape+[1])), dim=-1)

    mean = torch.mean(samples, dim=4)
    variance = torch.var(samples, dim=4)
    sigma = torch.sqrt(variance)

    return mean, sigma


if __name__ == "__main__":


    class FakeModel(nn.Module):
        """
        Model to be used only as a sanity checker for mc approach
        """
        def __init__(self):
            super().__init__()
            self.conv1d = nn.Conv1d(1, 1, kernel_size=1, bias=False)
            self.net = nn.Sequential(
                nn.Conv2d(27, 42, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.Dropout(0.2),
                nn.ReLU(inplace=True),
            )

        def forward(self, frames: torch.Tensor) -> torch.Tensor:
            batch_size, *pic_size = frames.shape
            frames = torch.reshape(frames, (batch_size, 1, pic_size[0] * pic_size[1] * pic_size[2]))
            fused = self.conv1d(frames)
            fused = torch.reshape(fused, (batch_size, *pic_size))
            out = self.net(fused)
            return out

    model = FakeModel()

    # data_path = "/home/robot/repos/nuscenes/v1.0-mini"
    data_path = "C:\\repos\\v1.0-mini"
    n_scenes = 1
    run_monte_carlo(model, data_path, n_scenes= n_scenes)
