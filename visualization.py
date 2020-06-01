from typing import TYPE_CHECKING, Union

import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import torch


def draw_bev(grid: Union["torch.Tensor", np.ndarray]) -> None:
    """
    Draw bird eye view of voxel grid using matplotlib
    :param grid: torch.Tensor of shape (depth, height, width) representing voxel grid
    """
    grid = np.asarray(grid)
    grid = grid.sum(axis=0).clip(max=1)
    plt.imshow(grid, cmap="gray")


def draw_bev_with_bboxes():
    pass
