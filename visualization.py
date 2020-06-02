from typing import TYPE_CHECKING, Union, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import utils

if TYPE_CHECKING:
    import torch


def auto_ax(func):
    """
    Decorator to automatically create plt.Axes in function arguments.
    If added to function with ax keyword argumen, it will automatically create ax object if that argument is None
    :return: wrapped function
    """
    def wrapped(*args, ax: Optional[plt.Axes] = None, **kwargs):
        if ax is None:
            _, ax = plt.subplots(1)
        return func(*args, ax=ax, **kwargs)
    return wrapped


@auto_ax
def draw_bev(grid: Union["torch.Tensor", np.ndarray], ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Draw bird eye view of voxel grid using matplotlib
    :param grid: torch.Tensor or np.ndarray of shape (depth, height, width) representing voxel grid
    :param ax: plt.Axes to draw in. If None, plt.Axes object will be created
    return: plt.Axes object with visualized grid
    """
    grid = np.asarray(grid)
    grid = grid.sum(axis=0).clip(max=1)
    ax.imshow(grid, cmap="gray")

    return ax


@auto_ax
def draw_bev_with_bboxes(grid: Union["torch.Tensor", np.ndarray],
                         bboxes: List[Union["torch.Tensor", np.ndarray]],
                         ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Draw bird eye view of voxel grid and corresponding bounding boxes using matplotlib
    :param grid: torch.Tensor or np.ndarray of shape (depth, height, width) representing voxel grid
    :param bboxes: list of bounding boxes of (y, x, w, l, a_sin, a_cos)
    :param ax: plt.Axes to draw in. If None, plt.Axes object will be created
    :return: plt.Axes object with visualized grid and bounding boxes
    """
    draw_bev(grid, ax=ax)

    # Create and add patch for each bbox
    for i, bbox in enumerate(bboxes):
        vertices = utils.bbox_to_coordinates(bbox, rot=True)[:, [1, 0]]
        patch = patches.Polygon(vertices, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(patch)

    return ax
