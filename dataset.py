from typing import Union, Tuple, List, Dict, Collection, Iterable

import numpy as np
import torch
import torch.utils.data as torchdata
import open3d as o3d
from nuscenes.utils import data_classes as nuscenes_data_classes
from nuscenes.nuscenes import NuScenes


def point_in_bounds(point: Union[Collection, Iterable],
                    min_bound: Union[Collection, Iterable],
                    max_bound: Union[Collection, Iterable]) -> bool:
    """
    Determines whether the point is within bounds
    :param point:
    :param min_bound:
    :param max_bound:
    :return:
    """
    assert(len(point) == len(min_bound) == len(max_bound))
    return np.all(np.asarray(point) >= np.asarray(min_bound)) and np.all(np.asarray(point) <= np.asarray(max_bound))


def create_nuscenes(root: str, version: str = "v1.0-trainval") -> NuScenes:
    """
    Creates nuScenes object that can be later passed to NuscenesBEVDataset.
    Warning, it takes up a considerable abound of RAM space.
    :param root: path to folder with nuscenes dataset
    :param version: version of the dataset
    :return: created NuScenes object
    """
    return NuScenes(dataroot=root, version=version)


class NuscenesBEVDataset(torchdata.Dataset):
    """
    Dataset for LiDAR images from nuScenes that are converted
    to voxel grid as described in Fast & Furious paper
    :param nuscenes: NuScenes object (can be created with create_nuscenes)
    :param voxels_per_meter: number of voxels per meter or 1 / voxel_size
    :param crop_min_bound: min bound of the box to crop point cloud to in by X, Y and Z in meters
    :param crop_max_bound: max bound of the box to crop point cloud to in by X, Y and Z in meters
    :param n_scenes: if not None represents the number of scenes downloaded
    (if you downloaded only a part of the dataset)

    Example:
        >>> nuscenes = create_nuscenes("./data")
        >>> ds = NuscenesBEVDataset(nuscenes, n_scenes=85)
        >>> grid, boxes = ds[0]
    """
    def __init__(self, nuscenes: NuScenes, voxels_per_meter: int = 5,
                 crop_min_bound: Tuple[int, int, int] = (-72, -40, -2),
                 crop_max_bound: Tuple[int, int, int] = (72, 40, 3.5),
                 n_scenes: int = None) -> None:
        self.voxels_per_meter = voxels_per_meter
        self.voxel_size = 1 / voxels_per_meter
        # Change to YXZ, because lidar's "forward" is Y (see https://www.nuscenes.org/data-collection)
        self.crop_min_bound = np.array(crop_min_bound)[[1, 0, 2]]
        self.crop_max_bound = np.array(crop_max_bound)[[1, 0, 2]]
        self.grid_size = \
            tuple(((self.crop_max_bound - self.crop_min_bound) * self.voxels_per_meter)[[2, 0, 1]].astype(int))

        # Initialize nuscenes dataset, skip samples without vehicles and determine  dataset's size
        self.nuscenes = nuscenes
        self.n_scenes = n_scenes or len(self.nuscenes.scene)
        self.n_samples_total = sum(self.nuscenes.scene[i]["nbr_samples"] for i in range(self.n_scenes))
        self.samples_ix = [ix for ix in range(self.n_samples_total) if self._sample_has_vehicles(ix)]
        self.n_samples = len(self.samples_ix)

        # TODO: Add train/validation split

    def __getitem__(self, ix: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get point cloud converted to voxel grid
        :param ix: index of element to get
        :return: tuple of:
            lidar voxel grid of shape (1, depth, height, width),
            list of bounding boxes of (y, x, w, l, a_sin, a_cos)
        """
        if ix >= len(self):
            raise IndexError(f"Index {ix} is out of bounds")
        ix = self.samples_ix[ix]  # Only get samples from our subset of indexes with vehicles

        sample = self.nuscenes.sample[ix]
        filepath, annotations, _ = self.nuscenes.get_sample_data(sample["data"]["LIDAR_TOP"])

        # Get lidar data
        grid = torch.from_numpy(self._get_point_cloud(filepath))
        grid.unsqueeze_(0)  # adds time dimension

        # Get GT boxes
        boxes = [self._annotation_to_bbox(ann, check_bounds=True)
                 for ann in annotations if ann.name.startswith("vehicle")]
        boxes = [b for b in boxes if b is not None]
        boxes = torch.stack(boxes) if boxes else torch.empty(0, 0)

        return grid, boxes

    def __len__(self) -> int:
        return self.n_samples

    def _get_point_cloud(self, filepath: str) -> np.ndarray:
        """
        Get open3d point cloud from .pcd.bin nuscenes file and convert it to
        numpy.ndarray representing voxelgrid
        :param filepath:
        :return: numpy.ndarray voxel grid of shape (depth, height, width)
        """
        pcd = o3d.geometry.PointCloud()
        scan = np.fromfile(filepath, dtype=np.float32)
        points = scan.reshape([-1, 5])[:, :3]  # Each dot in file has 5 values, but we only need 3 (x, y and z)
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(self.crop_min_bound, self.crop_max_bound - self.voxel_size))
        grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.voxel_size)
        grid = self._voxelgrid_to_numpy(grid)
        grid = np.rollaxis(grid, 2, 0)

        return grid

    def _voxelgrid_to_numpy(self, voxelgrid: o3d.geometry.VoxelGrid) -> np.ndarray:
        """
        Converts o3d.geometry.VoxelGrid to numpy.ndarray
        with ones on places corresponding to occupied voxels.
        This function also makes sure that resulting numpy array is within
        crop bounds
        :param voxelgrid: VoxelGrid to convert
        :return: numpy.ndarray representation of voxel grid
        """
        voxels_ix = np.array([v.grid_index for v in voxelgrid.get_voxels()])
        # if actual crop is smaller then desired, we have to add space to voxels
        voxels_ix += ((voxelgrid.get_min_bound() - self.crop_min_bound) * self.voxels_per_meter).astype(int)
        result = np.zeros(((self.crop_max_bound - self.crop_min_bound) * self.voxels_per_meter).astype(int),
                          dtype=np.float32)
        result[tuple(voxels_ix.T)] = 1  # If this breaks, god help you
        return result

    def _annotation_to_bbox(self, annotation: nuscenes_data_classes.Box, check_bounds: bool = False) \
            -> Union[torch.Tensor, None]:
        """
        Converts annotations to model-friendly bounding box
        if check_bounds is True, returns None if center of the annotation is out of bounds
        :param annotation: nuscenes Box containing information about the annotation
        must be received with nuscenes.get_sample_data(id, use_flat_vehicle_coordinates=True)
        :param check_bounds: if box is out of current dataset bounds, return None
        :return: torch.Tensor (y, x, w, l, a_sin, a_cos)
        """
        if check_bounds and not point_in_bounds(annotation.center, self.crop_min_bound, self.crop_max_bound):
            return None
        y, x, _ = np.array(annotation.center) * self.voxels_per_meter
        x = x - self.crop_min_bound[1] * self.voxels_per_meter      # in crop bounds LiDAR's X is our Y
        y = y - self.crop_min_bound[0] * self.voxels_per_meter      #
        w, l, _ = np.array(annotation.wlh) * self.voxels_per_meter

        a_rotated = annotation.rotation_matrix @ np.array([1, 0, 0])
        a_cos = a_rotated[0]
        a_sin = np.sqrt(1 - a_cos ** 2) * np.sign(a_rotated[1])

        return torch.tensor([y, x, w, l, a_sin, a_cos])

    def _sample_has_vehicles(self, ix: int) -> bool:
        """
        Check if sample has vehicles in it
        :param ix: index of the sample
        :return: whether sample has vehicles
        """
        sample = self.nuscenes.sample[ix]
        filepath, annotations, _ = self.nuscenes.get_sample_data(sample["data"]["LIDAR_TOP"])
        for ann in annotations:
            if ann.name.startswith("vehicle") and point_in_bounds(ann.center, self.crop_min_bound, self.crop_max_bound):
                return True
        return False
