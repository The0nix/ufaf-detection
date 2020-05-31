import os
from typing import Union, Tuple, List, Dict, Collection, Iterable

import numpy as np
import torch
import torch.utils.data as torchdata
import open3d as o3d
from nuscenes.nuscenes import NuScenes
from scipy.spatial.transform import Rotation


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


class NuscenesBEVDataset(torchdata.Dataset):
    """
    Dataset for LiDAR images from nuScenes that are converted
    to voxel grid as described in Fast & Furious paper
    :param root: path to directory with LiDAR images
    :param voxels_per_meter: number of voxels per meter or 1 / voxel_size
    :param crop_min_bound: min bound of the box to crop point cloud to in by X, Y and Z in meters
    :param crop_max_bound: max bound of the box to crop point cloud to in by X, Y and Z in meters
    :param nuscenes_version: version of the nuscenes dataset
    :param n_scenes: if not None represents the number of scenes downloaded
    (if you downloaded only a part of the dataset)
    """
    def __init__(self, root: str, voxels_per_meter: int=5,
                 crop_min_bound: Tuple[int, int, int]=(-40, -72, -2),
                 crop_max_bound: Tuple[int, int, int]=(40, 72, 3.5),
                 nuscenes_version: str="v1.0-trainval", n_scenes: int=None):
        self.root = root
        self.voxels_per_meter = voxels_per_meter
        self.voxel_size = 1 / voxels_per_meter
        self.crop_min_bound = np.array(crop_min_bound)
        self.crop_max_bound = np.array(crop_max_bound)
        self.filenames = sorted(os.listdir(self.root))

        # Initialize nuscenes dataset and determine it's size
        self.nuscenes = NuScenes(version=nuscenes_version, dataroot=root)
        self.n_scenes = n_scenes or len(self.nuscenes.scene)
        self.n_samples = sum(self.nuscenes.scene[i]["nbr_samples"] for i in range(self.n_scenes))

    def __getitem__(self, ix: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Get point cloud converted to voxel grid
        :param ix: index of element to get
        :return: tuple of (lidar voxel grid, list of bounding boxes)
        """
        sample = self.nuscenes.sample[ix]
        sample_data = self.nuscenes.get("sample_data", sample["data"]["LIDAR_TOP"])

        # Get lidar data
        filename = sample_data["filename"]
        filepath = os.path.join(self.root, filename)
        grid = torch.from_numpy(self._get_point_cloud(filepath))

        # Get GT boxes
        ego_pose = self.nuscenes.get("ego_pose", sample_data["ego_pose_token"])
        annotations = [self.nuscenes.get("sample_annotation", id_) for id_ in sample["anns"]]
        boxes = [self._annotation_to_bbox(ann, ego_pose, check_bounds=True)
                 for ann in annotations
                 if ann["category_name"].startswith("vehicle")]
        boxes = [b for b in boxes if b is not None]

        return grid, boxes

    def __len__(self) -> int:
        return self.n_samples

    def _get_point_cloud(self, filepath: str) -> np.ndarray:
        """
        Get open3d point cloud from .pcd.bin nuscenes file and convert it to
        numpy.ndarray representing voxelgrid
        :param filepath:
        :return: numpy.ndarray voxel grid
        """
        pcd = o3d.geometry.PointCloud()
        scan = np.fromfile(filepath, dtype=np.float32)
        points = scan.reshape([-1, 5])[:, :3]
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(self.crop_min_bound, self.crop_max_bound - self.voxel_size))
        grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.voxel_size)
        grid = self._voxelgrid_to_numpy(grid)

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
        voxels_ix = np.asarray([v.grid_index for v in voxelgrid.get_voxels()])
        # if actual crop is smaller then desired, we have to add space to voxels
        voxels_ix += ((voxelgrid.get_min_bound() - self.crop_min_bound).astype(int) * self.voxels_per_meter)
        result = np.zeros(((self.crop_max_bound - self.crop_min_bound) * self.voxels_per_meter).astype(int))
        result[tuple(voxels_ix.T)] = 1  # If this breaks, god help you
        return result

    def _annotation_to_bbox(self, annotation: Dict, ego_pose: Dict, check_bounds: bool=False) -> Union[torch.Tensor, None]:
        """
        Converts annotations to model-friendly bounding box
        if check_bounds is True, returns None if center of the annotation is out of bounds
        :param annotation: dict containing information about the annotation (nuscenes object)
        :param ego_pose: dict containing the of the LiDAR (nuscenes object)
        :param check_bounds: whether to return None if box is out of bounds
        :return: torch.Tensor [y, x, w, l, a_sin, a_cos]
        """
        translation = np.array(annotation["translation"]) - np.array(ego_pose["translation"])
        if check_bounds and not point_in_bounds(translation, self.crop_min_bound, self.crop_max_bound):
            return None
        x, y = translation[:2]
        w, l = annotation["size"][:2]

        rotation = Rotation(annotation["rotation"]) * Rotation(ego_pose["rotation"]).inv()
        a_rotated = rotation.apply([1, 0, 0])
        a_cos =  a_rotated[0]
        a_sin = np.sqrt(1 - a_cos ** 2) * np.sign(a_rotated[0])

        return torch.tensor([x, y, w, l, a_sin, a_cos])
