import os

import torch
import torch.utils.data as torchdata
import open3d as o3d


class LidarBIVDataset(torchdata.Dataset):
    def __init__(self, root: str, n_timesteps: int, voxel_size: float=0.05):
        """
        Dataset for LiDAR images that are converted to voxel grid as described
        in Fast & Furious paper
        :param root: path to directory with LiDAR images
        :param n_timesteps: number of time steps to include into the output tensor
        :param voxel_size: size of the voxel in the resulted grid
        """
        self.root = root
        self.voxel_size = voxel_size
        self.n_timesteps = n_timesteps
        self.filenames = sorted(os.listdir(self.root))

    def __getitem__(self, ix: int):
        """
        Get point cloud converted to voxel grid
        :param ix: index of element to get
        :return:
        """
        filepath = os.path.join(self.root, self.filenames[ix])
        pcd = o3d.io.read_point_cloud(filepath)
        grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.voxel_size)
        return torch.from_numpy(grid.voxels)


    def __len__(self):
        return len(self.filenames)
