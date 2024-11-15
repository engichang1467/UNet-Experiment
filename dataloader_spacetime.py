"""RB2 Experiment Dataloader"""
import os
import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage
import warnings
# pylint: disable=too-manz-arguments, too-manz-instance-attributes, too-manz-locals


class RB2DataLoader(Dataset):
    """Pytorch Dataset instance for loading Rayleigh Bernard 2D dataset.

    Loads a 2d space + time cubic cutout from the whole simulation.
    """
    def __init__(self, data_dir="./", data_filename="./data/rb2d_ra1e6_s1.npz",
                 nx=128, nz=64, nt=16, n_samp_pts_per_crop=1024,
                 downsamp_xz=2, downsamp_t=4, normalize_output=False, normalize_hres=False,
                 return_hres=False, lres_filter='none', lres_interp='linear'):
        """

        Initialize DataSet
        Args:
          data_dir: str, path to the dataset folder, default="./"
          data_filename: str, name of the dataset file, default="rb2d_ra1e6_s42"
          nx: int, number of 'pixels' in x dimension for high res dataset.
          nz: int, number of 'pixels' in z dimension for high res dataset.
          nt: int, number of timesteps in time for high res dataset.
          n_samp_pts_per_crop: int, number of sample points to return per crop.
          downsamp_xz: int, downsampling factor for the spatial dimensions.
          downsamp_t: int, downsampling factor for the temporal dimension.
          normalize_output: bool, whether to normalize the range of each channel to [0, 1].
          normalize_hres: bool, normalize high res grid.
          return_hres: bool, whether to return the high-resolution data.
          lres_filter: str, filter to apply on original high-res image before interpolation.
                       choice of 'none', 'gaussian', 'uniform', 'median', 'maximum'.
          lres_interp: str, interpolation scheme for generating low res.
                       choice of 'linear', 'nearest'.
        """
        self.data_dir = data_dir
        self.data_filename = data_filename
        self.nx_hres = nx
        self.nz_hres = nz
        self.nt_hres = nt
        self.nx_lres = nx
        self.nz_lres = nz
        self.nt_lres = nt
        self.n_samp_pts_per_crop = n_samp_pts_per_crop
        self.normalize_output = normalize_output
        self.normalize_hres = normalize_hres
        self.return_hres = return_hres
        self.lres_interp = lres_interp

        # concatenating buoyancy, pressure, x-velocity (horizontal velocity), z-velocity (vertical velocity), and vorticity as a 5 channel array: 
        npdata = np.load(os.path.join(self.data_dir, self.data_filename))

        self.data = np.stack([npdata['buoyancy'], npdata['pressure'], npdata['horizontal_velocity'], npdata['vertical_velocity'], npdata['vorticity']], axis=0)

        self.data = self.data.astype(np.float32)
        self.data = self.data.transpose(0, 1, 3, 2)  # [c, t, z, x] -> [channel, time, space_coord_z, space_coord_x]
        nc_data, nt_data, nz_data, nx_data = self.data.shape

        # assert nx, nz, nt are viable
        if (nx > nx_data) or (nz > nz_data) or (nt > nt_data):
            raise ValueError('Resolution in each spatial temporal dimension x ({}), z({}), t({})'
                             'must not exceed dataset limits x ({}) z ({}) t ({})'.format(
                                 nx, nz, nt, nx_data, nz_data, nt_data))

        self.nx_start_range = np.arange(0, nx_data-nx+1)
        self.nz_start_range = np.arange(0, nz_data-nz+1)
        self.nt_start_range = np.arange(0, nt_data-nt+1)
        self.rand_grid = np.stack(np.meshgrid(self.nt_start_range,
                                              self.nz_start_range,
                                              self.nx_start_range, indexing='ij'), axis=-1)
        # (xaug, zaug, taug, 3)
        self.rand_start_id = self.rand_grid.reshape([-1, 3])
        self.scale_hres = np.array([self.nt_hres, self.nz_hres, self.nx_hres], dtype=np.int32)
        self.scale_lres = np.array([self.nt_lres, self.nz_lres, self.nx_lres], dtype=np.int32)

        # compute channel-wise mean and std
        self._mean = np.mean(self.data, axis=(1, 2, 3))
        self._std = np.std(self.data, axis=(1, 2, 3))

    def __len__(self):
        return self.rand_start_id.shape[0]

    def __getitem__(self, idx):
        """Get the random cutout data cube corresponding to idx.

        Args:
          idx: int, index of the crop to return. must be smaller than len(self).

        Returns:
          space_time_crop_hres (*optional): array of shape [4, nt_hres, nz_hres, nx_hres],
          where 4 are the phys channels pbuw.
          space_time_crop_lres: array of shape [4, nt_lres, nz_lres, nx_lres], where 4 are the phys
          channels pbuw.
          point_coord: array of shape [n_samp_pts_per_crop, 3], where 3 are the t, x, z dims.
                       CAUTION - point_coord are normalized to (0, 1) for the relative window.
          point_value: array of shape [n_samp_pts_per_crop, 4], where 4 are the phys channels pbuw.
        """
        t_id, z_id, x_id = self.rand_start_id[idx]
        space_time_crop_hres = self.data[:,
                                         t_id:t_id+self.nt_hres,
                                         z_id:z_id+self.nz_hres,
                                         x_id:x_id+self.nx_hres]  # [c, t, z, x]

        # create low res grid from hi res space time crop
        space_time_crop_hres_fil = space_time_crop_hres # Not filtering for now

        interp = RegularGridInterpolator(
            (np.arange(self.nt_hres), np.arange(self.nz_hres), np.arange(self.nx_hres)),
            values=space_time_crop_hres_fil.transpose(1, 2, 3, 0), method=self.lres_interp)

        lres_coord = np.stack(np.meshgrid(np.linspace(0, self.nt_hres-1, self.nt_lres),
                                          np.linspace(0, self.nz_hres-1, self.nz_lres),
                                          np.linspace(0, self.nx_hres-1, self.nx_lres),
                                          indexing='ij'), axis=-1)
        space_time_crop_lres = interp(lres_coord).transpose(3, 0, 1, 2)  # [c, t, z, x]

        # create random point samples within space time crop
        point_coord = np.random.rand(self.n_samp_pts_per_crop, 3) * (self.scale_hres - 1)
        point_value = interp(point_coord)
        point_coord = point_coord / (self.scale_hres - 1)

        if self.normalize_output:
            space_time_crop_lres = self.normalize_grid(space_time_crop_lres)
            point_value = self.normalize_points(point_value)
        if self.normalize_hres:
            space_time_crop_hres = self.normalize_grid(space_time_crop_hres)

        return_tensors = [space_time_crop_lres, point_coord, point_value]

        # cast everything to float32
        return_tensors = [t.astype(np.float32) for t in return_tensors]

        if self.return_hres:
            return_tensors = [space_time_crop_hres] + return_tensors
        return tuple(return_tensors)

    @property
    def channel_mean(self):
        """channel-wise mean of dataset."""
        return self._mean

    @property
    def channel_std(self):
        """channel-wise mean of dataset."""
        return self._std

    @staticmethod
    def _normalize_array(array, mean, std):
        """normalize array (np or torch)."""
        if isinstance(array, torch.Tensor):
            dev = array.device
            std = torch.tensor(std, device=dev)
            mean = torch.tensor(mean, device=dev)
        return (array - mean) / std

    @staticmethod
    def _denormalize_array(array, mean, std):
        """normalize array (np or torch)."""
        if isinstance(array, torch.Tensor):
            dev = array.device
            std = torch.tensor(std, device=dev)
            mean = torch.tensor(mean, device=dev)
        return array * std + mean

    def normalize_grid(self, grid):
        """Normalize grid.

        Args:
          grid: np array or torch tensor of shape [4, ...], 4 are the num. of phys channels.
        Returns:
          channel normalized grid of same shape as input.
        """
        # reshape mean and std to be broadcastable.
        g_dim = len(grid.shape)
        mean_bc = self.channel_mean[(...,)+(None,)*(g_dim-1)]  # unsqueeze from the back
        std_bc = self.channel_std[(...,)+(None,)*(g_dim-1)]  # unsqueeze from the back
        return self._normalize_array(grid, mean_bc, std_bc)


    def normalize_points(self, points):
        """Normalize points.

        Args:
          points: np array or torch tensor of shape [..., 4], 4 are the num. of phys channels.
        Returns:
          channel normalized points of same shape as input.
        """
        # reshape mean and std to be broadcastable.
        g_dim = len(points.shape)
        mean_bc = self.channel_mean[(None,)*(g_dim-1)]  # unsqueeze from the front
        std_bc = self.channel_std[(None,)*(g_dim-1)]  # unsqueeze from the front
        return self._normalize_array(points, mean_bc, std_bc)

    def denormalize_grid(self, grid):
        """Denormalize grid.

        Args:
          grid: np array or torch tensor of shape [4, ...], 4 are the num. of phys channels.
        Returns:
          channel denormalized grid of same shape as input.
        """
        # reshape mean and std to be broadcastable.
        g_dim = len(grid.shape)
        mean_bc = self.channel_mean[(...,)+(None,)*(g_dim-1)]  # unsqueeze from the back
        std_bc = self.channel_std[(...,)+(None,)*(g_dim-1)]  # unsqueeze from the back
        return self._denormalize_array(grid, mean_bc, std_bc)


    def denormalize_points(self, points):
        """Denormalize points.

        Args:
          points: np array or torch tensor of shape [..., 4], 4 are the num. of phys channels.
        Returns:
          channel denormalized points of same shape as input.
        """
        # reshape mean and std to be broadcastable.
        g_dim = len(points.shape)
        mean_bc = self.channel_mean[(None,)*(g_dim-1)]  # unsqueeze from the front
        std_bc = self.channel_std[(None,)*(g_dim-1)]  # unsqueeze from the front
        return self._denormalize_array(points, mean_bc, std_bc)

