import torch.nn as nn
# import spconv
from ....utils.spconv_utils import replace_feature, spconv


class SparseHeightCompression(nn.Module):
    def __init__(self, model_cfg, grid_size=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.grid_size = grid_size  # (X, Y, Z)
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        if 'encoded_spconv_tensor' in batch_dict:
            encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
            spatial_features = encoded_spconv_tensor.dense()  # (N, C, D, H, W)
            N, C, D, H, W = spatial_features.shape
            spatial_features = spatial_features.view(N, C * D, H, W)

            indices_2d = spatial_features.sum(dim=1).nonzero()  # (num_points, 3)  [bs_idxs, y, x]
            pixel_features = spatial_features[indices_2d[:, 0], :, indices_2d[:, 1], indices_2d[:, 2]]
            spatial_shape = spatial_features.shape[2:4]
            batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        else:
            voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
            pixel_features = voxel_features
            indices_2d = voxel_coords[:, [0, 2, 3]]
            spatial_shape = self.grid_size[0:2][::-1]  # (H, W)

        batch_dict['pixel_features'] = pixel_features
        batch_dict['pixel_coords'] = indices_2d
        batch_dict['spatial_shape'] = spatial_shape
        return batch_dict


class SparseHeightCompressionWithConv(nn.Module):
    def __init__(self, model_cfg, grid_size=None, input_channels=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.grid_size = grid_size  # (X, Y, Z)
        self.input_channels = input_channels if input_channels is not None else self.model_cfg.IN_CHANNELS
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.conv_height = spconv.SparseSequential(
            # [200, 150, 2] ==> [200, 150]
            spconv.SparseConv3d(self.input_channels, self.num_bev_features, (2, 1, 1), stride=(2, 1, 1),
                                padding=0, bias=False, indice_key='spconv_height'),
            nn.BatchNorm1d(self.num_bev_features),
            nn.ReLU(),
        )

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        sparse_tensor = self.conv_height(encoded_spconv_tensor)
        spatial_shape = sparse_tensor.spatial_shape[1:3]
        indices_2d = sparse_tensor.indices[:, [0, 2, 3]]
        pixel_features = sparse_tensor.features

        batch_dict['pixel_features'] = pixel_features
        batch_dict['pixel_coords'] = indices_2d
        batch_dict['spatial_shape'] = spatial_shape
        return batch_dict