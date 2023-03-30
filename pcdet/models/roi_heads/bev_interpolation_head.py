import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...utils import common_utils, box_utils
from ..backbones_3d.pfe.voxel_set_abstraction import bilinear_interpolate_torch
from .roi_head_template import RoIHeadTemplate


class BEVInterpolationHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        self.point_cloud_range = kwargs['point_cloud_range']
        self.voxel_size = kwargs['voxel_size']

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        if self.model_cfg.ROI_GRID_POOL.get('GRID_TYPE', None) is None:
            pre_channel = self.model_cfg.ROI_GRID_POOL.IN_CHANNEL * GRID_SIZE * GRID_SIZE
        elif self.model_cfg.ROI_GRID_POOL.get('GRID_TYPE', None) == 'centerpoint':
            pre_channel = self.model_cfg.ROI_GRID_POOL.GRID_SIZE * self.model_cfg.ROI_GRID_POOL.IN_CHANNEL
        else:
            raise NotImplementedError

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

        if torch.__version__ >= '1.3':
            self.affine_grid = partial(F.affine_grid, align_corners=True)
            self.grid_sample = partial(F.grid_sample, align_corners=True)
        else:
            self.affine_grid = F.affine_grid
            self.grid_sample = F.grid_sample

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                spatial_features_2d: (B, C, H, W)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois'].detach()
        spatial_features_2d = batch_dict['spatial_features_2d'].detach()
        height, width = spatial_features_2d.size(2), spatial_features_2d.size(3)

        min_x = self.point_cloud_range[0]
        min_y = self.point_cloud_range[1]
        voxel_size_x = self.voxel_size[0]
        voxel_size_y = self.voxel_size[1]
        down_sample_ratio = self.model_cfg.ROI_GRID_POOL.DOWNSAMPLE_RATIO

        pooled_features_list = []
        torch.backends.cudnn.enabled = False
        for b_id in range(batch_size):
            # Map global boxes coordinates to feature map coordinates
            x1 = (rois[b_id, :, 0] - rois[b_id, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
            x2 = (rois[b_id, :, 0] + rois[b_id, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
            y1 = (rois[b_id, :, 1] - rois[b_id, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)
            y2 = (rois[b_id, :, 1] + rois[b_id, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)

            angle, _ = common_utils.check_numpy_to_torch(rois[b_id, :, 6])

            cosa = torch.cos(angle)
            sina = torch.sin(angle)

            theta = torch.stack((
                (x2 - x1) / (width - 1) * cosa, (x2 - x1) / (width - 1) * (-sina), (x1 + x2 - width + 1) / (width - 1),
                (y2 - y1) / (height - 1) * sina, (y2 - y1) / (height - 1) * cosa, (y1 + y2 - height + 1) / (height - 1)
            ), dim=1).view(-1, 2, 3).float()

            grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
            grid = self.affine_grid(
                theta,
                torch.Size((rois.size(1), spatial_features_2d.size(1), grid_size, grid_size))
            )

            pooled_features = self.grid_sample(
                spatial_features_2d[b_id].unsqueeze(0).expand(rois.size(1), spatial_features_2d.size(1), height, width),
                grid
            )

            pooled_features_list.append(pooled_features)

        torch.backends.cudnn.enabled = True
        pooled_features = torch.cat(pooled_features_list, dim=0)

        return pooled_features

    def get_roi_points(self, rois):
        batch_size = rois.shape[0]
        ret_points = []
        for bs_idx in range(batch_size):
            corners3d = box_utils.boxes_to_corners_3d(rois[bs_idx])  # (N, 8, 3)
            cur_points = torch.cat([
                (corners3d[:, None, 0] + corners3d[:, None, 2]) / 2,  # center
                (corners3d[:, None, 0] + corners3d[:, None, 1]) / 2,  # front-center
                (corners3d[:, None, 2] + corners3d[:, None, 3]) / 2,  # back-center
                (corners3d[:, None, 0] + corners3d[:, None, 3]) / 2,  # left-center
                (corners3d[:, None, 1] + corners3d[:, None, 2]) / 2,  # right-center
            ], dim=1)  # (N, 5, 3)
            ret_points.append(cur_points)

        ret_points = torch.stack(ret_points, dim=0)  # (B, N, 5, 3)
        return ret_points

    def roi_grid_pool_centerpoint(self, batch_dict):

        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois'].detach()
        spatial_features_2d = batch_dict['spatial_features_2d']
        if self.model_cfg.get('DETACH_BEV_MAP', False):
            spatial_features_2d = spatial_features_2d.detach()
        height, width = spatial_features_2d.size(2), spatial_features_2d.size(3)

        roi_points = self.get_roi_points(rois)  # (batch_size, N, num_points, 3)
        _, N, num_points, _ = roi_points.shape

        min_x = self.point_cloud_range[0]
        min_y = self.point_cloud_range[1]
        voxel_size_x = self.voxel_size[0]
        voxel_size_y = self.voxel_size[1]
        down_sample_ratio = self.model_cfg.ROI_GRID_POOL.DOWNSAMPLE_RATIO

        ret_roi_features = []
        for bs_idx in range(batch_size):
            cur_x = roi_points[bs_idx, :, :, 0].view(-1)  # (N*num_points)
            cur_y = roi_points[bs_idx, :, :, 1].view(-1)  # (N*num_points)

            cur_x = (cur_x - min_x) / voxel_size_x / down_sample_ratio
            cur_y = (cur_y - min_y) / voxel_size_y / down_sample_ratio

            cur_point_features = bilinear_interpolate_torch(spatial_features_2d[bs_idx].permute(1, 2, 0), cur_x, cur_y)
            ret_roi_features.append(cur_point_features.view(N, num_points, -1))

        ret_roi_features = torch.stack(ret_roi_features, dim=0)  # (B, N, num_points, C)
        return ret_roi_features.view(batch_size * N, num_points, -1)

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = batch_dict.get('roi_targets_dict', None)
            if targets_dict is None:
                targets_dict = self.assign_targets(batch_dict)
                batch_dict['rois'] = targets_dict['rois']
                batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        if self.model_cfg.ROI_GRID_POOL.get('GRID_TYPE', None) is None:
            pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)
        elif self.model_cfg.ROI_GRID_POOL.get('GRID_TYPE', None) == 'centerpoint':
            pooled_features = self.roi_grid_pool_centerpoint(batch_dict)
        else:
            raise NotImplementedError

        batch_size_rcnn = pooled_features.shape[0]

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict
