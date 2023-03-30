import torch.nn as nn
import pdb
import torch
import numpy as np
from numpy import *
import torch.nn.functional as F

from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate
from ..model_utils.ctrans import build_transformer


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class CT3DHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, voxel_size, point_cloud_range, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        self.up_dimension = MLP(input_dim = 28, hidden_dim = 64, output_dim = 256, num_layers = 3)


        num_queries = model_cfg.Transformer.num_queries
        hidden_dim = model_cfg.Transformer.hidden_dim
        self.num_points = model_cfg.Transformer.num_points

        self.class_embed = nn.Linear(hidden_dim, 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, self.box_coder.code_size * self.num_class, 4)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.transformer = build_transformer(model_cfg.Transformer)
        self.aux_loss = model_cfg.Transformer.aux_loss
        self.init_weights(weight_init='xavier')


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
        nn.init.normal_(self.bbox_embed.layers[-1].weight, mean=0, std=0.001)


    def get_global_grid_points_of_roi(self, rois):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_corner_points(rois, batch_size_rcnn)  # (BxN, 2x2x2, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        # pdb.set_trace()

        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_corner_points(rois, batch_size_rcnn):
        faked_features = rois.new_ones((2, 2, 2))

        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 2x2x2, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = dense_idx * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 2x2x2, 3)
        return roi_grid_points

    def spherical_coordinate(self, src, diag_dist):
        assert (src.shape[-1] == 27)
        device = src.device
        indices_x = torch.LongTensor([0,3,6,9,12,15,18,21,24]).to(device)  #
        indices_y = torch.LongTensor([1,4,7,10,13,16,19,22,25]).to(device) #
        indices_z = torch.LongTensor([2,5,8,11,14,17,20,23,26]).to(device)
        src_x = torch.index_select(src, -1, indices_x)
        src_y = torch.index_select(src, -1, indices_y)
        src_z = torch.index_select(src, -1, indices_z)
        dis = (src_x ** 2 + src_y ** 2 + src_z ** 2) ** 0.5
        phi = torch.atan(src_y / (src_x + 1e-5))
        the = torch.acos(src_z / (dis + 1e-5))
        dis = dis / (diag_dist + 1e-5)
        src = torch.cat([dis, phi, the], dim = -1)
        return src

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        rois = batch_dict['rois']
        batch_size = batch_dict['batch_size']
        num_rois = batch_dict['rois'].shape[-2]

        # corner
        corner_points, _ = self.get_global_grid_points_of_roi(rois)  # (BxN, 2x2x2, 3)
        corner_points = corner_points.view(batch_size, num_rois, -1, corner_points.shape[-1])  # (B, N, 2x2x2, 3)

        num_sample = self.num_points
        src = rois.new_zeros(batch_size, num_rois, num_sample, 4)

        for bs_idx in range(batch_size):
            cur_points = batch_dict['points'][(batch_dict['points'][:, 0] == bs_idx)][:,1:5]
            cur_batch_boxes = batch_dict['rois'][bs_idx]
            cur_radiis = torch.sqrt((cur_batch_boxes[:,3]/2) ** 2 + (cur_batch_boxes[:,4]/2) ** 2) * 1.2
            dis = torch.norm((cur_points[:,:2].unsqueeze(0) - cur_batch_boxes[:,:2].unsqueeze(1).repeat(1,cur_points.shape[0],1)), dim = 2)
            point_mask = (dis <= cur_radiis.unsqueeze(-1))
            for roi_box_idx in range(0, num_rois):
                cur_roi_points = cur_points[point_mask[roi_box_idx]]

                if cur_roi_points.shape[0] >= num_sample:
                    random.seed(0)
                    index = np.random.randint(cur_roi_points.shape[0], size=num_sample)
                    cur_roi_points_sample = cur_roi_points[index]

                elif cur_roi_points.shape[0] == 0:
                    cur_roi_points_sample = cur_roi_points.new_zeros(num_sample, 4)

                else:
                    empty_num = num_sample - cur_roi_points.shape[0]
                    add_zeros = cur_roi_points.new_zeros(empty_num, 4)
                    add_zeros = cur_roi_points[0].repeat(empty_num, 1)
                    cur_roi_points_sample = torch.cat([cur_roi_points, add_zeros], dim = 0)

                src[bs_idx, roi_box_idx, :, :] = cur_roi_points_sample

        src = src.view(batch_size * num_rois, -1, src.shape[-1])  # (b*128, 256, 4)


        corner_points = corner_points.view(batch_size * num_rois, -1)
        corner_add_center_points = torch.cat([corner_points, rois.view(-1, rois.shape[-1])[:,:3]], dim = -1)
        pos_fea = src[:,:,:3].repeat(1,1,9) - corner_add_center_points.unsqueeze(1).repeat(1,num_sample,1)  # 27 维度
        lwh = rois.view(-1, rois.shape[-1])[:,3:6].unsqueeze(1).repeat(1,num_sample,1)
        diag_dist = (lwh[:,:,0]**2 + lwh[:,:,1]**2 + lwh[:,:,2]**2) ** 0.5
        pos_fea = self.spherical_coordinate(pos_fea, diag_dist = diag_dist.unsqueeze(-1))
        # print(rois.shape)
        # print(rois)
        # print(torch.isnan(pos_fea).any())
        # # exit()

        src = torch.cat([pos_fea, src[:,:,-1].unsqueeze(-1)], dim = -1)
        src = self.up_dimension(src)

        # Transformer
        pos = torch.zeros_like(src)
        hs = self.transformer(src, self.query_embed.weight, pos)[0]

        # output
        rcnn_cls = self.class_embed(hs)[-1].squeeze(1)
        rcnn_reg = self.bbox_embed(hs)[-1].squeeze(1)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            # batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds

            if self.model_cfg.get('WITH_RESCORE', False):
                batch_dict['batch_cls_preds'] = batch_cls_preds
            else:
                batch_dict['batch_cls_preds'] = batch_dict['roi_scores'].unsqueeze(-1)

            if len(self.model_cfg.get('WITH_RESCORE_CLASS', [])) > 0 and self.model_cfg.get('WITH_RESCORE', False):
                batch_dict['batch_cls_preds'] = batch_dict['roi_scores'].unsqueeze(-1)
                for k in self.model_cfg.get('WITH_RESCORE_CLASS', []):
                    batch_dict['batch_cls_preds'] = torch.where(batch_dict['roi_labels'].unsqueeze(-1) == k,
                                                                torch.sigmoid(batch_cls_preds), batch_dict['batch_cls_preds'])

            batch_dict['cls_preds_normalized'] = True
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            self.forward_ret_dict = targets_dict

        return batch_dict