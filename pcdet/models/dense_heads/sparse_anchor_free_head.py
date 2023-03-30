import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ...utils import loss_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils


class SparseSeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Linear(input_channels, input_channels, bias=use_bias),
                    nn.BatchNorm1d(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Linear(input_channels, output_channels, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Linear):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


class SparseAnchorFreeHead(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        self.shared_mlp = nn.Sequential(
            nn.Linear(
                input_channels, self.model_cfg.SHARED_MLP_CHANNEL,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            ),
            nn.BatchNorm1d(self.model_cfg.SHARED_MLP_CHANNEL),
            nn.ReLU(),
        )

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(
                SparseSeparateHead(
                    input_channels=self.model_cfg.SHARED_MLP_CHANNEL,
                    sep_head_dict=cur_head_dict,
                    init_bias=-2.19,
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
                )
            )
        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.build_losses()

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())

    def visualize_heatmap(self, heatmap, pixel_coords, gt_boxes, feature_map_size, draw_gaussian):
        import matplotlib.pyplot as plt
        from pcdet.utils import box_utils
        import ss_visual_utils as V
        corners2d = box_utils.boxes_to_corners_2d(
            gt_boxes, point_cloud_range=[-75.2, -75.2, -2, 75.2, 75.2, 4], voxel_size=(0.1, 0.1, 0.2)
        )
        corners2d = np.round(corners2d / 8).astype(np.int)

        img = np.ones(feature_map_size) * -1
        img = V.draw_bev_boxes(img, corners2d, thickness=1, draw_arrow=False)  # the drawed lines will be zeros
        if draw_gaussian:
            mask = heatmap > 0
            pixel_coords = pixel_coords[mask]

            img[pixel_coords[:, 0], pixel_coords[:, 1]] = heatmap[mask] * 2
        else:
            mask = heatmap == 1
            pixel_coords = pixel_coords[mask]

            img[pixel_coords[:, 0], pixel_coords[:, 1]] = heatmap[mask] * 2

        plt.figure()
        plt.imshow(img)
        plt.show(block=False)
        import pdb
        pdb.set_trace()

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, pixel_coords, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2, use_diag_radius=False, use_rsn_heatmap=False
        ):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [for y-axis, for x-axis]
            pixel_coords: (N, 2), [y_idx, x_idx]

        Returns:

        """
        num_valid_pixel = pixel_coords.shape[0]
        heatmap = gt_boxes.new_zeros(num_classes, num_valid_pixel)
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        if use_diag_radius:
            radius = (dx ** 2 + dy ** 2).sqrt() / 2
        else:
            radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        pixel_x, pixel_y = pixel_coords[:, 1], pixel_coords[:, 0]
        pixel_x_float = pixel_x.float()
        pixel_y_float = pixel_y.float()

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            temp_dist = (pixel_x_float - coord_x[k]) ** 2 + (pixel_y_float - coord_y[k]) ** 2
            min_val, min_idx = temp_dist.min(dim=0)

            if min_val > (dx[k] / 2) ** 2 + (dy[k] / 2) ** 2:
                # even the nearest non-empty pixel is still far from the box center
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            cur_radius = radius[k].item()

            if use_rsn_heatmap:
                dist = temp_dist - ((pixel_x_float[min_idx] - coord_x[k]) ** 2 + (pixel_y_float[min_idx] - coord_y[k]) ** 2)
            else:
                dist = (pixel_x - pixel_x[min_idx]) ** 2 + (pixel_y - pixel_y[min_idx]) ** 2
            
            valid_mask = (dist <= cur_radius ** 2)
            sigma = (2 * cur_radius + 1) / 6

            masked_heatmap = torch.exp(-dist[valid_mask] / (2 * sigma * sigma))
            heatmap[cur_class_id, valid_mask] = torch.max(heatmap[cur_class_id, valid_mask], masked_heatmap)

            inds[k] = min_idx
            mask[k] = 1

            ret_boxes[k, 0] = coord_x[k] - pixel_x_float[min_idx].item()
            ret_boxes[k, 1] = coord_y[k] - pixel_y_float[min_idx].item()
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]

        # # for visualization only
        # for k in range(self.num_class):
        #     self.visualize_heatmap(
        #         heatmap[k].cpu().numpy(), pixel_coords.cpu().numpy(),
        #         gt_boxes=gt_boxes[gt_boxes[:, -1] == k + 1].cpu().numpy(),
        #         feature_map_size=feature_map_size, draw_gaussian=True
        #     )

        return heatmap, ret_boxes, inds, mask

    def assign_targets(self, gt_boxes, feature_map_size=None, feature_map_stride=None,
                       pixel_coords=None, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
            feature_map_size: (2) [for y-axis, for x-axis]
            pixel_coords: (N, 3)  [bs_idx, y_idx, x_idx]
        Returns:

        """
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': []
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list, heatmap_masks_list = [], [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, ret_boxes, inds, mask = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head,
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    pixel_coords=pixel_coords[pixel_coords[:, 0] == bs_idx, 1:],
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                    use_diag_radius=target_assigner_cfg.get('USE_DIAG_RADIUS', False), 
                    use_rsn_heatmap=target_assigner_cfg.get('USE_RSN_HEATMAP', False)
                )
                heatmap_list.append(heatmap)
                target_boxes_list.append(ret_boxes)
                inds_list.append(inds)
                masks_list.append(mask)

            ret_dict['heatmaps'].append(torch.cat(heatmap_list, dim=-1))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))

        ret_dict['pixel_coords'] = pixel_coords
        return ret_dict

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']
        pixel_coords = self.forward_ret_dict['pixel_coords']

        tb_dict = {}
        loss = 0

        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm']).T  # (N, num_classes)

            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])

            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)

            pred_boxes_full = pred_boxes.new_zeros(target_boxes.shape)  # (B, num_max_objs, code_size)
            for bs_idx in range(target_boxes.shape[0]):
                pred_boxes_full[bs_idx] = pred_boxes[pixel_coords[:, 0] == bs_idx][target_dicts['inds'][idx][bs_idx]]

            reg_loss = self.reg_loss_func(
                pred_boxes_full, target_dicts['masks'][idx], target=target_boxes
            )
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            loss += hm_loss + loc_loss
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()

        tb_dict['rpn_loss'] = loss.item()
        if torch.isnan(loss):
            # for debug
            print(hm_loss, loc_loss, loss, tb_dict, pred_dicts, target_dicts, reg_loss)
            import pickle
            pickle.dump(pred_dicts, open('pred_dicts.pkl', 'wb'))
            pickle.dump(target_dicts, open('target_dicts.pkl', 'wb'))
            exit(-1)
        return loss, tb_dict

    def decode_bbox_from_heatmap(self, heatmap, rot_cos, rot_sin, center, center_z, dim, pixel_coords, batch_size,
                             point_cloud_range=None, voxel_size=None, feature_map_stride=None, vel=None, K=100,
                             circle_nms=False, score_thresh=None, post_center_limit_range=None):
        """
        Args:
            heatmap: (num_class, N)
            rot_cos: (N)
            rot_sin: (N)
            center: (N, 2)
            center_z: (N)
            dim: (N, 3)
            pixel_coords: (N, 3)  [bs_idx, y_idx, x_idx]
            batch_size: int
        Returns:

        """

        if circle_nms:
            assert False, 'Do not support this feature currently'

        ret_pred_dicts = []
        for bs_idx in range(batch_size):
            mask = pixel_coords[:, 0] == bs_idx
            cur_scores = heatmap[:, mask]

            # decide the chosen positions
            topk_scores, topk_inds = torch.topk(cur_scores, K, dim=-1)  # (num_class, K)

            # decide the chosen categories, here one position could be chosen more than one categories
            final_scores, topk_ind = torch.topk(topk_scores.view(-1), K, dim=-1)  # (K)
            final_class_ids = (topk_ind // K).int()
            topk_inds = topk_inds.view(-1)[topk_ind]

            # box decoding
            cur_center = center[mask][topk_inds]
            cur_center_z = center_z[mask][topk_inds]
            cur_rot_sin = rot_sin[mask][topk_inds]
            cur_rot_cos = rot_cos[mask][topk_inds]
            cur_dim = dim[mask][topk_inds]
            cur_angle = torch.atan2(cur_rot_sin, cur_rot_cos)
            cur_pixel_coords = pixel_coords[mask][topk_inds]
            xs, ys = cur_pixel_coords[:, 2:3], cur_pixel_coords[:, 1:2]

            xs = xs.view(K, 1) + cur_center[:, 0:1]
            ys = ys.view(K, 1) + cur_center[:, 1:2]

            xs = xs * feature_map_stride * voxel_size[0] + point_cloud_range[0]
            ys = ys * feature_map_stride * voxel_size[1] + point_cloud_range[1]

            box_part_list = [xs, ys, cur_center_z, cur_dim, cur_angle]
            if vel is not None:
                cur_vel = vel[mask][topk_inds]
                box_part_list.append(cur_vel)

            final_box_preds = torch.cat((box_part_list), dim=-1)

            assert post_center_limit_range is not None
            mask = (final_box_preds[:, :3] >= post_center_limit_range[:3]).all(1)
            mask &= (final_box_preds[:, :3] <= post_center_limit_range[3:]).all(1)

            if score_thresh is not None:
                mask &= (final_scores > score_thresh)

            ret_pred_dicts.append({
                'pred_boxes': final_box_preds[mask],
                'pred_scores': final_scores[mask],
                'pred_labels': final_class_ids[mask]
            })
        return ret_pred_dicts

    def generate_predicted_boxes(self, batch_size, pred_dicts, pixel_coords):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm'].sigmoid()  # (N, num_classes)
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None

            final_pred_dicts = self.decode_bbox_from_heatmap(
                heatmap=batch_hm.T, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel,
                pixel_coords=pixel_coords, batch_size=batch_size,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                circle_nms=(post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range,
            )

            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]
                if post_process_cfg.NMS_CONFIG.NMS_TYPE != 'circle_nms':
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=None
                    )

                    final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                    final_dict['pred_scores'] = selected_scores
                    final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        return ret_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                pixel_features: (N, C)
                pixel_coords: (N, 3)
                spatial_shape: (for y-axis, for x-axis)
                gt_boxes (optional): (B, num_boxes, code_size + 1)
        Returns:
            for testing:
            data_dict:
                final_box_dicts (list of dict, length: batch_size):
                    pred_boxes: (M, code_size)
                    pred_scores: (M)
                    pred_labels: (M), index from 1 to num_class
        """
        pixel_features = data_dict['pixel_features']  # (N, C)
        pixel_coords = data_dict['pixel_coords']  # (N, 3) [bs_idx, y_idx, x_idx]
        spatial_shape = data_dict['spatial_shape_2d']  # (for y-axis, for x-axis)

        x = self.shared_mlp(pixel_features)

        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))

        if self.training:
            target_dict = self.assign_targets(
                data_dict['gt_boxes'], feature_map_size=spatial_shape,
                feature_map_stride=data_dict.get('spatial_features_2d_strides', None),
                pixel_coords=pixel_coords,
                debug_info=data_dict
            )
            self.forward_ret_dict['target_dicts'] = target_dict
            self.forward_ret_dict['pixel_coords'] = pixel_coords

        self.forward_ret_dict['pred_dicts'] = pred_dicts

        if (not self.training) or self.predict_boxes_when_training:
            final_box_dicts = self.generate_predicted_boxes(
                data_dict['batch_size'], pred_dicts,
                pixel_coords=pixel_coords
            )
            data_dict['final_box_dicts'] = final_box_dicts

        return data_dict