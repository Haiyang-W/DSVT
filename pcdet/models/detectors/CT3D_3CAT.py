from .detector3d_template import Detector3DTemplate
import torch
from ..model_utils import model_nms_utils

import time


class CT3D_3CAT(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.time_record1 = [0.] * (len(self.module_list)+1)
        self.time_record2 = 0.
        self.count = 0
        self.fix_module_id = model_cfg.get('FIX_MODULE', [])
        for i in self.fix_module_id:
            for name, p in self.module_list[i].named_parameters():
                p.requires_grad = False
        if len(self.fix_module_id) > 0:
            self.swith2eval = True
        else:
            self.swith2eval = False

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds

            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            elif post_process_cfg.NMS_CONFIG.get('MULTI_THRESH_NMS', False):
                assert isinstance(post_process_cfg.NMS_CONFIG.NMS_THRESH, list)
                all_cls_box_scores = []
                all_cls_box_preds = []
                all_cls_pred_labels = []
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1
                for i, cls_nms_th in enumerate(post_process_cfg.NMS_CONFIG.NMS_THRESH):
                    cls_select = label_preds == i+1
                    cls_pred_labels = label_preds[cls_select]
                    cls_box_scores = cls_preds[cls_select]
                    cls_box_preds = box_preds[cls_select]
                    nms_selected, cls_selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=cls_box_scores, box_preds=cls_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=None,
                        idx=i
                    )
                    if post_process_cfg.OUTPUT_RAW_SCORE:
                        max_cls_preds = torch.max(src_cls_preds, dim=-1)[0][cls_select]
                        cls_selected_scores = max_cls_preds[nms_selected]
                    all_cls_box_preds.append(cls_box_preds[nms_selected])
                    all_cls_box_scores.append(cls_selected_scores)
                    all_cls_pred_labels.append(cls_pred_labels[nms_selected])

                final_boxes = torch.cat(all_cls_box_preds, dim=0)
                final_scores = torch.cat(all_cls_box_scores, dim=0)
                final_labels = torch.cat(all_cls_pred_labels, dim=0)

            else:
                # print('DEBUG 0:', batch_dict.keys(), batch_dict['roi_labels'], batch_dict['has_class_labels'], cls_preds)
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1
                if post_process_cfg.NMS_CONFIG.NMS_TYPE == 'nms_gpu_bev':
                    selected, selected_scores = model_nms_utils.class_agnostic_nms_bev(
                        box_scores=cls_preds, box_preds=box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH)
                elif post_process_cfg.NMS_CONFIG.NMS_TYPE == 'multi_class_nms':
                    selected, selected_scores = model_nms_utils.multi_classes_nms_mmdet(
                        box_scores=cls_preds, box_preds=box_preds,
                        box_labels=label_preds-1, nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                else:
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=cls_preds, box_preds=box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]


                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

            # print('DEBUG 1:', final_scores, final_labels)
            # final_scores[(final_scores<post_process_cfg.CAT_THRE.Car)&(final_labels==1)] = 0
            # final_scores[(final_scores<post_process_cfg.CAT_THRE.Ped)&(final_labels==2)] = 0
            # final_scores[(final_scores<post_process_cfg.CAT_THRE.Cyc)&(final_labels==3)] = 0
            index_thre = final_scores!=0
            final_scores = final_scores[index_thre]
            final_labels = final_labels[index_thre]
            final_boxes = final_boxes[index_thre]

            # print('DEBUG 2:', final_scores.shape, final_labels, selected)
            # exit()

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)
        return pred_dicts, recall_dict

    def forward(self, batch_dict):
        if self.training and self.swith2eval:
            for i in self.fix_module_id:
                self.module_list[i].eval()
                if i == 4 and self.training:
                    self.module_list[i].dynamic_nms_type = 'TRAIN'
        if not self.training and self.swith2eval:
            for i in self.fix_module_id:
                self.module_list[i].eval()
                if i == 4 and not self.training:
                    self.module_list[i].dynamic_nms_type = 'TEST'

        start = time.time()
        for i, cur_module in enumerate(self.module_list):
            module_start = time.time()
            batch_dict = cur_module(batch_dict)
            self.time_record1[i] += time.time() - module_start
        end1 = time.time()

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            end = time.time()
            self.time_record1[-1] += end1 - start
            self.time_record2 += end - start
            self.count += 1
            # if self.count % 200 == 0:
            #     print('Average time: {} {}'.format(self.time_record1[-1] / self.count, self.time_record2 / self.count))
            #     print('Module time:', [self.time_record1[i] / self.count for i in range(len(self.module_list))])
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        if self.swith2eval:
            loss_rcnn, tb_dict = self.roi_head.get_loss({})
            tb_dict = {
                **tb_dict
            }

            loss = loss_rcnn
        else:
            loss_rpn, tb_dict = self.dense_head.get_loss()
            loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
            tb_dict = {
                'loss_rpn': loss_rpn.item(),
                **tb_dict
            }

            loss = loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict