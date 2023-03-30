import torch
import numpy as np

from ...ops.iou3d_nms import iou3d_nms_utils
from ...ops.ioubev_nms import ioubev_nms_utils


def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
    src_box_scores = box_scores
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    selected = []
    if box_scores.shape[0] > 0:
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
        boxes_for_nms = box_preds[indices]
        keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
        )
        selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    return selected, src_box_scores[selected]

def class_agnostic_nms_bev(box_scores, box_preds, nms_config, score_thresh=None):
    box_preds_bev = box_preds[:, [0,1,3,4,6]]
    # xywhr2xyxyr
    boxes_bev_for_nms = torch.zeros_like(box_preds_bev)
    half_w = box_preds_bev[:, 2] / 2
    half_h = box_preds_bev[:, 3] / 2
    boxes_bev_for_nms[:, 0] = box_preds_bev[:, 0] - half_w
    boxes_bev_for_nms[:, 1] = box_preds_bev[:, 1] - half_h
    boxes_bev_for_nms[:, 2] = box_preds_bev[:, 0] + half_w
    boxes_bev_for_nms[:, 3] = box_preds_bev[:, 1] + half_h
    boxes_bev_for_nms[:, 4] = box_preds_bev[:, 4]

    src_box_scores = box_scores
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        boxes_bev_for_nms = boxes_bev_for_nms[scores_mask]

    selected = []
    if box_scores.shape[0] > 0:
        # box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
        box_scores_nms = box_scores
        boxes_for_nms = boxes_bev_for_nms
        # transformer to old mmdet3d coordinate
        """
        .. code-block:: none

                            up z    x front (yaw=-0.5*pi)
                               ^   ^
                               |  /
                               | /
        (yaw=-pi) left y <------ 0 -------- (yaw=0)
        """
        boxes_bev_for_nms[:, 4] = (-boxes_bev_for_nms[:, 4] + np.pi / 2 * 1)
        boxes_bev_for_nms[:, 4] = (boxes_bev_for_nms[:, 4] + np.pi) % (2*np.pi) - np.pi


        keep_idx = getattr(ioubev_nms_utils, nms_config.NMS_TYPE)(
            boxes_for_nms[:, 0:5], box_scores_nms,
            thresh=nms_config.NMS_THRESH,
            pre_maxsize=nms_config.NMS_PRE_MAXSIZE,
            post_max_size=nms_config.NMS_POST_MAXSIZE
        )
        selected = keep_idx
    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    return selected, src_box_scores[selected]


def multi_classes_nms(cls_scores, box_preds, nms_config, score_thresh=None):
    """
    Args:
        cls_scores: (N, num_class)
        box_preds: (N, 7 + C)
        nms_config:
        score_thresh:

    Returns:

    """
    pred_scores, pred_labels, pred_boxes = [], [], []
    for k in range(cls_scores.shape[1]):
        if score_thresh is not None:
            scores_mask = (cls_scores[:, k] >= score_thresh)
            box_scores = cls_scores[scores_mask, k]
            cur_box_preds = box_preds[scores_mask]
        else:
            box_scores = cls_scores[:, k]
            cur_box_preds = box_preds

        selected = []
        if box_scores.shape[0] > 0:
            box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
            boxes_for_nms = cur_box_preds[indices]
            keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                    boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
            )
            selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

        pred_scores.append(box_scores[selected])
        pred_labels.append(box_scores.new_ones(len(selected)).long() * k)
        pred_boxes.append(cur_box_preds[selected])

    pred_scores = torch.cat(pred_scores, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    pred_boxes = torch.cat(pred_boxes, dim=0)

    return pred_scores, pred_labels, pred_boxes

def multi_classes_nms_mmdet(box_scores, box_preds, box_labels, nms_config, score_thresh=None):
    """
    Args:
        cls_scores: (N,)
        box_preds: (N, 7 + C)
        box_labels: (N,)
        nms_config:

    Returns:

    """
    selected = []
    for k in range(len(nms_config.NMS_THRESH)):
        curr_mask = box_labels == k
        if score_thresh is not None and isinstance(score_thresh, float):
            curr_mask *= (box_scores > score_thresh)
        elif score_thresh is not None and isinstance(score_thresh, list):
            curr_mask *= (box_scores > score_thresh[k])
        curr_idx = torch.nonzero(curr_mask)[:, 0]
        curr_box_scores = box_scores[curr_mask]
        cur_box_preds = box_preds[curr_mask]

        curr_box_preds_bev = cur_box_preds[:, [0,1,3,4,6]]
        # xywhr2xyxyr
        curr_boxes_bev_for_nms = torch.zeros_like(curr_box_preds_bev)
        half_w = curr_box_preds_bev[:, 2] / 2
        half_h = curr_box_preds_bev[:, 3] / 2
        curr_boxes_bev_for_nms[:, 0] = curr_box_preds_bev[:, 0] - half_w
        curr_boxes_bev_for_nms[:, 1] = curr_box_preds_bev[:, 1] - half_h
        curr_boxes_bev_for_nms[:, 2] = curr_box_preds_bev[:, 0] + half_w
        curr_boxes_bev_for_nms[:, 3] = curr_box_preds_bev[:, 1] + half_h
        curr_boxes_bev_for_nms[:, 4] = curr_box_preds_bev[:, 4]

        if curr_box_scores.shape[0] > 0:
            # box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
            curr_box_scores_nms = curr_box_scores
            curr_boxes_for_nms = curr_boxes_bev_for_nms
            # transformer to old mmdet3d coordinate
            """
            .. code-block:: none

                                up z    x front (yaw=-0.5*pi)
                                ^   ^
                                |  /
                                | /
            (yaw=-pi) left y <------ 0 -------- (yaw=0)
            """
            curr_boxes_bev_for_nms[:, 4] = (-curr_boxes_bev_for_nms[:, 4] + np.pi / 2 * 1)
            curr_boxes_bev_for_nms[:, 4] = (curr_boxes_bev_for_nms[:, 4] + np.pi) % (2*np.pi) - np.pi


            keep_idx = getattr(ioubev_nms_utils, 'nms_gpu_bev')(
                curr_boxes_for_nms[:, 0:5], curr_box_scores_nms,
                thresh=nms_config.NMS_THRESH[k],
                pre_maxsize=nms_config.NMS_PRE_MAXSIZE[k],
                post_max_size=nms_config.NMS_POST_MAXSIZE[k]
            )
            curr_selected = curr_idx[keep_idx]
            selected.append(curr_selected)
    if len(selected) != 0:
        selected = torch.cat(selected)
        

    return selected, box_scores[selected]
