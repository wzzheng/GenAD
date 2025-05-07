import copy
import warnings

import numpy as np
import torch
from mmcv import ConfigDict
from mmcv.ops import nms

from mmcv.ops.iou3d_det.iou3d_utils import nms_gpu, nms_normal_gpu
from ..bbox.transforms import bbox_mapping_back, bbox3d2result, bbox3d_mapping_back
from ..bbox.structures.utils import xywhr2xyxyr

def merge_aug_proposals(aug_proposals, img_metas, cfg):
    """Merge augmented proposals (multiscale, flip, etc.)

    Args:
        aug_proposals (list[Tensor]): proposals from different testing
            schemes, shape (n, 5). Note that they are not rescaled to the
            original image size.

        img_metas (list[dict]): list of image info dict where each dict has:
            'img_shape', 'scale_factor', 'flip', and may also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys see
            `mmcv/datasets/pipelines/formatting.py:Collect`.

        cfg (dict): rpn test config.

    Returns:
        Tensor: shape (n, 4), proposals corresponding to original image scale.
    """

    cfg = copy.deepcopy(cfg)

    # deprecate arguments warning
    if 'nms' not in cfg or 'max_num' in cfg or 'nms_thr' in cfg:
        warnings.warn(
            'In rpn_proposal or test_cfg, '
            'nms_thr has been moved to a dict named nms as '
            'iou_threshold, max_num has been renamed as max_per_img, '
            'name of original arguments and the way to specify '
            'iou_threshold of NMS will be deprecated.')
    if 'nms' not in cfg:
        cfg.nms = ConfigDict(dict(type='nms', iou_threshold=cfg.nms_thr))
    if 'max_num' in cfg:
        if 'max_per_img' in cfg:
            assert cfg.max_num == cfg.max_per_img, f'You set max_num and ' \
                f'max_per_img at the same time, but get {cfg.max_num} ' \
                f'and {cfg.max_per_img} respectively' \
                f'Please delete max_num which will be deprecated.'
        else:
            cfg.max_per_img = cfg.max_num
    if 'nms_thr' in cfg:
        assert cfg.nms.iou_threshold == cfg.nms_thr, f'You set ' \
            f'iou_threshold in nms and ' \
            f'nms_thr at the same time, but get ' \
            f'{cfg.nms.iou_threshold} and {cfg.nms_thr}' \
            f' respectively. Please delete the nms_thr ' \
            f'which will be deprecated.'

    recovered_proposals = []
    for proposals, img_info in zip(aug_proposals, img_metas):
        img_shape = img_info['img_shape']
        scale_factor = img_info['scale_factor']
        flip = img_info['flip']
        flip_direction = img_info['flip_direction']
        _proposals = proposals.clone()
        _proposals[:, :4] = bbox_mapping_back(_proposals[:, :4], img_shape,
                                              scale_factor, flip,
                                              flip_direction)
        recovered_proposals.append(_proposals)
    aug_proposals = torch.cat(recovered_proposals, dim=0)
    merged_proposals, _ = nms(aug_proposals[:, :4].contiguous(),
                              aug_proposals[:, -1].contiguous(),
                              cfg.nms.iou_threshold)
    scores = merged_proposals[:, 4]
    _, order = scores.sort(0, descending=True)
    num = min(cfg.max_per_img, merged_proposals.shape[0])
    order = order[:num]
    merged_proposals = merged_proposals[order, :]
    return merged_proposals


def merge_aug_bboxes(aug_bboxes, aug_scores, img_metas, rcnn_test_cfg):
    """Merge augmented detection bboxes and scores.

    Args:
        aug_bboxes (list[Tensor]): shape (n, 4*#class)
        aug_scores (list[Tensor] or None): shape (n, #class)
        img_shapes (list[Tensor]): shape (3, ).
        rcnn_test_cfg (dict): rcnn test config.

    Returns:
        tuple: (bboxes, scores)
    """
    recovered_bboxes = []
    for bboxes, img_info in zip(aug_bboxes, img_metas):
        img_shape = img_info[0]['img_shape']
        scale_factor = img_info[0]['scale_factor']
        flip = img_info[0]['flip']
        flip_direction = img_info[0]['flip_direction']
        bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip,
                                   flip_direction)
        recovered_bboxes.append(bboxes)
    bboxes = torch.stack(recovered_bboxes).mean(dim=0)
    if aug_scores is None:
        return bboxes
    else:
        scores = torch.stack(aug_scores).mean(dim=0)
        return bboxes, scores


def merge_aug_scores(aug_scores):
    """Merge augmented bbox scores."""
    if isinstance(aug_scores[0], torch.Tensor):
        return torch.mean(torch.stack(aug_scores), dim=0)
    else:
        return np.mean(aug_scores, axis=0)


def merge_aug_masks(aug_masks, img_metas, rcnn_test_cfg, weights=None):
    """Merge augmented mask prediction.

    Args:
        aug_masks (list[ndarray]): shape (n, #class, h, w)
        img_shapes (list[ndarray]): shape (3, ).
        rcnn_test_cfg (dict): rcnn test config.

    Returns:
        tuple: (bboxes, scores)
    """
    recovered_masks = []
    for mask, img_info in zip(aug_masks, img_metas):
        flip = img_info[0]['flip']
        flip_direction = img_info[0]['flip_direction']
        if flip:
            if flip_direction == 'horizontal':
                mask = mask[:, :, :, ::-1]
            elif flip_direction == 'vertical':
                mask = mask[:, :, ::-1, :]
            elif flip_direction == 'diagonal':
                mask = mask[:, :, :, ::-1]
                mask = mask[:, :, ::-1, :]
            else:
                raise ValueError(
                    f"Invalid flipping direction '{flip_direction}'")
        recovered_masks.append(mask)

    if weights is None:
        merged_masks = np.mean(recovered_masks, axis=0)
    else:
        merged_masks = np.average(
            np.array(recovered_masks), axis=0, weights=np.array(weights))
    return merged_masks

def merge_aug_bboxes_3d(aug_results, img_metas, test_cfg):
    """Merge augmented detection 3D bboxes and scores.

    Args:
        aug_results (list[dict]): The dict of detection results.
            The dict contains the following keys

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.
        img_metas (list[dict]): Meta information of each sample.
        test_cfg (dict): Test config.

    Returns:
        dict: Bounding boxes results in cpu mode, containing merged results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Merged detection bbox.
            - scores_3d (torch.Tensor): Merged detection scores.
            - labels_3d (torch.Tensor): Merged predicted box labels.
    """

    assert len(aug_results) == len(img_metas), \
        '"aug_results" should have the same length as "img_metas", got len(' \
        f'aug_results)={len(aug_results)} and len(img_metas)={len(img_metas)}'

    recovered_bboxes = []
    recovered_scores = []
    recovered_labels = []

    for bboxes, img_info in zip(aug_results, img_metas):
        scale_factor = img_info[0]['pcd_scale_factor']
        pcd_horizontal_flip = img_info[0]['pcd_horizontal_flip']
        pcd_vertical_flip = img_info[0]['pcd_vertical_flip']
        recovered_scores.append(bboxes['scores_3d'])
        recovered_labels.append(bboxes['labels_3d'])
        bboxes = bbox3d_mapping_back(bboxes['boxes_3d'], scale_factor,
                                     pcd_horizontal_flip, pcd_vertical_flip)
        recovered_bboxes.append(bboxes)

    aug_bboxes = recovered_bboxes[0].cat(recovered_bboxes)
    aug_bboxes_for_nms = xywhr2xyxyr(aug_bboxes.bev)
    aug_scores = torch.cat(recovered_scores, dim=0)
    aug_labels = torch.cat(recovered_labels, dim=0)

    # TODO: use a more elegent way to deal with nms
    if test_cfg.use_rotate_nms:
        nms_func = nms_gpu
    else:
        nms_func = nms_normal_gpu

    merged_bboxes = []
    merged_scores = []
    merged_labels = []

    # Apply multi-class nms when merge bboxes
    if len(aug_labels) == 0:
        return bbox3d2result(aug_bboxes, aug_scores, aug_labels)

    for class_id in range(torch.max(aug_labels).item() + 1):
        class_inds = (aug_labels == class_id)
        bboxes_i = aug_bboxes[class_inds]
        bboxes_nms_i = aug_bboxes_for_nms[class_inds, :]
        scores_i = aug_scores[class_inds]
        labels_i = aug_labels[class_inds]
        if len(bboxes_nms_i) == 0:
            continue
        selected = nms_func(bboxes_nms_i, scores_i, test_cfg.nms_thr)

        merged_bboxes.append(bboxes_i[selected, :])
        merged_scores.append(scores_i[selected])
        merged_labels.append(labels_i[selected])

    merged_bboxes = merged_bboxes[0].cat(merged_bboxes)
    merged_scores = torch.cat(merged_scores, dim=0)
    merged_labels = torch.cat(merged_labels, dim=0)

    _, order = merged_scores.sort(0, descending=True)
    num = min(test_cfg.max_num, len(aug_bboxes))
    order = order[:num]

    merged_bboxes = merged_bboxes[order]
    merged_scores = merged_scores[order]
    merged_labels = merged_labels[order]

    return bbox3d2result(merged_bboxes, merged_scores, merged_labels)

