import time
import copy

import torch
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmcv.runner import force_fp32, auto_fp16
from scipy.optimize import linear_sum_assignment
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.GenAD.planner.metric_stp3 import PlanningMetric
import numpy as np
import os
import json
import glob
import pickle
import torch.distributed as dist
import torch.nn.functional as F
import re
from pyquaternion import Quaternion

import mmcv
from pyquaternion import Quaternion
import torch.distributed as dist


# import logging
# logging.basicConfig(filename='/mnt/kuebiko/users/qdeng/GenAD/training.log', level=logging.INFO, format='%(asctime)s - %(message)s')


@DETECTORS.register_module()
class GenAD(MVXTwoStageDetector):
    """GenAD model.
    """
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 fut_ts=6,
                 fut_mode=6
                 ):

        super(GenAD,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.valid_fut_ts = pts_bbox_head['valid_fut_ts']

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

        self.planning_metric = None
        from nuscenes.nuscenes import NuScenes
        self.nusc = NuScenes(version='v1.0-mini', dataroot='/mnt/kuebiko/users/qdeng/GenAD/data-mini/nuscenes', verbose=False)


    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        # img.shape -- [1或2, 6, 3, 384, 640]=[bs*queue,num_cams=6,C=3,H=384,W=640]
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:     # img.shape为 [1, N, C, H, W]的情况
                img.squeeze_()                          # 去掉第一维，变为 [N, C, H, W]
            elif img.dim() == 5 and img.size(0) > 1:    # img.shape为 [B, N, C, H, W]的情况
                B, N, C, H, W = img.size()              # 把B和N合并，变为 [B*N, C, H, W]
                img = img.reshape(B * N, C, H, W)       # N=num_cams
            # 图像增强的一种手段，利用mask遮挡部分图像，让网络学习目标更多的特征，避免过拟合
            if self.use_grid_mask:
                img = self.grid_mask(img)
            
            # 如果检测到 img_feats 是字典，就把它转为列表，便于后面统一处理
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        # 如果定义了img_neck，就把img_feats传入img_neck处理
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        # 重塑img_feats的形状
        img_feats_reshaped = []
        for img_feat in img_feats:          # 遍历 img_feats 中的每一个特征图（可能对应不同层次 / 不同分辨率），分别进行重塑
            BN, C, H, W = img_feat.size()
            if len_queue is not None:       # 如果传入了 len_queue（多帧场景）
                # 在上层调用时，B被reshape成了batch_size * len_queue，B/len_queue还原成真实的batch_size
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          map_gt_bboxes_3d,
                          map_gt_labels_3d,                          
                          img_metas,
                          gt_bboxes_ignore=None,
                          map_gt_bboxes_ignore=None,
                          prev_bev=None,
                          ego_his_trajs=None,
                          ego_fut_trajs=None,
                          ego_fut_masks=None,
                          ego_fut_cmd=None,
                          ego_lcf_feat=None,
                          gt_attr_labels=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """
        # pts_bbox_head在config中的定义是GenADHead这个类
        # 调用GenADHead的forward得到预测结果
        # GenADHead --> DETRHead --> BaseDenseHead --> BaseHead --> torch.nn.Module
        # nn.Module 的 __call__ 方法将调用子类中定义的 forward 方法
        # 从pts_bbox_head进入下一环节BEVFormerHead（包含encoder、decoder）
        outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev,
                                  ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat,
                                  gt_labels_3d=gt_labels_3d, gt_attr_labels=gt_attr_labels,
                                  ego_fut_trajs=ego_fut_trajs)
        loss_inputs = [
            gt_bboxes_3d, gt_labels_3d,                 # detection Head的监督信号
            map_gt_bboxes_3d, map_gt_labels_3d,         # map detection Head的监督信号
            outs,                                       # pts_bbox_head的输出，包含了bev_embed
            ego_fut_trajs, ego_fut_masks, ego_fut_cmd,  # future generation的监督信号
            gt_attr_labels,                             # instance encoder的监督信号?    
        ]

        # bev_features = outs['bev_embed']  # shape: [10000, 1, 256]

        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        # return losses
        return losses, outs

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        # return_loss=True，训练模式，计算损失用于反向传播
        # return_loss=False，测试模式，只进行前向推理不计算损失，双层嵌套用于数据增强，参考MultiScaleFlipAug3D的类定义
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def save_bev_features(self, bev_features, img_metas, bev_h, bev_w, base_path='/mnt/kuebiko/users/qdeng/GenAD/bev_features_pretrain_epoch_0'):
        """保存BEV特征
        Args:
            bev_features (Tensor): [H*W, bs, C] = [10000,1,256]BEV特征图
            img_metas (list): 包含场景信息的字典列表
            bev_h (int): BEV特征图高度
            bev_w (int): BEV特征图宽度
            base_path (str): 保存路径基准目录
        """
        for batch_idx, meta in enumerate(img_metas):
            scene_token = meta['scene_token']
            sample_token = meta['sample_idx']
            
            lidar_file = meta['pts_filename']
            timestamp = re.search(r'__(\d+)\.pcd\.bin$', lidar_file).group(1)
            
            save_dir = os.path.join(base_path, scene_token)
            os.makedirs(save_dir, exist_ok=True)
            
            save_name = f"{sample_token}_{timestamp}.pth"
            save_path = os.path.join(save_dir, save_name)
            
            # 保存特征
            save_dict = {
                'features': bev_features.detach().cpu(),  
                'bev_h': bev_h,
                'bev_w': bev_w,
                'timestamp': timestamp,
                'scene_token': scene_token,
                'sample_token': sample_token
            }
            torch.save(save_dict, save_path)

    # 利用img_queue中除当前帧之外的前2帧生成pre_bev，供后续TSA使用
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval() # # 1) 把BEVFormer 的 backbone、neck、pts_bbox_head 等子模块切换到eval模式，不计算梯度
        # imgs_queue.shape -- [1, 2, 6, 3, 384, 640]=[bs,queue=2,num_cams=6,C=3,H=384,W=640]
        with torch.no_grad():   # 只想做一次前向推断，不更新任何权重，也不改变 BN 的统计量、也不会进行 Dropout
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            # 把批大小（bs）和帧数（len_queue）这两个维度“折叠”到一起，得到 [bs*len_queue, num_cams, C, H, W] 的形式，方便后续一次性送进 Backbone / Neck 做卷积运算。
            # 提取图像特征并reshape
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                # img_feats 按照queue维度进行切片，从6维度降到5维，制作成一个列表
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                # 从pts_bbox_head进入下一环节BEVFormerHead（包含encoder、decoder）
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()    # 2) 恢复到train模式，继续训练
            return prev_bev

    def merge_inference_results(self, base_dir=None, epoch=None, delete_rank_files=True):
        """
        合并所有进程的推理结果文件
        
        Args:
            base_dir: 结果文件的基础目录，如果为None则使用默认路径
            epoch: 训练轮次
            delete_rank_files: 是否在合并后删除各个进程的文件
        """
        
        # 确定目录路径
        if base_dir is None:
            base_dir = os.path.join('/mnt/kuebiko/users/qdeng/GenAD', 'trajectory_inference_format_results')
        
        if epoch is not None:
            result_dir = os.path.join(base_dir, f'epoch_{epoch}')
        else:
            result_dir = base_dir
        
        # 搜索所有进程的结果文件
        rank_files = glob.glob(os.path.join(result_dir, 'results_nusc_rank_*.pkl'))
        
        if not rank_files:
            print(f"未找到任何进程结果文件在 {result_dir}")
            return
        
        # 创建合并后的结果字典
        merged_results = {
            "meta": {
                "use_lidar": False,
                "use_camera": True,
                "use_radar": False,
            },
            "results": {},
            "map_results": {},
            "plan_results": {}
        }
        
        # 逐个加载并合并结果
        total_samples = 0
        for rank_file in rank_files:
            try:
                with open(rank_file, 'rb') as f:
                    rank_results = pickle.load(f)
                
                # 合并结果字典
                merged_results["results"].update(rank_results["results"])
                merged_results["map_results"].update(rank_results["map_results"])
                merged_results["plan_results"].update(rank_results["plan_results"])
                
                total_samples += len(rank_results["results"])
                print(f"从 {rank_file} 加载了 {len(rank_results['results'])} 个样本")
            except Exception as e:
                print(f"加载 {rank_file} 时出错: {e}")
        
        # 保存合并后的结果
        merged_file = os.path.join(result_dir, 'results_nusc.pkl')
        with open(merged_file, 'wb') as f:
            pickle.dump(merged_results, f)
        
        print(f"合并了 {total_samples} 个样本的结果到 {merged_file}")
        
        # 可选：删除各个进程的文件
        if delete_rank_files:
            for rank_file in rank_files:
                try:
                    os.remove(rank_file)
                    print(f"已删除 {rank_file}")
                except Exception as e:
                    print(f"删除 {rank_file} 时出错: {e}")

    # @auto_fp16(apply_to=('img', 'points'))
    @force_fp32(apply_to=('img','points','prev_bev'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      map_gt_bboxes_3d=None,
                      map_gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      map_gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ego_his_trajs=None,
                      ego_fut_trajs=None,
                      ego_fut_masks=None,
                      ego_fut_cmd=None,
                      ego_lcf_feat=None,
                      gt_attr_labels=None
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        #img: (bs,queue/frames,num_cams=6,C=3,H,W）
        len_queue = img.size(1)         # 一次训练/推理时所用到的帧（或时序序列）的数量
        prev_img = img[:, :-1, ...]     # 提取历史帧
        img = img[:, -1, ...]           # 提取当前帧

        # 利用img_queue中除当前帧之外的前几帧生成pre_bev
        prev_img_metas = copy.deepcopy(img_metas)
        # prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        # import pdb;pdb.set_trace()
        # 一次性处理所有历史帧的坐标变换
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas) if len_queue > 1 else None
        # obtain_history_bev用于利用t-2、t-1时刻的图像和img_metas生成pre_bev
        # 然后将当前帧图像特征img_feats、obtain_history_bev生成的prev_bev和当前图像帧对应的img_metas以及bboxes_labels、class_labels输入forward_pts_train计算loss

        # 提取当前帧图像特征
        img_metas = [each[len_queue-1] for each in img_metas]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        # 在forward_pts_train中进入BEVFormerHaed
        # losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d,
        #                                     map_gt_bboxes_3d, map_gt_labels_3d, img_metas,
        #                                     gt_bboxes_ignore, map_gt_bboxes_ignore, prev_bev,
        #                                     ego_his_trajs=ego_his_trajs, ego_fut_trajs=ego_fut_trajs,
        #                                     ego_fut_masks=ego_fut_masks, ego_fut_cmd=ego_fut_cmd,
        #                                     ego_lcf_feat=ego_lcf_feat, gt_attr_labels=gt_attr_labels)
        losses_pts,outs = self.forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d,
                                            map_gt_bboxes_3d, map_gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, map_gt_bboxes_ignore, prev_bev,
                                            ego_his_trajs=ego_his_trajs, ego_fut_trajs=ego_fut_trajs,
                                            ego_fut_masks=ego_fut_masks, ego_fut_cmd=ego_fut_cmd,
                                            ego_lcf_feat=ego_lcf_feat, gt_attr_labels=gt_attr_labels)
        
        # 初始化或获取累积的可视化结果
        if not hasattr(self, 'visualization_results'):
            self.visualization_results = {
                'meta': {
                    'use_lidar': False,
                    'use_camera': True,
                    'use_radar': False,
                    'use_map': False,
                    'use_external': True
                },
                'results': {},
                'map_results': {},
                'plan_results': {},
                'lateral_shift': {}  # 添加偏移量信息

            }
            self.processed_tokens = set()  # 跟踪已处理过的样本token

        # 获取当前迭代次数和epoch
        iter_num = getattr(self, 'iter', 0)
        epoch = getattr(self.pts_bbox_head, 'epoch', 0)

        nuscenes_classes = [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]
        
        NUSCENES_ATTRIBUTE_MAP = {
            # 车辆类
            'car': ['vehicle.moving', 'vehicle.parked', 'vehicle.stopped'],
            'truck': ['vehicle.moving', 'vehicle.parked', 'vehicle.stopped'],
            'bus': ['vehicle.moving', 'vehicle.parked', 'vehicle.stopped'],
            'trailer': ['vehicle.moving', 'vehicle.parked', 'vehicle.stopped'],
            'construction_vehicle': ['vehicle.moving', 'vehicle.parked', 'vehicle.stopped'],
            
            # 行人类
            'pedestrian': ['pedestrian.moving', 'pedestrian.standing', 'pedestrian.sitting_lying_down'],
            
            # 自行车/摩托车类
            'motorcycle': ['cycle.with_rider', 'cycle.without_rider'],
            'bicycle': ['cycle.with_rider', 'cycle.without_rider'],
            
            # 静态物体类 - 这些通常没有属性
            'barrier': [''],
            'traffic_cone': ['']
        }

        # 每隔一定迭代次数保存一次
        visualization_interval = 1  # 可以根据需要调整
        if iter_num % visualization_interval == 0:
            try:
                # 使用pts_bbox_head的get_bboxes方法将outs转换为bbox格式
                bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=False)
                
                # 处理每个样本的结果
                for i, (bboxes, scores, labels, trajs, map_bboxes, 
                        map_scores, map_labels, map_pts) in enumerate(bbox_list):
                    
                    sample_token = img_metas[i]['sample_idx']
                    # 如果该样本已处理过，跳过
                    if sample_token in self.processed_tokens:
                        continue
                    # 标记该样本为已处理
                    self.processed_tokens.add(sample_token)

                    # 使用NuScenes API获取必要的记录
                    # 循环前获取转换参数
                    try:
                        sample_rec = self.nusc.get('sample', sample_token)
                        sd_record = self.nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
                        cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                        pose_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
                        
                        # 保存转换参数供循环内使用
                        lidar2ego_rot = Quaternion(cs_record['rotation'])
                        lidar2ego_trans = np.array(cs_record['translation'])
                        ego2global_rot = Quaternion(pose_record['rotation'])
                        ego2global_trans = np.array(pose_record['translation'])
                        
                        have_transform = True
                    except Exception as e:
                        print(f"坐标转换失败: {e}, 将使用局部坐标...")
                        have_transform = False

                    # 1. 处理3D检测框结果
                    obj_results = []
                    for j in range(len(bboxes)):
                        box = bboxes[j]
                        score = float(scores[j])
                        label = int(labels[j])
                        traj = trajs[j] if trajs is not None else None
                        
                        # 过滤低分检测框
                        if score < 0.3:
                            continue
                        
                        # 处理LiDARInstance3DBoxes对象
                        # LiDARInstance3DBoxes类定义在/mnt/kuebiko/users/qdeng/GenAD/projects/mmdet3d_plugin/core/bbox/structures/box_3d.py文件中
                        if hasattr(box, 'tensor'):
                            # 使用tensor属性获取底层数据
                            box_tensor = box.tensor.cpu().detach().numpy()[0]  # 形状是[1, 9]
                            lidar_translation = box_tensor[:3].tolist()        # 局部坐标系中的translation
                            size = box_tensor[3:6].tolist()
                            original_yaw = float(box_tensor[6])
                        else:
                            lidar_translation = box.center.cpu().detach().numpy().tolist()
                            size = [float(box.width), float(box.length), float(box.height)]
                            original_yaw = float(box.yaw.cpu().detach().numpy())
                        
                        # rotation的计算需要参考/mnt/kuebiko/users/qdeng/GenAD/projects/mmdet3d_plugin/datasets/nuscenes_vad_dataset.py的内容（重点是format_results这个函数）
                        # 转换yaw角为四元数
                        yaw = -(original_yaw + np.pi / 2)  # 调整方向与output_to_nusc_box一致
                        lidar_rotation = Quaternion(axis=[0, 0, 1], radians=yaw)

                        # 在这里执行坐标转换，确保每个框都有正确的全局坐标
                        if have_transform:
                            try:
                                # 从LiDAR坐标系转换到自车坐标系
                                ego_translation = lidar2ego_rot.rotate(lidar_translation) + lidar2ego_trans
                                ego_rotation = lidar2ego_rot * lidar_rotation
                                
                                # 从自车坐标系转换到全局坐标系
                                global_translation = ego2global_rot.rotate(ego_translation) + ego2global_trans
                                global_rotation = ego2global_rot * ego_rotation
                                
                                # 将numpy数组转换为Python列表
                                global_translation_list = global_translation.tolist() if isinstance(global_translation, np.ndarray) else global_translation
                                global_rotation_elements = global_rotation.elements.tolist() if hasattr(global_rotation, 'elements') else global_rotation
                                
                                # # 调试打印
                                # if j == 0:  # 只打印第一个框以减少输出量
                                #     print(f"框 {j} - 局部坐标: {lidar_translation}")
                                #     print(f"框 {j} - 自车坐标: {ego_translation.tolist() if isinstance(ego_translation, np.ndarray) else ego_translation}")
                                #     print(f"框 {j} - 全局坐标: {global_translation_list}")
                            except Exception as e:
                                print(f"框 {j} 坐标转换失败: {e}, 使用局部坐标...")
                                global_translation_list = lidar_translation
                                global_rotation_elements = lidar_rotation.elements.tolist() if hasattr(lidar_rotation, 'elements') else lidar_rotation
                        else:
                            # 如果没有转换参数，使用局部坐标
                            global_translation_list = lidar_translation
                            global_rotation_elements = lidar_rotation.elements.tolist() if hasattr(lidar_rotation, 'elements') else lidar_rotation

                        # 处理速度
                        velocity = [0.0, 0.0]
                        if traj is not None and traj.shape[0] > 0:
                            first_point = traj[0].cpu().detach().numpy()
                            if len(first_point) >= 2:
                                velocity = [float(first_point[0]), float(first_point[1])]
                        
                        is_moving = trajs is not None and torch.sum(torch.abs(trajs[j])) > 0.1
                        class_name = nuscenes_classes[label]
                        
                        # 确定属性
                        if class_name in NUSCENES_ATTRIBUTE_MAP:
                            if class_name in ['car', 'truck', 'bus', 'trailer', 'construction_vehicle']:
                                # 车辆类
                                if is_moving:
                                    attribute = 'vehicle.moving'
                                else:
                                    attribute = 'vehicle.parked'  # 或'vehicle.stopped'，根据具体情况选择
                            elif class_name == 'pedestrian':
                                # 行人类
                                if is_moving:
                                    attribute = 'pedestrian.moving'
                                else:
                                    attribute = 'pedestrian.standing'
                            elif class_name in ['motorcycle', 'bicycle']:
                                # 自行车/摩托车类
                                attribute = 'cycle.with_rider'  # 默认假设有骑手
                            else:
                                # 静态物体或其他类别
                                attribute = ''
                        else:
                            # 未知类别
                            attribute = ''

                        obj_result = {
                            'sample_token': sample_token,
                            'translation': global_translation_list,  # 使用全局坐标系中的位置
                            'size': size,
                            'rotation': global_rotation_elements,  # 使用全局坐标系中的旋转
                            'velocity': velocity,
                            'detection_name': nuscenes_classes[label],
                            'detection_score': score,
                            'attribute_name': attribute,
                            'num_pts': -1,
                            'fut_traj': trajs[j].cpu().detach().numpy().tolist(),
                        }
                        obj_results.append(obj_result)
                    
                    # 将结果添加到全局字典
                    self.visualization_results['results'][sample_token] = obj_results
                                        
                    # 2. 处理地图结果
                    map_vectors = []
                    map_types = ['divider', 'ped_crossing', 'boundary']

                    for j in range(len(map_pts)):
                        pts = map_pts[j].cpu().detach().numpy()
                        map_score = float(map_scores[j])
                        map_label = int(map_labels[j])
                        
                        # 过滤低分地图元素
                        if map_score < 0.6:
                            continue
                            
                        map_vector = {
                            'pts': pts,
                            'pts_num': pts.shape[0],            # 添加测试数据中存在的这个字段
                            'cls_name': map_types[map_label],   # 添加测试数据中存在的这个字段
                            'type': map_label,
                            'confidence_level': map_score
                        }
                        map_vectors.append(map_vector)
                    
                    # 将地图结果添加到全局字典
                    self.visualization_results['map_results'][sample_token] = {
                        'sample_token': sample_token,
                        'vectors': map_vectors
                    }        

                    # 3. 处理规划结果
                    if 'ego_fut_preds' in outs:
                         # 创建cmd默认元素
                        default_cmd = torch.zeros((1, 1, 1, 3), device='cuda')
                        default_cmd[0, 0, 0, 2] = 1.0  # 默认为"Go Straight"
                        self.visualization_results['plan_results'][sample_token] = [
                            outs['ego_fut_preds'][i].detach().cpu(),  
                            default_cmd.detach().cpu()
                        ]
                
                    # 4. 保存lateral_shift信息
                    if 'lateral_shift' in outs:
                        print(f"outs['lateral_shift'] = {outs['lateral_shift']}")
                        if 'lateral_shift' not in self.visualization_results:
                            self.visualization_results['lateral_shift'] = {}
                        self.visualization_results['lateral_shift'][sample_token] = outs['lateral_shift'][i].detach().cpu()

                # 保存结果
                # 定期显示累积进度
                if iter_num % 10 == 0:
                    print(f"Visualization data collection: {len(self.processed_tokens)} samples so far")
                
                # 当收集到足够多的样本或达到特定迭代次数时保存
                save_interval = 1  # 每1次迭代保存一次
                save_path = '/mnt/kuebiko/users/qdeng/GenAD/visualization/training_pkl_shift_1'
                os.makedirs(save_path, exist_ok=True)
                if iter_num % save_interval == 0:
                    # === 多GPU分布式保存 ===
                    # 从所有GPU收集数据
                    world_size = torch.distributed.get_world_size()
                    rank = torch.distributed.get_rank()

                    # 创建包含当前GPU数据的对象
                    local_results = {
                        'results': self.visualization_results['results'].copy(),
                        'map_results': self.visualization_results['map_results'].copy(),
                        'plan_results': self.visualization_results['plan_results'].copy(),
                        'lateral_shift': self.visualization_results['lateral_shift'].copy()  
                    }

                    # 收集所有GPU的数据
                    all_results = [None for _ in range(world_size)]
                    torch.distributed.all_gather_object(all_results, local_results)
                    
                    # 仅在主进程(rank 0)合并和保存数据
                    if rank == 0:
                        # 合并所有GPU的结果
                        combined_results = {
                            'meta': self.visualization_results['meta'],
                            'results': {},
                            'map_results': {},
                            'plan_results': {},
                            'lateral_shift': {}
                        }
                        
                        for results in all_results:
                            combined_results['results'].update(results['results'])
                            combined_results['map_results'].update(results['map_results'])
                            combined_results['plan_results'].update(results['plan_results'])
                            combined_results['lateral_shift'].update(results['lateral_shift'])  # 添加这一行
                        
                        save_file = os.path.join(save_path, f'iter_{iter_num}.pkl')
                        mmcv.dump(combined_results, save_file)
                        # mmcv.dump(self.visualization_results, save_file)
                        print(f'Visualization results with {len(self.processed_tokens)} samples saved to {save_file}')
                    
                    # 在这里添加一个分布式同步点，确保所有进程等待主进程完成保存
                    torch.distributed.barrier()

            except Exception as e:
                print(f"Error in visualization: {e}")
                import traceback
                print(traceback.format_exc())

        # 记录当前迭代次数
        if not hasattr(self, 'iter'):
            self.iter = 0
        self.iter += 1

        # # 提取并保存BEV嵌入
        # if 'bev_embed' in outs and epoch == 0:
        #     self.save_bev_features(
        #         outs['bev_embed'], 
        #         img_metas, 
        #         self.pts_bbox_head.bev_h, 
        #         self.pts_bbox_head.bev_w
        #     )

        # logging.info(f"{outs}: {outs}")
        losses.update(losses_pts)

        # # 保存轨迹信息
        # self.save_inference_format_results(
        #     outs,
        #     img_metas,
        #     gt_bboxes_3d,
        #     gt_labels_3d,
        #     gt_attr_labels,
        #     ego_fut_trajs,
        #     ego_fut_masks,
        #     ego_fut_cmd,
        #     epoch=epoch
        # )
    
        # self.save_trajectory_info(
        #     outs,
        #     img_metas,
        #     gt_bboxes_3d,
        #     gt_labels_3d,
        #     gt_attr_labels,
        #     ego_fut_trajs,
        #     ego_fut_masks,
        #     ego_fut_cmd,
        #     epoch=epoch
        # )
        
        return losses

    def forward_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        img=None,
        ego_his_trajs=None,     # 历史轨迹
        ego_fut_trajs=None,     # 未来轨迹
        ego_fut_cmd=None,       # 控制命令
        ego_lcf_feat=None,
        gt_attr_labels=None,
        **kwargs
    ):
        # 检查img_metas是否为list类型
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        # 如果img为None,将其转换为list
        img = [img] if img is None else img

        # 场景切换检测
        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            # 如果是新场景，清空历史BEV
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # 时序处理控制
        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # 需要逐帧处理
        # Get the delta of ego position and angle between two timestamps.
        # 位置和角度增量计算
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        # 如有上一帧,计算相对增量;否则置零
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas=img_metas[0],
            img=img[0],
            prev_bev=self.prev_frame_info['prev_bev'],
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            ego_his_trajs=ego_his_trajs[0],
            ego_fut_trajs=ego_fut_trajs[0],
            ego_fut_cmd=ego_fut_cmd[0],
            ego_lcf_feat=ego_lcf_feat[0],
            gt_attr_labels=gt_attr_labels,
            **kwargs
        )
        # 状态更新
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev

        return bbox_results

    # 训练过程中执行模型前向传播并生成预测结果（执行训练脚本时如果不是no-validate模式，会调用此函数）
    def simple_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        img=None,
        prev_bev=None,
        points=None,
        fut_valid_flag=True,
        rescale=False,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        **kwargs
    ):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]     # 初始化bbox_list列表，为每个img_metas创建空字典存储结果
        new_prev_bev, bbox_pts, metric_dict = self.simple_test_pts(
            img_feats,
            img_metas,
            gt_bboxes_3d,
            gt_labels_3d,
            prev_bev,
            fut_valid_flag=fut_valid_flag,
            rescale=rescale,
            start=None,
            ego_his_trajs=ego_his_trajs,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
            ego_lcf_feat=ego_lcf_feat,
            gt_attr_labels=gt_attr_labels,
        )


        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):  # 更新bbox_list
            # result_dict: 来自bbox_list的字典元素
            # pts_bbox: 来自bbox_pts的对应元素
            result_dict['pts_bbox'] = pts_bbox                  # 将预测结果存入result_dict
            result_dict['metric_results'] = metric_dict         # 将评估指标存入result_dict
        
        return new_prev_bev, bbox_list

    # 单帧预测
    def simple_test_pts(
        self,
        x,              # img_feats
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        prev_bev=None,
        fut_valid_flag=True,
        rescale=False,
        start=None,
        ego_his_trajs=None,     # 自车的历史轨迹
        ego_fut_trajs=None,     # 自车的未来轨迹
        ego_fut_cmd=None,       # 自车未来的控制命令
        ego_lcf_feat=None,      # 与自车局部坐标框架相关的特征
        gt_attr_labels=None,
    ):
        """Test function"""
        mapped_class_names = [
            'car', 'truck', 'construction_vehicle', 'bus',
            'trailer', 'barrier', 'motorcycle', 'bicycle', 
            'pedestrian', 'traffic_cone'
        ]

        ### 1.生成预测结果

        # 测试模式不涉及GenADHead的梯度计算
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev,
                                  ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)

        bbox_results = []
        # 遍历 bbox_list，处理每帧预测结果
        for i, (bboxes, scores, labels, trajs, map_bboxes, \
                map_scores, map_labels, map_pts) in enumerate(bbox_list):
            # 将 GPU 上的 3D 检测结果转移到 CPU，并组织成一个统一的字典格式
            bbox_result = bbox3d2result(bboxes, scores, labels)
            bbox_result['trajs_3d'] = trajs.cpu()
            map_bbox_result = self.map_pred2result(map_bboxes, map_scores, map_labels, map_pts)
            bbox_result.update(map_bbox_result)
            bbox_result['ego_fut_preds'] = outs['ego_fut_preds'][i].cpu()
            bbox_result['ego_fut_cmd'] = ego_fut_cmd.cpu()
            bbox_results.append(bbox_result)

        assert len(bbox_results) == 1, 'only support batch_size=1 now'
        # 后处理
        score_threshold = 0.6
        with torch.no_grad():
            c_bbox_results = copy.deepcopy(bbox_results)
            # 因为是单帧预测，所以只有一个元素，索引都为0
            bbox_result = c_bbox_results[0]                     # 一帧预测的所有3D边界框信息
            # gt_bbox_3d是一个嵌套列表，第一层索引batch，第二层索引frame
            gt_bbox = gt_bboxes_3d[0][0]                        # 一帧真实标注的所有3D边界框信息
            gt_label = gt_labels_3d[0][0].to('cpu')             # 一帧真实标注的所有3D边界框标签
            # gt_attr_labels提供了更详细的属性标签
            # e.g. 物体的运动状态(静止/运动)
            # e.g. 车辆的特定属性(尺寸、颜色等)
            # e.g. 行人的特定属性(姿态、行为等)
            gt_attr_label = gt_attr_labels[0][0].to('cpu')      # 一帧真实标注的所有3D边界框属性标签
            fut_valid_flag = bool(fut_valid_flag[0][0])         # 未来轨迹有效性标志
            # filter pred bbox by score_threshold
            # 假设scores_3d = [0.8, 0.3, 0.7, 0.5, 0.9]
            # score_threshold = 0.6
            # 则生成的mask = [True, False, True, False, True]
            mask = bbox_result['scores_3d'] > score_threshold   # 过滤低分检测框
            bbox_result['boxes_3d'] = bbox_result['boxes_3d'][mask]
            bbox_result['scores_3d'] = bbox_result['scores_3d'][mask]
            bbox_result['labels_3d'] = bbox_result['labels_3d'][mask]
            bbox_result['trajs_3d'] = bbox_result['trajs_3d'][mask]

            # 将预测结果与 Ground Truth 进行匹配
            matched_bbox_result = self.assign_pred_to_gt_vip3d(
                bbox_result, gt_bbox, gt_label)
            
            ### 2.计算评估指标
            # 计算运动相关指标
            metric_dict = self.compute_motion_metric_vip3d(
                gt_bbox, gt_label, gt_attr_label, bbox_result,
                matched_bbox_result, mapped_class_names,img_metas[0])

            # 自车规划评估
            # ego planning metric
            assert ego_fut_trajs.shape[0] == 1, 'only support batch_size=1 for testing'
            ego_fut_preds = bbox_result['ego_fut_preds']        # 自车预测未来轨迹
            ego_fut_trajs = ego_fut_trajs[0, 0]                 # 自车真实未来轨迹
            ego_fut_cmd = ego_fut_cmd[0, 0, 0]                  # 自车控制命令
            ego_fut_cmd_idx = torch.nonzero(ego_fut_cmd)[0, 0]  # 自车控制命令索引
            ego_fut_pred = ego_fut_preds[ego_fut_cmd_idx]       # 自车预测未来轨迹
            ego_fut_pred = ego_fut_pred.cumsum(dim=-2)          # 自车预测未来轨迹累积和
            ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)        # 自车真实未来轨迹累积和

            # 计算规划相关指标
            metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                pred_ego_fut_trajs = ego_fut_pred[None],
                gt_ego_fut_trajs = ego_fut_trajs[None],
                gt_agent_boxes = gt_bbox,
                gt_agent_feats = gt_attr_label.unsqueeze(0),
                fut_valid_flag = fut_valid_flag
            )
            metric_dict.update(metric_dict_planner_stp3)

        return outs['bev_embed'], bbox_results, metric_dict

    def map_pred2result(self, bboxes, scores, labels, pts, attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        # 将预测结果转移到 CPU
        result_dict = dict(
            map_boxes_3d=bboxes.to('cpu'),      # 3D边界框
            map_scores_3d=scores.cpu(),         # 预测分数
            map_labels_3d=labels.cpu(),         # 边界框标签
            map_pts_3d=pts.to('cpu'))           # 3D点云

        if attrs is not None:
            result_dict['map_attrs_3d'] = attrs.cpu()

        return result_dict

    def assign_pred_to_gt_vip3d(
        self,
        bbox_result,
        gt_bbox,
        gt_label,
        match_dis_thresh=2.0
    ):
        """Assign pred boxs to gt boxs according to object center preds in lcf.
        Args:
            bbox_result (dict): Predictions.
                'boxes_3d': (LiDARInstance3DBoxes)
                'scores_3d': (Tensor), [num_pred_bbox]
                'labels_3d': (Tensor), [num_pred_bbox]
                'trajs_3d': (Tensor), [fut_ts*2]
            gt_bboxs (LiDARInstance3DBoxes): GT Bboxs.
            gt_label (Tensor): GT labels for gt_bbox, [num_gt_bbox].
            match_dis_thresh (float): dis thresh for determine a positive sample for a gt bbox.

        Returns:
            matched_bbox_result (np.array): assigned pred index for each gt box [num_gt_bbox].
        """     
        dynamic_list = [0,1,3,4,6,7,8]      # 定义动态物体标签列表
        # trailer不是自主运动的物体，它的运动完全依赖于牵引车辆
        # 初始化匹配结果数组，这个数组的长度等于gt_bbox的数量
        matched_bbox_result = torch.ones(
            (len(gt_bbox)), dtype=torch.long) * -1  # -1: not assigned
        
        # 获取边界框的中心点坐标
        gt_centers = gt_bbox.center[:, :2]                      # 真实框的xy中心坐标
        pred_centers = bbox_result['boxes_3d'].center[:, :2]    # 预测框的xy中心坐标
        
        # 计算距离矩阵
        dist = torch.linalg.norm(pred_centers[:, None, :] - gt_centers[None, :, :], dim=-1)
        # dist的第一维度是预测框的数量，第二维度是真实框的数量
        pred_not_dyn = [label not in dynamic_list for label in bbox_result['labels_3d']]
        gt_not_dyn = [label not in dynamic_list for label in gt_label]
        
        # 处理静态物体
        # 只需要预测动态物体的未来轨迹，静态物体的轨迹预测没有实际意义
        dist[pred_not_dyn] = 1e6    # 对dist的第一维（预测框维度）进行索引
        dist[:, gt_not_dyn] = 1e6   # 对dist的第二维（真实框维度）进行索引
        dist[dist > match_dis_thresh] = 1e6

        # 使用匈牙利算法进行匹配
        r_list, c_list = linear_sum_assignment(dist)
        # r_list: 预测框的索引
        # c_list: 真实框的索引
        # 记录有效匹配
        for i in range(len(r_list)):
            if dist[r_list[i], c_list[i]] <= match_dis_thresh:
                matched_bbox_result[c_list[i]] = r_list[i]
                # matched_bbox_result[i]表示第i个真实框匹配到了哪个预测框（-1表示没匹配到）

        return matched_bbox_result

    def compute_motion_metric_vip3d(
        self,
            gt_bbox: object,
            gt_label: object,
            gt_attr_label: object,
            pred_bbox: object,
            matched_bbox_result: object,
            mapped_class_names: object,
            img_metas: object,
            match_dis_thresh: object = 2.0,
    ) -> object:
        """Compute EPA metric for one sample.
        Args:
            gt_bboxs (LiDARInstance3DBoxes): GT Bboxs.
            gt_label (Tensor): GT labels for gt_bbox, [num_gt_bbox].
            pred_bbox (dict): Predictions.
                'boxes_3d': (LiDARInstance3DBoxes)
                'scores_3d': (Tensor), [num_pred_bbox]
                'labels_3d': (Tensor), [num_pred_bbox]
                'trajs_3d': (Tensor), [fut_ts*2]
            matched_bbox_result (np.array): assigned pred index for each gt box [num_gt_bbox].
            match_dis_thresh (float): dis thresh for determine a positive sample for a gt bbox.

        Returns:
            EPA_dict (dict): EPA metric dict of each cared class.
        """
        # 这些指标会输出在测试结果中,e.x. EPA_car/ADE_pedestrian/...
        motion_cls_names = ['car', 'pedestrian']                    # 定义需要计算运动指标的类别
        motion_metric_names = ['gt', 'cnt_ade', 'cnt_fde', 'hit',   # 定义需要计算的指标名称
                               'fp', 'ADE', 'FDE', 'MR']
        
        # 初始化存储各类别各运动指标的字典
        metric_dict = {}
        for met in motion_metric_names:
            for cls in motion_cls_names:
                metric_dict[met+'_'+cls] = 0.0



        # mapped_class_names = [
        #     'car', 'truck', 'construction_vehicle', 'bus',
        #     'trailer', 'barrier', 'motorcycle', 'bicycle', 
        #     'pedestrian', 'traffic_cone'
        # ]
        # ignore_list = ['construction_vehicle', 'barrier',
        #                'traffic_cone', 'motorcycle', 'bicycle']
        veh_list = [0, 1, 2, 3, 4, 6, 7]            # 将多种车辆类型统一映射为car(0)
        # 虽然 trailer 不自主运动，但它会随牵引车辆移动，有motion变化
        ignore_list = ['barrier', 'traffic_cone']   # 忽略的类别

        # 遍历一个sample的所有预测框
        for i in range(pred_bbox['labels_3d'].shape[0]):
            # 统一车辆类别
            pred_bbox['labels_3d'][i] = 0 if pred_bbox['labels_3d'][i] in veh_list else pred_bbox['labels_3d'][i]
            box_name = mapped_class_names[pred_bbox['labels_3d'][i]]
            if box_name in ignore_list:
                continue
            if i not in matched_bbox_result:          # 统计误检(False Positive)
                metric_dict['fp_'+box_name] += 1

        # 初始化轨迹数据字典，使用img_metas中的信息
        trajectory_data = {
            'sample_token': img_metas['sample_idx'],     
            'scene_token': img_metas['scene_token'],
            'timestamp': img_metas.get('timestamp', None),
            'prev_idx': img_metas['prev_idx'],          
            'next_idx': img_metas['next_idx'],
            'filename': img_metas['filename'],         
            'pts_filename': img_metas['pts_filename'], 
            'can_bus': img_metas['can_bus'].tolist() if isinstance(img_metas['can_bus'], np.ndarray) else img_metas['can_bus'],
            'trajectories': []
        }

        # 创建保存路径
        save_root = '/mnt/kuebiko/users/qdeng/GenAD/trajs_test'
        scene_dir = os.path.join(save_root, str(trajectory_data['scene_token']))
        sample_dir = os.path.join(scene_dir, str(trajectory_data['sample_token']))
        traj_points_dir = os.path.join(sample_dir, 'traj_points')
        traj_bev_dir = os.path.join(sample_dir, 'traj_BEV')
        # 创建每个sample的目录
        os.makedirs(traj_points_dir, exist_ok=True)
        os.makedirs(traj_bev_dir, exist_ok=True)


        # 遍历一个sample的所有真实框
        for i in range(gt_label.shape[0]):
            # 统一车辆类别
            gt_label[i] = 0 if gt_label[i] in veh_list else gt_label[i]
            box_name = mapped_class_names[gt_label[i]]
            if box_name in ignore_list:
                continue

            # 获取未来轨迹的有效掩码
            # 可能因为：目标离开摄像机视野/被遮挡/跟踪丢失/标注缺失等，并不是所有目标在所有未来时间步都有有效轨迹
            # gt_attr_label 的数据结构为：
            # [0:self.fut_ts*2]：未来轨迹坐标 (x,y pairs)
            # [self.fut_ts*2:self.fut_ts*3]：掩码值 (masks)
            # fut_ts=6
            gt_fut_masks = gt_attr_label[i][self.fut_ts*2:self.fut_ts*3]
            num_valid_ts = sum(gt_fut_masks==1)         # 统计有效时间步

            # 如果所有未来时间步都有效，增加gt计数
            if num_valid_ts == self.fut_ts:     # 检查是否所有预测时间步都有效
                metric_dict['gt_'+box_name] += 1
            
            # *如果有匹配的预测框且存在有效的未来轨迹，计算ADE*
            if matched_bbox_result[i] >= 0 and num_valid_ts > 0:
                metric_dict['cnt_ade_'+box_name] += 1
                # 获取匹配的预测框索引
                m_pred_idx = matched_bbox_result[i]

                # 提取真实轨迹和预测轨迹
                gt_fut_trajs = gt_attr_label[i][:self.fut_ts*2].reshape(-1, 2)
                gt_fut_trajs = gt_fut_trajs[:num_valid_ts]
                # 提取预测轨迹
                # fut_mode表示预测了多种可能的轨迹，fut_mode=6
                pred_fut_trajs = pred_bbox['trajs_3d'][m_pred_idx].reshape(self.fut_mode, self.fut_ts, 2)
                pred_fut_trajs = pred_fut_trajs[:, :num_valid_ts, :]
                
                # 计算累积轨迹并加上当前位置
                # cumsum计算累积和，将相对位移转换为相对于起始点的位置
                gt_fut_trajs = gt_fut_trajs.cumsum(dim=-2)
                pred_fut_trajs = pred_fut_trajs.cumsum(dim=-2)
                # 将相对于起点的位移转换为场景中的绝对坐标
                gt_fut_trajs = gt_fut_trajs + gt_bbox[i].center[0, :2]
                pred_fut_trajs = pred_fut_trajs + pred_bbox['boxes_3d'][int(m_pred_idx)].center[0, :2]

                # 计算预测轨迹和真实轨迹之间的距离
                dist = torch.linalg.norm(gt_fut_trajs[None, :, :] - pred_fut_trajs, dim=-1)
                # 计算ADE
                ade = dist.sum(-1) / num_valid_ts
                ade = ade.min()         # 在多个预测轨迹中选择最小的ADE
                metric_dict['ADE_'+box_name] += ade

                # 保存轨迹点信息
                # traj_id = len(os.listdir(traj_points_dir)) + 1
                traj_id = i + 1  # 直接使用真实框的索引+1作为轨迹ID
                traj_points_save_dir = os.path.join(traj_points_dir, f'traj{traj_id}')
                os.makedirs(traj_points_save_dir, exist_ok=True)
                
                # 将轨迹点信息保存为.json格式
                traj_data = {
                    'gt_trajectory': gt_fut_trajs.cpu().numpy().tolist(),
                    'pred_trajectories': pred_fut_trajs.cpu().numpy().tolist(),
                    'ade': float(ade.item()),  # 确保是普通的Python float类型
                    'point_errors': dist.min(dim=0)[0].cpu().numpy().tolist(),
                    'object_type': box_name,
                    'object_id': int(i),
                    'valid_mask': gt_fut_masks.cpu().numpy().tolist(),
                    'num_valid_ts': int(num_valid_ts)
                }

                with open(os.path.join(traj_points_save_dir, 'trajectories.json'), 'w') as f:
                    json.dump(traj_data, f, indent=4)  # indent=4使json文件格式化展示,更易读


                # # 将轨迹点信息保存为.npz格式
                # np.savez(os.path.join(traj_points_save_dir, 'trajectories.npz'),
                #         gt_trajectory=gt_fut_trajs.cpu().numpy(),
                #         pred_trajectories=pred_fut_trajs.cpu().numpy(),
                #         ade=ade.item(),
                #         point_errors=dist.min(dim=0)[0].cpu().numpy(),
                #         object_type=box_name,
                #         object_id=i)
                
                # 保存BEV可视化所需信息
                traj_bev_save_dir = os.path.join(traj_bev_dir, f'traj{traj_id}')
                os.makedirs(traj_bev_save_dir, exist_ok=True)

                # 先将边界框转换为张量
                gt_bbox_tensor = gt_bbox.tensor.cpu().numpy()  # 获取所有真实框的张量表示
                pred_bbox_tensor = pred_bbox['boxes_3d'].tensor.cpu().numpy()  # 获取所有预测框的张量表示

                # 保存BEV信息为.json格式
                bev_data = {
                    'ego_pose': trajectory_data['can_bus'] if isinstance(trajectory_data['can_bus'], list) 
                                else trajectory_data['can_bus'].tolist(),
                    'gt_bbox': gt_bbox_tensor[i].tolist(),  
                    'pred_bbox': pred_bbox_tensor[int(m_pred_idx)].tolist(),
                    'gt_trajectory': gt_fut_trajs.cpu().numpy().tolist(),
                    'pred_trajectories': pred_fut_trajs.cpu().numpy().tolist()
                }

                with open(os.path.join(traj_bev_save_dir, 'bev_info.json'), 'w') as f:
                    json.dump(bev_data, f, indent=4)

                # # 保存BEV信息为.npz格式
                # np.savez(os.path.join(traj_bev_save_dir, 'bev_info.npz'),
                #         ego_pose=trajectory_data['can_bus'],
                #         gt_bbox=gt_bbox_tensor[i],  # 保存真实边界框信息
                #         pred_bbox=pred_bbox_tensor[int(m_pred_idx)],  # 保存预测边界框信息
                #         gt_trajectory=gt_fut_trajs.cpu().numpy(),
                #         pred_trajectories=pred_fut_trajs.cpu().numpy())

                # 如果所有时间步都有效，计算最终位移误差(FDE)和命中率(Hit)
                if num_valid_ts == self.fut_ts:
                    fde = dist[:, -1].min()
                    metric_dict['cnt_fde_'+box_name] += 1
                    metric_dict['FDE_'+box_name] += fde
                    if fde <= match_dis_thresh:
                        metric_dict['hit_'+box_name] += 1
                    else:
                        metric_dict['MR_'+box_name] += 1

        return metric_dict

    ### same planning metric as stp3
    # 计算一个sample的规划指标
    def compute_planner_metric_stp3(
        self,
        pred_ego_fut_trajs,
        gt_ego_fut_trajs,
        gt_agent_boxes,         # gt_bbox
        gt_agent_feats,         # gt_attr_label
        fut_valid_flag
    ):
        """Compute planner metric for one sample same as stp3."""
        metric_dict = {
            'plan_L2_1s':0,
            'plan_L2_2s':0,
            'plan_L2_3s':0,
            'plan_obj_col_1s':0,
            'plan_obj_col_2s':0,
            'plan_obj_col_3s':0,
            'plan_obj_box_col_1s':0,
            'plan_obj_box_col_2s':0,
            'plan_obj_box_col_3s':0,
        }
        metric_dict['fut_valid_flag'] = fut_valid_flag
        future_second = 3
        assert pred_ego_fut_trajs.shape[0] == 1, 'only support bs=1'

        # 初始化规划指标类
        if self.planning_metric is None:
            self.planning_metric = PlanningMetric()
        # 获取车辆和行人分割标签，生成时序BEV占用图
        segmentation, pedestrian = self.planning_metric.get_label(
            gt_agent_boxes, gt_agent_feats)
        # segmentation: 表示场景中车辆的占用情况的二值化张量
        # pedestrian: 表示场景中行人的占用情况的二值化张量
        # segmentation和pedestrian都是从鸟瞰图视角下获得的占用标签，而不是简单的类别标签。
        # 计算总体占用情况，鸟瞰图视角的二值化的占用图(0表示空闲，1表示被占用)
        occupancy = torch.logical_or(segmentation, pedestrian)
        
        # 对每一个未来时间步进行评估（1s, 2s, 3s）
        for i in range(future_second):
            # 有效轨迹的评估
            if fut_valid_flag:
                cur_time = (i+1)*2
                # 计算L2距离
                # ADE
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )
                # FDE
                traj_L2_stp3 = self.planning_metric.compute_L2_stp3(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )
                # 计算碰撞
                # 自车lidar系为轨迹参考系
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(),
                    gt_ego_fut_trajs[:, :cur_time],
                    occupancy)
                # 存储各项指标
                metric_dict['plan_L2_{}s'.format(i+1)] = traj_L2
                metric_dict['plan_L2_stp3_{}s'.format(i+1)] = traj_L2_stp3
                metric_dict['plan_obj_col_{}s'.format(i+1)] = obj_coll.mean().item()
                metric_dict['plan_obj_col_stp3_{}s'.format(i + 1)] = obj_coll[-1].item()
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = obj_box_coll.mean().item()
                metric_dict['plan_obj_box_col_stp3_{}s'.format(i + 1)] = obj_box_coll[-1].item()
                # if (i == 0):
                #     metric_dict['plan_1'] = obj_box_coll[0].item()
                #     metric_dict['plan_2'] = obj_box_coll[1].item()
                # if (i == 1):
                #     metric_dict['plan_3'] = obj_box_coll[2].item()
                #     metric_dict['plan_4'] = obj_box_coll[3].item()
                # if (i == 2):
                #     metric_dict['plan_5'] = obj_box_coll[4].item()
                #     metric_dict['plan_6'] = obj_box_coll[5].item()
            # 如果轨迹无效，所有指标置为0
            else:
                metric_dict['plan_L2_{}s'.format(i+1)] = 0.0
                metric_dict['plan_L2_stp3_{}s'.format(i + 1)] = 0.0
                metric_dict['plan_obj_col_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = 0.0
            
        return metric_dict

    def set_epoch(self, epoch): 
        self.pts_bbox_head.epoch = epoch