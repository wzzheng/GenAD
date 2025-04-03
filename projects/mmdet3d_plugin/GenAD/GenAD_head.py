import copy
from math import pi, cos, sin

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads import DETRHead
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import build_assigner, build_sampler
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from projects.mmdet3d_plugin.GenAD.utils.traj_lr_warmup import get_traj_warmup_loss_weight
from projects.mmdet3d_plugin.GenAD.utils.map_utils import (
    normalize_2d_pts, normalize_2d_bbox, denormalize_2d_pts, denormalize_2d_bbox
)

from projects.mmdet3d_plugin.GenAD.generator import DistributionModule, PredictModel
from projects.mmdet3d_plugin.GenAD.generator import FuturePrediction

import os
import json
import numpy as np
import re
import random
from datetime import datetime
import matplotlib.pyplot as plt
import math
import wandb

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_unit, verbose=False):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class LaneNet(nn.Module):
    def __init__(self, in_channels, hidden_unit, num_subgraph_layers):
        super(LaneNet, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(
                f'lmlp_{i}', MLP(in_channels, hidden_unit))
            in_channels = hidden_unit * 2

    def forward(self, pts_lane_feats):
        '''
            Extract lane_feature from vectorized lane representation

        Args:
            pts_lane_feats: [batch size, max_pnum, pts, D]

        Returns:
            inst_lane_feats: [batch size, max_pnum, D]
        '''
        x = pts_lane_feats
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                # x [bs,max_lane_num,9,dim]
                x = layer(x)
                x_max = torch.max(x, -2)[0]
                x_max = x_max.unsqueeze(2).repeat(1, 1, x.shape[2], 1)
                x = torch.cat([x, x_max], dim=-1)
        x_max = torch.max(x, -2)[0]
        return x_max


@HEADS.register_module()
class GenADHead(DETRHead):
    """Head of VAD model.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 fut_ts=6,
                 fut_mode=6,
                 loss_traj=dict(type='L1Loss', loss_weight=0.25),
                 loss_traj_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=0.8),
                 map_bbox_coder=None,
                 map_num_query=900,
                 map_num_classes=3,
                 map_num_vec=20,
                 map_num_pts_per_vec=2,
                 map_num_pts_per_gt_vec=2,
                 map_query_embed_type='all_pts',
                 map_transform_method='minmax',
                 map_gt_shift_pts_pattern='v0',
                 map_dir_interval=1,
                 map_code_size=None,
                 map_code_weights=None,
                 loss_map_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_map_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_map_iou=dict(type='GIoULoss', loss_weight=2.0),
                 loss_map_pts=dict(
                     type='ChamferDistance', loss_src_weight=1.0, loss_dst_weight=1.0
                 ),
                 loss_map_dir=dict(type='PtsDirCosLoss', loss_weight=2.0),
                 loss_vae_gen=dict(type='ProbabilisticLoss', loss_weight=1.0),
                 tot_epoch=None,
                 use_traj_lr_warmup=False,
                 motion_decoder=None,
                 motion_map_decoder=None,
                 use_pe=False,
                 motion_det_score=None,
                 map_thresh=0.5,
                 dis_thresh=0.2,
                 pe_normalization=True,
                 ego_his_encoder=None,
                 ego_fut_mode=3,
                 loss_plan_reg=dict(type='L1Loss', loss_weight=0.25),
                 loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=0.1),
                 loss_plan_col=dict(type='PlanAgentDisLoss', loss_weight=0.1),
                 loss_plan_dir=dict(type='PlanMapThetaLoss', loss_weight=0.1),
                 ego_agent_decoder=None,
                 ego_map_decoder=None,
                 query_thresh=None,
                 query_use_fix_pad=None,
                 ego_lcf_feat_idx=None,
                 valid_fut_ts=6,
                 agent_dim=300,
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.tot_epoch = tot_epoch
        self.use_traj_lr_warmup = use_traj_lr_warmup
        self.motion_decoder = motion_decoder
        self.motion_map_decoder = motion_map_decoder
        self.use_pe = use_pe
        self.motion_det_score = motion_det_score
        self.map_thresh = map_thresh
        self.dis_thresh = dis_thresh
        self.pe_normalization = pe_normalization
        self.ego_his_encoder = ego_his_encoder
        self.ego_fut_mode = ego_fut_mode
        self.query_thresh = query_thresh
        self.query_use_fix_pad = query_use_fix_pad
        self.ego_lcf_feat_idx = ego_lcf_feat_idx
        self.valid_fut_ts = valid_fut_ts
        self.agent_dim = agent_dim
        self.with_cur = True

        # 初始化偏移相关属性
        self.shift_log = []
        self.scene_shifts = {}
        self.shift_file = os.path.join('/mnt/kuebiko/users/qdeng/GenAD', 'scene_shifts.json')
        self.shift_rng = np.random.RandomState(42)  # 固定种子确保可重复性
        
        self.visualization_counter = 0
        self.save_data_freq = 1  # 保存数据的频率
        self.visualize_freq = 1

        # if not hasattr(wandb, 'run') or wandb.run is None:
        #     wandb.init(
        #         project="GenAD-LateralShift",
        #         config={
        #             "fut_ts": fut_ts,
        #             "fut_mode": fut_mode,
        #             "map_thresh": map_thresh,
        #             "dis_thresh": dis_thresh,
        #             "valid_fut_ts": valid_fut_ts,
        #         },
        #         name=f"GenAD-LateralShift-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        #     )
        # self.wandb_initialized = True

        if loss_traj_cls['use_sigmoid'] == True:
            self.traj_num_cls = 1
        else:
            self.traj_num_cls = 2

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        if map_code_size is not None:
            self.map_code_size = map_code_size
        else:
            self.map_code_size = 10
        if map_code_weights is not None:
            self.map_code_weights = map_code_weights
        else:
            self.map_code_weights = [1.0, 1.0, 1.0,
                                     1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        self.map_bbox_coder = build_bbox_coder(map_bbox_coder)
        self.map_query_embed_type = map_query_embed_type
        self.map_transform_method = map_transform_method
        self.map_gt_shift_pts_pattern = map_gt_shift_pts_pattern
        map_num_query = map_num_vec * map_num_pts_per_vec
        self.map_num_query = map_num_query
        self.map_num_classes = map_num_classes
        self.map_num_vec = map_num_vec
        self.map_num_pts_per_vec = map_num_pts_per_vec
        self.map_num_pts_per_gt_vec = map_num_pts_per_gt_vec
        self.map_dir_interval = map_dir_interval

        if loss_map_cls['use_sigmoid'] == True:
            self.map_cls_out_channels = map_num_classes
        else:
            self.map_cls_out_channels = map_num_classes + 1

        self.map_bg_cls_weight = 0
        map_class_weight = loss_map_cls.get('class_weight', None)
        if map_class_weight is not None and (self.__class__ is GenADHead):
            assert isinstance(map_class_weight, float), 'Expected ' \
                                                        'class_weight to have type float. Found ' \
                                                        f'{type(map_class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            map_bg_cls_weight = loss_map_cls.get('bg_cls_weight', map_class_weight)
            assert isinstance(map_bg_cls_weight, float), 'Expected ' \
                                                         'bg_cls_weight to have type float. Found ' \
                                                         f'{type(map_bg_cls_weight)}.'
            map_class_weight = torch.ones(map_num_classes + 1) * map_class_weight
            # set background class as the last indice
            map_class_weight[map_num_classes] = map_bg_cls_weight
            loss_map_cls.update({'class_weight': map_class_weight})
            if 'bg_cls_weight' in loss_map_cls:
                loss_map_cls.pop('bg_cls_weight')
            self.map_bg_cls_weight = map_bg_cls_weight

        self.traj_bg_cls_weight = 0

        super(GenADHead, self).__init__(*args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.map_code_weights = nn.Parameter(torch.tensor(
            self.map_code_weights, requires_grad=False), requires_grad=False)

        if kwargs['train_cfg'] is not None:
            assert 'map_assigner' in kwargs['train_cfg'], 'map assigner should be provided ' \
                                                          'when train_cfg is set.'
            map_assigner = kwargs['train_cfg']['map_assigner']
            assert loss_map_cls['loss_weight'] == map_assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_map_bbox['loss_weight'] == map_assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                           'should be exactly the same.'
            assert loss_map_iou['loss_weight'] == map_assigner['iou_cost']['weight'], \
                'The regression iou weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_map_pts['loss_weight'] == map_assigner['pts_cost']['weight'], \
                'The regression l1 weight for map pts loss and matcher should be' \
                'exactly the same.'

            self.map_assigner = build_assigner(map_assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.map_sampler = build_sampler(sampler_cfg, context=self)

        self.loss_traj = build_loss(loss_traj)
        self.loss_traj_cls = build_loss(loss_traj_cls)
        self.loss_map_bbox = build_loss(loss_map_bbox)
        self.loss_map_cls = build_loss(loss_map_cls)
        self.loss_map_iou = build_loss(loss_map_iou)
        self.loss_map_pts = build_loss(loss_map_pts)
        self.loss_map_dir = build_loss(loss_map_dir)
        self.loss_plan_reg = build_loss(loss_plan_reg)
        self.loss_plan_bound = build_loss(loss_plan_bound)
        self.loss_plan_col = build_loss(loss_plan_col)
        self.loss_plan_dir = build_loss(loss_plan_dir)
        self.loss_vae_gen = build_loss(loss_vae_gen)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        traj_branch = []
        if self.with_cur:
            traj_in_dim = self.embed_dims * 4
        else:
            traj_in_dim = self.embed_dims * 2
        for _ in range(self.num_reg_fcs):
            traj_branch.append(Linear(traj_in_dim, traj_in_dim))
            traj_branch.append(nn.ReLU())
        traj_branch.append(Linear(traj_in_dim, 2))
        traj_branch = nn.Sequential(*traj_branch)

        traj_cls_branch = []
        # for _ in range(self.num_reg_fcs):
        traj_cls_branch.append(Linear(self.embed_dims * 14, self.embed_dims * 2))
        traj_cls_branch.append(nn.LayerNorm(self.embed_dims * 2))
        traj_cls_branch.append(nn.ReLU(inplace=True))
        traj_cls_branch.append(Linear(self.embed_dims * 2, self.embed_dims * 2))
        traj_cls_branch.append(nn.LayerNorm(self.embed_dims * 2))
        traj_cls_branch.append(nn.ReLU(inplace=True))
        traj_cls_branch.append(Linear(self.embed_dims * 2, self.traj_num_cls))
        traj_cls_branch = nn.Sequential(*traj_cls_branch)

        map_cls_branch = []
        for _ in range(self.num_reg_fcs):
            map_cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            map_cls_branch.append(nn.LayerNorm(self.embed_dims))
            map_cls_branch.append(nn.ReLU(inplace=True))
        map_cls_branch.append(Linear(self.embed_dims, self.map_cls_out_channels))
        map_cls_branch = nn.Sequential(*map_cls_branch)

        map_reg_branch = []
        for _ in range(self.num_reg_fcs):
            map_reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            map_reg_branch.append(nn.ReLU())
        map_reg_branch.append(Linear(self.embed_dims, self.map_code_size))
        map_reg_branch = nn.Sequential(*map_reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_decoder_layers = 1
        num_map_decoder_layers = 1
        if self.transformer.decoder is not None:
            num_decoder_layers = self.transformer.decoder.num_layers
        if self.transformer.map_decoder is not None:
            num_map_decoder_layers = self.transformer.map_decoder.num_layers
        num_motion_decoder_layers = 1
        num_pred = (num_decoder_layers + 1) if \
            self.as_two_stage else num_decoder_layers
        motion_num_pred = (num_motion_decoder_layers + 1) if \
            self.as_two_stage else num_motion_decoder_layers
        map_num_pred = (num_map_decoder_layers + 1) if \
            self.as_two_stage else num_map_decoder_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(cls_branch, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            self.traj_branches = _get_clones(traj_branch, motion_num_pred)
            self.traj_cls_branches = _get_clones(traj_cls_branch, motion_num_pred)
            self.map_cls_branches = _get_clones(map_cls_branch, map_num_pred)
            self.map_reg_branches = _get_clones(map_reg_branch, map_num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [cls_branch for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
            self.traj_branches = nn.ModuleList(
                [traj_branch for _ in range(motion_num_pred)])
            self.traj_cls_branches = nn.ModuleList(
                [traj_cls_branch for _ in range(motion_num_pred)])
            self.map_cls_branches = nn.ModuleList(
                [map_cls_branch for _ in range(map_num_pred)])
            self.map_reg_branches = nn.ModuleList(
                [map_reg_branch for _ in range(map_num_pred)])

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)
            if self.map_query_embed_type == 'all_pts':
                self.map_query_embedding = nn.Embedding(self.map_num_query,
                                                        self.embed_dims * 2)
            elif self.map_query_embed_type == 'instance_pts':
                self.map_query_embedding = None
                self.map_instance_embedding = nn.Embedding(self.map_num_vec, self.embed_dims * 2)
                self.map_pts_embedding = nn.Embedding(self.map_num_pts_per_vec, self.embed_dims * 2)

        if self.motion_decoder is not None:
            self.motion_decoder = build_transformer_layer_sequence(self.motion_decoder)
            self.motion_mode_query = nn.Embedding(self.fut_mode, self.embed_dims)
            self.motion_mode_query.weight.requires_grad = True
            if self.use_pe:
                self.pos_mlp_sa = nn.Linear(2, self.embed_dims)
        else:
            raise NotImplementedError('Not implement yet')

        if self.motion_map_decoder is not None:
            self.lane_encoder = LaneNet(256, 128, 3)
            self.motion_map_decoder = build_transformer_layer_sequence(self.motion_map_decoder)
            if self.use_pe:
                self.pos_mlp = nn.Linear(2, self.embed_dims)

        if self.ego_his_encoder is not None:
            self.ego_his_encoder = LaneNet(2, self.embed_dims // 2, 3)
        else:
            self.ego_query = nn.Embedding(1, self.embed_dims)

        self.ego_agent_pos_mlp = nn.Linear(2, self.embed_dims)

        ego_fut_decoder = []
        ego_fut_dec_in_dim = self.embed_dims * 2 + len(self.ego_lcf_feat_idx) \
            if self.ego_lcf_feat_idx is not None else self.embed_dims * 2
        if self.with_cur:
            ego_fut_dec_in_dim = int(ego_fut_dec_in_dim * 2)
        for _ in range(self.num_reg_fcs):
            ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, ego_fut_dec_in_dim))
            ego_fut_decoder.append(nn.ReLU())
        ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, self.ego_fut_mode * 2))
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)

        self.ego_coord_mlp = nn.Linear(2, 2)

        # generator motion & planning
        self.layer_dim = 4
        self.present_distribution_in_channels = 512
        self.future_distribution_in_channels = 524
        self.now_pred_in_channels = 64
        self.probabilistic = True
        self.latent_dim = 32
        self.min_log_sigma = -5.0
        self.max_log_sigma = 5.0

        self.present_distribution = DistributionModule(
            self.present_distribution_in_channels,
            self.latent_dim,
            min_log_sigma=self.min_log_sigma,
            max_log_sigma=self.max_log_sigma,
        )

        self.future_distribution = DistributionModule(
            self.future_distribution_in_channels,
            self.latent_dim,
            min_log_sigma=self.min_log_sigma,
            max_log_sigma=self.max_log_sigma,
        )

        # Future prediction
        self.predict_model = PredictModel(
            in_channels=self.latent_dim,
            out_channels=self.embed_dims * 2,
            hidden_channels=self.latent_dim * 4,
            num_layers=self.layer_dim
        )






    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        if self.loss_map_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.map_cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        if self.loss_traj_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.traj_cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        # for m in self.map_reg_branches:
        #     constant_init(m[-1], 0, bias=0)
        # nn.init.constant_(self.map_reg_branches[0][-1].bias.data[2:], 0.)
        if self.motion_decoder is not None:
            for p in self.motion_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            nn.init.orthogonal_(self.motion_mode_query.weight)
            if self.use_pe:
                xavier_init(self.pos_mlp_sa, distribution='uniform', bias=0.)
        if self.motion_map_decoder is not None:
            for p in self.motion_map_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            for p in self.lane_encoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            if self.use_pe:
                xavier_init(self.pos_mlp, distribution='uniform', bias=0.)
        if self.ego_his_encoder is not None:
            for p in self.ego_his_encoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        # @auto_fp16(apply_to=('mlvl_feats'))

    def se2_transform(self, x, T):
        """SE(2) transformation of BEV features
        Args:
            x: 输入特征，可以是bev_mask/bev_pos/bev_embed等
            T: SE(2)变换矩阵，shape为[batch_size, 3]，包含[dx, dy, dtheta]
        """
        device = x.device
        
        # 特殊处理bev_embed的情况
        if len(x.shape) == 3 and x.shape[0] == self.bev_h * self.bev_w:  # bev_embed case [H*W, B, C]
            L, B, C = x.shape
            H = W = int(math.sqrt(L))

            # 使用clone()创建新的tensor，避免原地操作
            x_transformed = x.clone()
            # reshape to [B, C, H, W]
            x_4d = x_transformed.permute(1, 2, 0).reshape(B, C, H, W)
                
            # 进行网格坐标生成和变换
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H-1, H, device=device),
                torch.linspace(0, W-1, W, device=device)
            )
            grid_x = (grid_x - W/2) * (self.real_w / W)
            grid_y = (grid_y - H/2) * (self.real_h / H)

            transformed_x = torch.zeros_like(x_4d)
            # 应用SE(2)变换
            for b in range(B):
                dx, dy, dtheta = T[b]
                # 旋转变换
                rot_x = grid_x * torch.cos(dtheta) - grid_y * torch.sin(dtheta)
                rot_y = grid_x * torch.sin(dtheta) + grid_y * torch.cos(dtheta)
                # 平移变换 
                trans_x = rot_x + dx
                trans_y = rot_y + dy
                
                # 转回网格坐标
                trans_x = (trans_x / (self.real_w / W)) + W/2
                trans_y = (trans_y / (self.real_h / H)) + H/2
                
                # 准备grid_sample的输入
                grid = torch.stack([
                    2 * trans_x / (W-1) - 1,
                    2 * trans_y / (H-1) - 1
                ], dim=-1).unsqueeze(0)  # [1, H, W, 2]
                
                # 应用变换
                transformed_x[b:b+1] = F.grid_sample(
                    x_4d[b:b+1],
                    grid,
                    mode='bilinear',
                    padding_mode='zeros'
                )
                
            # 变换回原始形状 [H*W, B, C]，避免原地操作
            result = transformed_x.reshape(B, C, -1).permute(2, 0, 1).contiguous()
            return result
            
        # 原有的处理逻辑保持不变
        elif len(x.shape) == 3:  # bev_mask case [B, H, W]
            B, H, W = x.shape
            x_transformed = x.clone()
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H-1, H, device=device),
                torch.linspace(0, W-1, W, device=device)
            )
            grid_x = (grid_x - W/2) * (self.real_w / W)
            grid_y = (grid_y - H/2) * (self.real_h / H)
            
        elif len(x.shape) == 4:  # bev_pos case [B, C, H, W]
            B, C, H, W = x.shape
            x_transformed = x.clone()
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H-1, H, device=device),
                torch.linspace(0, W-1, W, device=device)
            )
            grid_x = (grid_x - W/2) * (self.real_w / W)
            grid_y = (grid_y - H/2) * (self.real_h / H)
        
        transformed = torch.zeros_like(x_transformed)
        # 应用SE(2)变换
        for b in range(B):
            dx, dy, dtheta = T[b]
            rot_x = grid_x * torch.cos(dtheta) - grid_y * torch.sin(dtheta)
            rot_y = grid_x * torch.sin(dtheta) + grid_y * torch.cos(dtheta)
            trans_x = rot_x + dx
            trans_y = rot_y + dy
            
            trans_x = (trans_x / (self.real_w / W)) + W/2
            trans_y = (trans_y / (self.real_h / H)) + H/2
            
            grid = torch.stack([
                2 * trans_x / (W-1) - 1,
                2 * trans_y / (H-1) - 1
            ], dim=-1).unsqueeze(0)
            
            if len(x.shape) == 3:
                transformed[b:b+1] = F.grid_sample(
                    x_transformed[b:b+1].unsqueeze(1),
                    grid,
                    mode='bilinear',
                    padding_mode='zeros'
                ).squeeze(1)
            elif len(x.shape) == 4:
                transformed[b:b+1] = F.grid_sample(
                    x_transformed[b:b+1],
                    grid,
                    mode='bilinear',
                    padding_mode='zeros'
                )
                
        return transformed
    
    def improved_se2_transform(self, x, T, padding_mode='reflection'):
        """改进的SE(2)变换函数，保持自车在BEV中心，并处理超出边界的区域
        
        Args:
            x: 输入特征，可以是bev_mask/bev_pos/bev_embed等
            T: SE(2)变换矩阵，shape为[batch_size, 3]，包含[dx, dy, dtheta]
            padding_mode: 填充模式，可选'zeros', 'replicate', 'reflection', 'edge_mask'
            
        Returns:
            变换后的特征
        """
        device = x.device
        
        # 特殊处理bev_embed的情况
        if len(x.shape) == 3 and x.shape[0] == self.bev_h * self.bev_w:  # bev_embed case [H*W, B, C]
            L, B, C = x.shape
            H = W = int(math.sqrt(L))

            # 使用clone()创建新的tensor，避免原地操作
            x_transformed = x.clone()
            # reshape to [B, C, H, W]
            x_4d = x_transformed.permute(1, 2, 0).reshape(B, C, H, W)
                
            # 进行网格坐标生成
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H-1, H, device=device),
                torch.linspace(0, W-1, W, device=device)
            )
            grid_x = (grid_x - W/2) * (self.real_w / W)  # 转换到实际坐标系
            grid_y = (grid_y - H/2) * (self.real_h / H)  # 转换到实际坐标系

            transformed_x = torch.zeros_like(x_4d)
            edge_masks = []  # 用于存储边缘掩码（如果需要）
            
            # 对每个batch样本进行处理
            for b in range(B):
                dx, dy, dtheta = T[b]
                
                # 直接使用原始的dx值，不取反
                # 这样做的效果是：移动BEV特征来模拟自车的偏移，而不是移动场景
                    
                # 旋转变换
                rot_x = grid_x * torch.cos(dtheta) - grid_y * torch.sin(dtheta)
                rot_y = grid_x * torch.sin(dtheta) + grid_y * torch.cos(dtheta)
                
                # 平移变换
                trans_x = rot_x + dx
                trans_y = rot_y + dy
                
                # 转回网格坐标
                trans_x = (trans_x / (self.real_w / W)) + W/2
                trans_y = (trans_y / (self.real_h / H)) + H/2
                
                # 准备grid_sample的输入
                grid = torch.stack([
                    2 * trans_x / (W-1) - 1,
                    2 * trans_y / (H-1) - 1
                ], dim=-1).unsqueeze(0)  # [1, H, W, 2]
                
                # 创建边缘掩码（如果需要）
                if padding_mode == 'edge_mask':
                    # 计算哪些点会超出原始边界
                    edge_mask = (trans_x < 0) | (trans_x >= W) | (trans_y < 0) | (trans_y >= H)
                    edge_mask = edge_mask.float().unsqueeze(0)
                    edge_masks.append(edge_mask)
                    # 使用replicate模式填充，后续会应用掩码
                    curr_padding_mode = 'replicate'
                else:
                    curr_padding_mode = padding_mode
                
                # 应用变换
                transformed_x[b:b+1] = F.grid_sample(
                    x_4d[b:b+1],
                    grid,
                    mode='bilinear',
                    padding_mode=curr_padding_mode,
                    align_corners=False
                )
                
            # 如果使用边缘掩码，将填充区域设为特定值
            if padding_mode == 'edge_mask':
                edge_mask = torch.stack(edge_masks, dim=0)  # [B, 1, H, W]
                # 为填充区域使用一个特殊的标记值
                # 通常选择一个极端值，如接近零或负值
                fill_value = 0.0  # 可以根据需要调整
                transformed_x = transformed_x * (1 - edge_mask) + fill_value * edge_mask
            
            # 变换回原始形状 [H*W, B, C]
            result = transformed_x.reshape(B, C, -1).permute(2, 0, 1).contiguous()
            return result
            
        # 处理其他维度的输入（保持与之前代码的兼容性）
        elif len(x.shape) == 3:  # bev_mask case [B, H, W]
            B, H, W = x.shape
            x_transformed = x.clone()
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H-1, H, device=device),
                torch.linspace(0, W-1, W, device=device)
            )
            grid_x = (grid_x - W/2) * (self.real_w / W)
            grid_y = (grid_y - H/2) * (self.real_h / H)
            
        elif len(x.shape) == 4:  # bev_pos case [B, C, H, W]
            B, C, H, W = x.shape
            x_transformed = x.clone()
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H-1, H, device=device),
                torch.linspace(0, W-1, W, device=device)
            )
            grid_x = (grid_x - W/2) * (self.real_w / W)
            grid_y = (grid_y - H/2) * (self.real_h / H)
        
        transformed = torch.zeros_like(x_transformed)
        edge_masks = []  # 用于存储边缘掩码（如果需要）
        
        # 对每个batch样本进行处理
        for b in range(B):
            dx, dy, dtheta = T[b]
            
            # 旋转变换
            rot_x = grid_x * torch.cos(dtheta) - grid_y * torch.sin(dtheta)
            rot_y = grid_x * torch.sin(dtheta) + grid_y * torch.cos(dtheta)
            
            # 平移变换
            trans_x = rot_x + dx
            trans_y = rot_y + dy
            
            # 转回网格坐标
            trans_x = (trans_x / (self.real_w / W)) + W/2
            trans_y = (trans_y / (self.real_h / H)) + H/2
            
            # 准备grid_sample的输入
            grid = torch.stack([
                2 * trans_x / (W-1) - 1,
                2 * trans_y / (H-1) - 1
            ], dim=-1).unsqueeze(0)  # [1, H, W, 2]
            
            # 创建边缘掩码（如果需要）
            if padding_mode == 'edge_mask':
                # 计算哪些点会超出原始边界
                edge_mask = (trans_x < 0) | (trans_x >= W) | (trans_y < 0) | (trans_y >= H)
                edge_mask = edge_mask.float().unsqueeze(0)
                edge_masks.append(edge_mask)
                # 使用replicate模式填充，后续会应用掩码
                curr_padding_mode = 'replicate'
            else:
                curr_padding_mode = padding_mode
            
            # 应用变换
            if len(x.shape) == 3:
                transformed[b:b+1] = F.grid_sample(
                    x_transformed[b:b+1].unsqueeze(1),
                    grid,
                    mode='bilinear',
                    padding_mode=curr_padding_mode,
                    align_corners=False
                ).squeeze(1)
            elif len(x.shape) == 4:
                transformed[b:b+1] = F.grid_sample(
                    x_transformed[b:b+1],
                    grid,
                    mode='bilinear',
                    padding_mode=curr_padding_mode,
                    align_corners=False
                )
        
        # 如果使用边缘掩码，将填充区域设为特定值
        if padding_mode == 'edge_mask':
            edge_mask = torch.stack(edge_masks, dim=0)  # [B, 1, H, W]
            if len(x.shape) == 3:
                edge_mask = edge_mask.squeeze(1)
            # 为填充区域使用一个特殊的标记值
            fill_value = 0.0  # 可以根据需要调整
            transformed = transformed * (1 - edge_mask) + fill_value * edge_mask
        
        return transformed

    def save_shift_log(self):
        """保存记录的偏移量数据"""
        
        # 创建保存目录
        save_dir = '/mnt/kuebiko/users/qdeng/GenAD/lateral_shift_logs_pretrained'
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成文件名，包含时间戳以避免覆盖
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        shift_data = {
                #  JSON 不能直接序列化 NumPy 数组，需要在保存之前将 NumPy 数组转换为普通的 Python 列表。
                'shifts': [{
                    'lateral_shift': info['lateral_shift'].tolist(),  # 转换为列表
                    'timestamp': info['timestamp'],
                    'scene_token': info['scene_token'],
                    'sample_idx': info['sample_idx']
                } for info in self.shift_log],            
                'parameters': {
                'mean': 0.0,
                'std': 1.0,
                # 'random_seed': 42
            }
        }
        
        # 保存为NPZ文件（如果主要是数值数据）
        save_path_npz = os.path.join(save_dir, f'lateral_shifts_{timestamp}.npz')
        lateral_shifts = np.array([info['lateral_shift'] for info in self.shift_log])
        timestamps = np.array([info['timestamp'] for info in self.shift_log])
        scene_tokens = np.array([info['scene_token'] for info in self.shift_log])
        sample_idxs = np.array([info['sample_idx'] for info in self.shift_log])
        
        np.savez(save_path_npz,
                lateral_shifts=lateral_shifts,
                timestamps=timestamps,
                scene_tokens=scene_tokens,
                sample_idxs=sample_idxs,
                mean=0.0,
                std=1.0)
        
        # 同时保存一个JSON文件，包含更多可读信息
        save_path_json = os.path.join(save_dir, f'lateral_shifts_{timestamp}.json')
        with open(save_path_json, 'w') as f:
            json.dump(shift_data, f, indent=4)
        
        # 清空列表，防止内存占用过大
        self.shift_log = []
        
        # print(f"Saved shift log to {save_path_npz} and {save_path_json}")

    # 析构方法
    def __del__(self):
        """在对象被销毁时保存剩余的数据,关闭wandb"""
        if hasattr(self, 'shift_log') and self.shift_log:
            self.save_shift_log()
            wandb.finish()


    # MTG--20250321
    def plan_recovery_trajectory(self, lateral_shift, ego_fut_trajs=None, lookahead_distance=3.0, num_points=40, dt=0.1, wheelbase=2.7):
        """使用Pure Pursuit算法计算从横向偏移位置恢复到原始轨迹的路径
        
        Args:
            lateral_shift (Tensor): 横向偏移量，形状为 [batch_size]
            ego_fut_trajs (Tensor): 真实的未来轨迹，用作参考路径
            lookahead_distance (float): 前视距离，用于Pure Pursuit算法
            num_points (int): 生成的轨迹点数量
            dt (float): 时间步长，用于模拟
            wheelbase (float): 车辆轴距，用于Ackerman模型
            
        Returns:
            Tensor: 恢复轨迹，形状为 [batch_size, num_points, 2]，每个点包含 [x, y] 坐标
        """
        device = lateral_shift.device
        batch_size = lateral_shift.shape[0]

        # 初始化恢复轨迹
        trajectories = torch.zeros((batch_size, num_points, 2), device=device)
        
        # 初始状态：偏移位置在y轴（前向）为0，x轴（横向）为lateral_shift
        # 车辆初始朝向为正前方（y轴正方向）
        for b in range(batch_size):
            # 获取参考路径 - 使用ego_fut_trajs的第一个模态(最可能的轨迹)
            if ego_fut_trajs is not None and ego_fut_trajs.shape[1] > 0:
                ref_path = ego_fut_trajs[b, 0, :, :].cpu().numpy()  # [future_length, 2]
            else:
                # 如果没有提供未来轨迹，则使用默认的垂直线作为参考
                ref_path = np.array([[0, i] for i in range(10)])

            x = lateral_shift[b].item()  # 初始x位置（横向偏移）
            y = 0.0                      # 初始y位置
            theta = 0.0                  # 初始朝向角（弧度）
            v = 5.0                      # 初始速度 (m/s)
            
            for i in range(num_points):
                # 记录当前位置
                trajectories[b, i, 0] = x
                trajectories[b, i, 1] = y
                
                # 在参考路径上找到最近点的索引
                distances = np.sqrt(np.sum(np.square(ref_path - np.array([x, y])), axis=1))
                closest_idx = np.argmin(distances)

                # 确定前视点索引（沿参考路径向前看）
                lookahead_idx = closest_idx
                accumulated_distance = 0

                # 沿参考路径寻找满足前视距离的点
                while lookahead_idx + 1 < len(ref_path) and accumulated_distance < lookahead_distance:
                    next_idx = lookahead_idx + 1
                    segment_distance = np.sqrt(np.sum(np.square(ref_path[next_idx] - ref_path[lookahead_idx])))
                    accumulated_distance += segment_distance
                    lookahead_idx = next_idx

                # 确保不超出范围
                lookahead_idx = min(lookahead_idx, len(ref_path) - 1)

                # 获取目标点（前视点）
                target_x, target_y = ref_path[lookahead_idx]
                
                # # 如果目标点会导致车辆向后行驶，则选择一个向前的目标点
                # if target_y < y:  # 如果目标点在当前位置的后方
                #     # 选择在当前位置正前方的点作为目标
                #     target_y = y + lookahead_distance
                #     # 保持横向位置不变或向中心线靠拢
                #     target_x = x * 0.8  # 逐渐向中心线靠拢

                # 计算当前位置到目标点的向量
                dx = target_x - x
                dy = target_y - y
                
                # 计算目标点在车体坐标系中的位置
                target_distance = math.sqrt(dx**2 + dy**2)
                
                # 坐标转换到车体坐标系
                dx_body = dx * math.cos(-theta) - dy * math.sin(-theta)
                dy_body = dx * math.sin(-theta) + dy * math.cos(-theta)
                
                # 计算曲率 (Pure Pursuit核心)
                if target_distance < 1e-6 or abs(dy_body) < 1e-6:
                    curvature = 0.0
                else:
                    curvature = 2 * dx_body / (target_distance**2)
                
                # =====  简化的ackerman模型  ===== #
                # 计算转向角 (Ackerman模型)
                steering = math.atan(wheelbase * curvature)

                # 限制转向角的范围，防止过度转向
                max_steering = math.radians(30)  # 最大转向角为30度
                steering = max(min(steering, max_steering), -max_steering)
                
                # 确保车辆向前移动 - 如果y方向速度可能为负，则调整转向角
                # 预测下一步的朝向
                predicted_theta = theta + v * dt * math.tan(steering) / wheelbase
                # 检查该朝向下y方向的速度分量
                predicted_vy = v * math.cos(predicted_theta)

                if predicted_vy < 0:  # 如果预测的y方向速度为负（向后行驶）
                    # 强制将转向角调整到保证向前运动的值
                    if x > 0:  # 如果在中心线右侧
                        steering = -max_steering  # 向左转
                    else:  # 如果在中心线左侧
                        steering = max_steering  # 向右转

                # 更新位置和朝向（简化的运动学模型--Bicycle Model）
                x = x + v * dt * math.sin(theta)
                y = y + v * dt * math.cos(theta)
                theta = theta + v * dt * math.tan(steering) / wheelbase
                # =====  简化的ackerman模型  ===== #

                # # =====  完整的Ackerman模型  ===== #
                # # 计算Ackerman转向角 - 这是对中线的转向角
                # steering_center = math.atan(wheelbase * curvature)
                
                # # 限制转向角的范围
                # max_steering = math.radians(30)  # 最大转向角为30度
                # steering_center = max(min(steering_center, max_steering), -max_steering)
                
                # # Ackerman几何下，内外轮转向角度不同
                # # 计算转向方向
                # turning_right = steering_center < 0
                # turning_left = steering_center > 0
                
                # # 确保车辆向前移动 - 检查下一步的朝向是否会导致向后运动
                # next_theta = theta + v * dt * math.tan(steering_center) / wheelbase
                # # 在新朝向下，检查y方向速度分量
                # vy_next = v * math.cos(next_theta)
                
                # if vy_next < 0:  # 如果会导致向后运动
                #     # 调整转向角使车辆转向中心线
                #     if x > 0:  # 如果在中心线右侧
                #         steering_center = -max_steering  # 向左转
                #     else:
                #         steering_center = max_steering   # 向右转
                    
                #     # 重新计算内外轮转向角
                #     turning_right = steering_center < 0
                #     turning_left = steering_center > 0
                
                # # 使用Ackerman模型更新车辆位置
                # # 根据Ackerman几何，车辆会沿着一个圆弧运动
                # if abs(steering_center) < 1e-6:  # 直线运动
                #     x = x + v * dt * math.sin(theta)
                #     y = y + v * dt * math.cos(theta)
                #     # 朝向角不变
                # else:  # 转弯运动
                #     # 计算转弯半径
                #     turn_radius = wheelbase / math.tan(abs(steering_center))
                    
                #     # 计算角速度
                #     angular_velocity = v / turn_radius
                    
                #     # 如果向左转，角速度为正；如果向右转，角速度为负
                #     if turning_left:
                #         angular_velocity = abs(angular_velocity)
                #     else:
                #         angular_velocity = -abs(angular_velocity)
                    
                #     # 计算转过的角度
                #     delta_theta = angular_velocity * dt
                    
                #     # 更新朝向角
                #     theta = theta + delta_theta
                    
                #     # 确保theta保持在合理范围内
                #     theta = math.atan2(math.sin(theta), math.cos(theta))
                    
                #     # 计算车辆在圆弧上的位移
                #     delta_x = turn_radius * (math.cos(theta - delta_theta) - math.cos(theta))
                #     delta_y = turn_radius * (math.sin(theta) - math.sin(theta - delta_theta))
                    
                #     # 如果向左转，车辆中心在转弯圆心的左侧；如果向右转，在右侧
                #     if turning_left:
                #         x = x + delta_x
                #         y = y + delta_y
                #     else:
                #         x = x - delta_x
                #         y = y - delta_y
                # # =====  完整的Ackerman模型  ===== #

                # 调整速度 - 离参考路径越远，速度越慢
                dist_to_ref = np.min(np.sqrt(np.sum(np.square(ref_path - np.array([x, y])), axis=1)))
                v = max(3.0, 5.0 * (1.0 - 0.5 * math.exp(-3.0 / (dist_to_ref + 0.1))))
        
        return trajectories

    # MTG--20250321
    def visualize_recovery_trajectory(self, lateral_shift, img_metas, ego_fut_trajs=None, recovery_trajectories=None, ego_futures=None, base_dir='/mnt/kuebiko/users/qdeng/GenAD/recovery_trajectory_vis_1'):
        """可视化恢复轨迹，始终以自车位置为中心，前进方向向上
        
        Args:
            lateral_shift (Tensor): 横向偏移量
            img_metas (list): 包含场景信息的字典列表
            ego_fut_trajs (Tensor): 真实的未来轨迹，用作参考路径
            base_dir (str): 基础保存目录
        """
        batch_size = lateral_shift.shape[0]
        
        # 生成ego未来恢复轨迹
        if recovery_trajectories is None or ego_futures is None:
            ego_futures, recovery_trajectories = self.generate_ego_future_from_recovery(
                lateral_shift,
                ego_fut_trajs=ego_fut_trajs,
                base_future_length=6,
                time_interval=0.5
            )
        
        # 从配置中获取BEV边界
        x_min, y_min = self.pc_range[0], self.pc_range[1]
        x_max, y_max = self.pc_range[3], self.pc_range[4]
        
        # 可视化每个样本的轨迹
        for b in range(batch_size):
            # 获取场景信息
            scene_token = img_metas[b]['scene_token'] if img_metas else 'unknown'
            
            # 创建按场景组织的保存目录
            save_dir = os.path.join(base_dir, scene_token)
            os.makedirs(save_dir, exist_ok=True)
            
            # 创建图形
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            
            # 设置与BEV特征相同的坐标范围
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            
            # 获取当前样本的偏移量
            shift = lateral_shift[b].item()
            
            # 绘制原始自车位置(蓝色) - 在(0,0)位置
            vehicle_length, vehicle_width = 4.0, 1.8  # 自车尺寸近似值
            half_length, half_width = vehicle_length/2, vehicle_width/2
            
            # 绘制原始自车位置(以黑色表示) - 在(0,0)位置
            rect_orig = plt.Rectangle(
                (-half_width, -half_length), 
                vehicle_width, vehicle_length, 
                color='black', alpha=0.8, label='Original Position'
            )
            ax.add_patch(rect_orig)
            # 添加朝向指示
            ax.arrow(0, 0, 0, half_length, head_width=0.3, head_length=0.5, fc='black', ec='black')
            
            # 绘制偏移后自车位置(以绿色表示) - 在(shift,0)位置
            rect_shifted = plt.Rectangle(
                (shift-half_width, -half_length), 
                vehicle_width, vehicle_length, 
                color='green', alpha=0.8, label='Shifted Position'
            )
            ax.add_patch(rect_shifted)
            # 添加朝向指示
            ax.arrow(shift, 0, 0, half_length, head_width=0.3, head_length=0.5, fc='green', ec='green')
            
            # 绘制原始轨迹 (使用真实的未来轨迹)
            if ego_fut_trajs is not None and ego_fut_trajs.shape[1] > 0:
                # 获取原始轨迹数据 - 这是局部坐标系下的相对位移
                orig_traj_deltas = ego_fut_trajs[b, 0, :, :].cpu().numpy()  # [future_length, 2]
                
                # 第一步：转换为局部坐标系下的绝对位置
                local_abs_traj = np.zeros_like(orig_traj_deltas)
                local_abs_traj[0] = orig_traj_deltas[0]  # 第一个点可能已经是位移
                
                # 累积求和得到局部坐标系下的绝对位置
                for i in range(1, len(orig_traj_deltas)):
                    local_abs_traj[i] = local_abs_traj[i-1] + orig_traj_deltas[i]
                
                # 使用转换后的局部绝对坐标绘制原始轨迹
                # 注意：在BEV视图中，所有轨迹都应该相对于原点(0,0)绘制
                ax.plot(local_abs_traj[:, 0], local_abs_traj[:, 1], 'k-', linewidth=4, 
                    label='Original Trajectory', marker='o', markersize=5, zorder=5)
                
                # 在轨迹上添加方向箭头
                for i in range(0, len(local_abs_traj)-1, 2):
                    dx = local_abs_traj[i+1, 0] - local_abs_traj[i, 0]
                    dy = local_abs_traj[i+1, 1] - local_abs_traj[i, 1]
                    if dx**2 + dy**2 > 0.01:  # 只在点之间有足够距离时添加箭头
                        ax.arrow(local_abs_traj[i, 0], local_abs_traj[i, 1], dx*0.7, dy*0.7, 
                                head_width=0.3, head_length=0.5, fc='black', ec='black', zorder=6)
                
            # 绘制恢复轨迹 - 从(shift,0)开始
            # recovery_trajectories已经是在BEV坐标系下的绝对位置
            recovery_traj = recovery_trajectories[b].cpu().numpy()
            ax.plot(recovery_traj[:, 0], recovery_traj[:, 1], 'b-', linewidth=3, label='Recovery Trajectory')
            # 在轨迹上添加方向箭头
            for i in range(1, len(recovery_traj), 5):
                if i < len(recovery_traj)-1:
                    dx = recovery_traj[i+1, 0] - recovery_traj[i, 0]
                    dy = recovery_traj[i+1, 1] - recovery_traj[i, 1]
                    if dx**2 + dy**2 > 0.01:    # 只有在点之间有显著移动时才添加箭头
                        ax.arrow(recovery_traj[i, 0], recovery_traj[i, 1], dx*0.7, dy*0.7, 
                                head_width=0.3, head_length=0.5, fc='blue', ec='blue', zorder=4)
                
            # 绘制ego未来轨迹（不同模态）
            ego_future = ego_futures[b].cpu().numpy()
            for m in range(ego_future.shape[0]):
                if m == 0:
                    ax.plot(ego_future[m, :, 0], ego_future[m, :, 1], 'g-', linewidth=3, 
                        marker='o', markersize=5, label=f'Ego Future Mode {m}', zorder=3)
                else:
                    ax.plot(ego_future[m, :, 0], ego_future[m, :, 1], 'g--', linewidth=2, 
                        alpha=0.7, zorder=2)
                    
            # 设置图形属性
            plt.grid(True)
            plt.xlabel('Lateral Position (m)')
            plt.ylabel('Longitudinal Position (m)')
            plt.title(f'Recovery Trajectory from Lateral Shift: {shift:.2f} m')
            plt.legend(loc='upper right')
            # 保持坐标轴相等比例，确保视觉上不失真
            plt.axis('equal')
            
            # 添加参考线
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            plt.axvline(x=shift, color='gray', linestyle='--', alpha=0.5)
            
            # 添加BEV边界框参考线
            plt.plot([x_min, x_max, x_max, x_min, x_min], 
                    [y_min, y_min, y_max, y_max, y_min], 
                    'k:', alpha=0.3)
            
            # 标注前进方向
            plt.annotate('Forward Direction', xy=(0, y_max*0.9), xytext=(0, y_max*0.95), 
                        arrowprops=dict(arrowstyle='->'), ha='center')
            
            # 保存图形
            lidar_file = img_metas[b]['pts_filename'] if img_metas else 'unknown'
            timestamp = re.search(r'__(\d+)\.pcd\.bin$', lidar_file).group(1) if lidar_file != 'unknown' else datetime.now().strftime('%Y%m%d_%H%M%S')
            sample_idx = img_metas[b]['sample_idx'] if img_metas else 'unknown'
            
            save_path = os.path.join(save_dir, f'sample_{sample_idx}_{timestamp}_shift_{shift:.2f}.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()
            
            # print(f"Saved visualization to {save_path}")

    # # 加入了不同的lookahead调整模式（time/distance/dynamic）
    def improved_plan_recovery_trajectory(self, lateral_shift, ego_fut_trajs=None, num_points=40, dt=0.1, wheelbase=2.7, 
                                min_lookahead=2.0, max_lookahead=6.0, speed_factor=0.5, 
                                lookahead_method='time', lookahead_time=5.0, 
                                steering_smoothing=0.9, decay_constant=6.0):
        """Uses an improved Pure Pursuit algorithm to calculate recovery trajectory from lateral offset
        
        Args:
            lateral_shift (Tensor): Lateral shift magnitude [batch_size]
            ego_fut_trajs (Tensor): Ground truth future trajectory as reference
            num_points (int): Number of trajectory points to generate
            dt (float): Time step for simulation
            wheelbase (float): Vehicle wheelbase for Ackerman steering model
            min_lookahead (float): Minimum lookahead distance (m)。最小前瞻距离，无论车速多低，前瞻点都不会小于这个距离
            max_lookahead (float): Maximum lookahead distance (m)。最大前瞻距离，无论车速多高，前瞻点都不会大于这个距离
            speed_factor (float): Factor to convert speed to lookahead distance。速度因子，用于计算速度对前瞻距离的影响，值越大，高速时前瞻越远，转向越平滑但反应越慢。
            lookahead_method (str): Method to calculate lookahead - 'dynamic', 'time', or 'adaptive'
            lookahead_time (float): Time in seconds for time-based lookahead
            steering_smoothing (float): Smoothing factor for steering commands (0.0-1.0)
            decay_constant (float): Controls how quickly to converge to reference path。轨迹收敛衰减常数，值越大，收敛越慢，轨迹越平滑；值越小，收敛越快但可能出现过冲。
            
        Returns:
            Tensor: Recovery trajectories [batch_size, num_points, 2]
        """
        device = lateral_shift.device
        batch_size = lateral_shift.shape[0]

        # Initialize recovery trajectories
        # 初始化恢复轨迹
        trajectories = torch.zeros((batch_size, num_points, 2), device=device)
        
        # 标记是否有有效的恢复轨迹
        valid_trajectories = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for b in range(batch_size):
            has_valid_trajectory = False  # 初始化为False

            # Get reference path - use ground truth future trajectory if available
            if ego_fut_trajs is not None:
                ref_path = ego_fut_trajs[b, 0, :, :].cpu().numpy()  # [future_length, 2]
                # 计算轨迹总移动距离，判断是否为停止状态
                total_movement = np.sum(np.sqrt(np.sum(np.diff(ref_path, axis=0)**2, axis=1)))
                
                if total_movement > 0.5:        # 如果总移动超过50cm，认为是有效轨迹
                    has_valid_trajectory = True
                    valid_trajectories[b] = True
            else:
                # Default straight path if no ground truth is provided
                ref_path = np.array([[0, i] for i in range(10)])

            if not has_valid_trajectory:    # 如果没有有效轨迹，则跳过该样本的处理
                # 将轨迹设置为零或特殊标记值，表示无需恢复
                trajectories[b] = torch.zeros((num_points, 2), device=device)
                continue
                
            # Initial state
            x = lateral_shift[b].item()     # Initial lateral position (offset)
            y = 0.0                         # Initial longitudinal position
            theta = 0.0                     # Initial heading angle (radians)
            v = 5.0                         # Initial speed (m/s)
            prev_steering = 0.0             # For steering smoothing
            
            # Initial lateral error (for adaptive lookahead)
            initial_lateral_error = abs(x)
            
            # 初始横向误差（用于自适应前视）
            initial_lateral_error = abs(x)
            lateral_error_threshold = 0.05  # 5cm阈值，达到后认为已回到原始轨迹
            convergence_factor = 0.0        # 收敛因子初始化
            distance_traveled = 0.0         # 初始化行驶距离
            steering = 0.0                  # 初始化steering变量

            for i in range(num_points):
                # 基于当前速度预设lookahead_distance的值
                lookahead_distance = v * lookahead_time
                lookahead_distance = max(min_lookahead, min(lookahead_distance, max_lookahead))
                
                # Record current position
                trajectories[b, i, 0] = x
                trajectories[b, i, 1] = y
                
                # Calculate current lateral error to reference path
                lateral_error = None
                closest_idx = None
                min_dist = float('inf')
                
                # 找到参考路径上的最近点并计算横向误差
                # Find closest point on reference path and calculate lateral error
                for idx, path_point in enumerate(ref_path):
                    dist = np.sqrt((path_point[0] - x)**2 + (path_point[1] - y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = idx
                        
                if closest_idx is not None:
                    # Calculate vector from vehicle to closest point
                    # 计算向量从车辆到最近点
                    path_x, path_y = ref_path[closest_idx]
                    # Project displacement vector onto vehicle's lateral axis to get lateral error
                    # 将位移向量投影到车辆的横向轴以获得横向误差
                    displacement_x = path_x - x
                    displacement_y = path_y - y
                    lateral_error = displacement_x * math.cos(theta) - displacement_y * math.sin(theta)
                else:   # 回退到简单的横向偏移
                    lateral_error = x  # Fallback to simple lateral offset
                
                # ↓↓↓ 新代码，误差小于阈值时直接修改航向 ↓↓↓
                # 计算到参考路径的当前横向误差
                current_lateral_error = abs(lateral_error)
                
                # 随着前进距离增加，增强向原始轨迹的吸引力
                distance_traveled += v * dt
                convergence_factor = min(1.0, distance_traveled / (5.0 * decay_constant))
                
                # 当接近原始轨迹时，显著减小前视距离，更精确地跟踪
                if current_lateral_error < 0.5:  # 50cm内开始精确跟踪
                    lookahead_distance = max(min_lookahead, lookahead_distance * (0.5 + 0.5 * current_lateral_error))
                
                # 当误差小于阈值时，直接对准原始轨迹上的点
                if current_lateral_error < lateral_error_threshold:
                    # # option1: 直接调整航向对准该点
                    # # desired_heading = math.atan2(ref_path[closest_idx, 0] - x, ref_path[closest_idx, 1] - y)
                    # # 找到最近的原始轨迹点
                    # nearest_path_point = ref_path[closest_idx]
                    # # 直接调整航向对准该点
                    # desired_heading = math.atan2(nearest_path_point[0] - x, nearest_path_point[1] - y)
                    # # 平滑过渡到期望航向
                    # heading_diff = desired_heading - theta
                    # theta = theta + 0.8 * heading_diff  # 平滑过渡

                    # option2: 直接调整到原始轨迹上的点
                    # 计算轨迹的切向朝向
                    if closest_idx < len(ref_path) - 1:
                        path_heading = math.atan2(ref_path[closest_idx+1, 0] - ref_path[closest_idx, 0],
                                                ref_path[closest_idx+1, 1] - ref_path[closest_idx, 1])
                    else:
                        # 如果是最后一个点，使用前一段的方向
                        path_heading = math.atan2(ref_path[closest_idx, 0] - ref_path[closest_idx-1, 0],
                                                ref_path[closest_idx, 1] - ref_path[closest_idx-1, 1])
                    
                    # 检查朝向误差（标准化到±π范围内）
                    heading_error = abs((theta - path_heading + math.pi) % (2 * math.pi) - math.pi)
                    heading_threshold = math.radians(15)  # 15度朝向容差
                    
                    # 检查速度投影，确保速度方向与参考轨迹一致
                    velocity_alignment = v * math.cos(theta - path_heading)
                    velocity_threshold = 3.0  # 最小前向速度要求
                    
                    # 同时满足位置、朝向和速度条件时认为恢复完成
                    if heading_error < heading_threshold and velocity_alignment > velocity_threshold:
                        # 记录恢复成功
                        recovered = True
                        
                        # 使用基于车辆动力学的平滑过渡填充剩余轨迹点
                        current_x, current_y = x, y
                        current_theta = theta
                        current_v = v
                        current_steering = steering
                        
                        for j in range(i+1, num_points):
                            # 寻找参考轨迹上的最近点
                            ref_min_dist = float('inf')
                            next_closest_idx = closest_idx
                            
                            for idx in range(max(0, closest_idx-3), min(len(ref_path), closest_idx+15)):
                                ref_dist = np.sqrt((ref_path[idx, 0] - current_x)**2 + (ref_path[idx, 1] - current_y)**2)
                                if ref_dist < ref_min_dist:
                                    ref_min_dist = ref_dist
                                    next_closest_idx = idx
                            
                            # 查找前视点（适应当前速度）
                            look_ahead_dist = max(min_lookahead, current_v * 1.0)
                            look_ahead_idx = next_closest_idx
                            accum_dist = 0
                            
                            while look_ahead_idx + 1 < len(ref_path) and accum_dist < look_ahead_dist:
                                seg_dist = np.sqrt(np.sum(np.square(ref_path[look_ahead_idx+1] - ref_path[look_ahead_idx])))
                                accum_dist += seg_dist
                                look_ahead_idx += 1
                            
                            look_ahead_idx = min(look_ahead_idx, len(ref_path)-1)
                            
                            # 计算目标点
                            target_x, target_y = ref_path[look_ahead_idx]
                            
                            # 计算车辆局部坐标系下的目标位置
                            dx = target_x - current_x
                            dy = target_y - current_y
                            dx_local = dx * math.cos(-current_theta) - dy * math.sin(-current_theta)
                            dy_local = dx * math.sin(-current_theta) + dy * math.cos(-current_theta)
                            
                            # 计算曲率和转向角
                            target_dist = math.sqrt(dx_local**2 + dy_local**2)
                            if target_dist > 0.1:
                                curvature = 2 * dx_local / (target_dist**2)
                                raw_steering = math.atan(wheelbase * curvature)
                                max_steering = math.radians(30)
                                raw_steering = max(min(raw_steering, max_steering), -max_steering)
                                
                                # 平滑转向输入
                                current_steering = current_steering * steering_smoothing + raw_steering * (1 - steering_smoothing)
                            
                            # 更新位置（使用bicycle模型）
                            current_x += current_v * dt * math.sin(current_theta)
                            current_y += current_v * dt * math.cos(current_theta)
                            current_theta += current_v * dt * math.tan(current_steering) / wheelbase
                            
                            # 速度调整以匹配参考轨迹特性
                            if next_closest_idx < len(ref_path) - 1:
                                # 估算参考轨迹的期望速度
                                ref_segment = np.sqrt(np.sum(np.square(ref_path[next_closest_idx+1] - ref_path[next_closest_idx])))
                                target_v = ref_segment / dt  # 简单估计
                                target_v = min(8.0, max(3.0, target_v))  # 合理范围限制
                                
                                # 平滑过渡到目标速度
                                current_v = current_v * 0.9 + target_v * 0.1
                            
                            # 保存轨迹点
                            trajectories[b, j, 0] = current_x
                            trajectories[b, j, 1] = current_y
                        
                        # 跳出主循环
                        break
                
                # ↑↑↑ 新代码结束 ↑↑↑

                # Determine lookahead distance based on method
                if lookahead_method == 'time':
                    # Time-based lookahead (lookahead distance = speed * time)
                    lookahead_distance = v * lookahead_time
                    lookahead_distance = max(min_lookahead, min(lookahead_distance, max_lookahead))
                    
                elif lookahead_method == 'adaptive':
                    # Adaptive lookahead based on lateral error and speed
                    # Larger lateral error -> larger lookahead for smoother approach
                    error_ratio = min(1.0, abs(lateral_error) / initial_lateral_error)
                    lookahead_distance = min_lookahead + (max_lookahead - min_lookahead) * error_ratio
                    # Also factor in speed for stability
                    speed_adjustment = v * speed_factor
                    lookahead_distance = min(max_lookahead, lookahead_distance + speed_adjustment)
                    
                else:  # 'dynamic' (default)
                    # Speed-based dynamic lookahead
                    lookahead_distance = min_lookahead + v * speed_factor
                    lookahead_distance = min(max_lookahead, lookahead_distance)
                
                # Find lookahead point on reference path
                accumulated_distance = 0
                lookahead_idx = closest_idx

                # Search forward from closest point to find lookahead point
                while lookahead_idx + 1 < len(ref_path) and accumulated_distance < lookahead_distance:
                    next_idx = lookahead_idx + 1
                    segment_distance = np.sqrt(np.sum(np.square(ref_path[next_idx] - ref_path[lookahead_idx])))
                    accumulated_distance += segment_distance
                    lookahead_idx = next_idx

                # Safety bound check
                lookahead_idx = min(lookahead_idx, len(ref_path) - 1)
                
                # Get target point coordinates
                target_x, target_y = ref_path[lookahead_idx]
                
                # Exponential decay approach - adjust target point to create smoother approach
                if decay_constant > 0:
                    # Distance traveled along path
                    distance_traveled = y
                    # Calculate desired lateral error based on exponential decay
                    desired_error = initial_lateral_error * math.exp(-distance_traveled / decay_constant)
                    # Sign preservation (left or right)
                    if x < 0:
                        desired_error = -desired_error
                        
                    # Blend between original target and exponentially decaying target
                    blend_factor = min(1.0, distance_traveled / (decay_constant * 2))
                    target_x = target_x * blend_factor + desired_error * (1 - blend_factor)
                
                # Transform target to vehicle's local coordinate frame
                dx = target_x - x
                dy = target_y - y
                
                # Convert to vehicle's coordinate system
                dx_local = dx * math.cos(-theta) - dy * math.sin(-theta)
                dy_local = dx * math.sin(-theta) + dy * math.cos(-theta)
                
                # Calculate curvature/steering angle (Pure Pursuit core)
                target_distance = math.sqrt(dx_local**2 + dy_local**2)
                
                # Avoid division by zero
                if target_distance < 1e-6 or abs(dy_local) < 1e-6:
                    curvature = 0.0
                else:
                    # Pure Pursuit formula
                    curvature = 2 * dx_local / (target_distance**2)
                
                # Calculate steering angle from curvature
                raw_steering = math.atan(wheelbase * curvature)
                
                # Apply steering angle limits
                max_steering = math.radians(30)  # Maximum 30 degrees steering
                raw_steering = max(min(raw_steering, max_steering), -max_steering)
                
                # Apply steering smoothing to prevent jerky movements
                if i > 0:
                    steering = prev_steering * steering_smoothing + raw_steering * (1 - steering_smoothing)
                else:
                    steering = raw_steering
                    
                prev_steering = steering
                
                # Predict next state with current steering
                pred_theta = theta + v * dt * math.tan(steering) / wheelbase
                pred_vy = v * math.cos(pred_theta)  # Longitudinal velocity component
                
                # Check if we'd move backward and correct if needed
                if pred_vy < 0:
                    if x > 0:  # Right of centerline
                        steering = -max_steering  # Turn left
                    else:  # Left of centerline
                        steering = max_steering   # Turn right
                
                # Update position using bicycle model
                x = x + v * dt * math.sin(theta)
                y = y + v * dt * math.cos(theta)
                theta = theta + v * dt * math.tan(steering) / wheelbase
                
                # Adapt speed based on distance to reference and steering angle
                # Slow down when far from reference or when steering heavily
                steering_factor = 1.0 - 0.5 * abs(steering) / max_steering
                path_dist_factor = 1.0 - 0.3 * min(1.0, abs(lateral_error) / initial_lateral_error)
                target_speed = 5.0 * min(steering_factor, path_dist_factor)
                # Smooth speed changes
                v = v * 0.9 + target_speed * 0.1
                v = max(3.0, v)  # Minimum speed
        
        return trajectories

    def improved_plan_recovery_trajectory_velocity(self, lateral_shift, ego_fut_trajs=None, num_points=40, dt=0.1, wheelbase=2.7, 
                                    min_lookahead=2.0, max_lookahead=6.0, speed_factor=0.5, 
                                    lookahead_method='time', lookahead_time=5.0, 
                                    steering_smoothing=0.9, decay_constant=6.0):
        """使用改进的Pure Pursuit算法计算从横向偏移位置恢复到原始轨迹的路径
        
        Args:
            lateral_shift (Tensor): 横向偏移量 [batch_size]
            ego_fut_trajs (Tensor): 真实的未来轨迹，作为参考路径
            num_points (int): 要生成的轨迹点数量
            dt (float): 仿真时间步长
            wheelbase (float): 车辆轴距，用于Ackerman转向模型
            min_lookahead (float): 最小前瞻距离(m)
            max_lookahead (float): 最大前瞻距离(m)
            speed_factor (float): 速度到前瞻距离的转换因子
            lookahead_method (str): 前瞻计算方法 - 'dynamic', 'time', 'adaptive'
            lookahead_time (float): 基于时间的前瞻时间(秒)
            steering_smoothing (float): 转向平滑因子(0.0-1.0)
            decay_constant (float): 控制收敛到参考路径的速度
            
        Returns:
            Tensor: 恢复轨迹 [batch_size, num_points, 2]
        """
        device = lateral_shift.device
        batch_size = lateral_shift.shape[0]

        # 初始化恢复轨迹
        trajectories = torch.zeros((batch_size, num_points, 2), device=device)
        
        # 标记有效的恢复轨迹
        valid_trajectories = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for b in range(batch_size):
            # 检查是否存在有效的恢复轨迹
            has_valid_trajectory = False
            
            # 获取参考路径 - 使用真实未来轨迹(如果可用)
            if ego_fut_trajs is not None:
                ref_path = ego_fut_trajs[b, 0, :, :].cpu().numpy()  # [future_length, 2]
                # 计算轨迹总移动距离，判断是否为静止状态
                total_movement = np.sum(np.sqrt(np.sum(np.diff(ref_path, axis=0)**2, axis=1)))
                
                if total_movement > 0.5:  # 如果总移动超过50cm，认为是有效轨迹
                    has_valid_trajectory = True
                    valid_trajectories[b] = True
            else:
                # 如果没有提供地面真实轨迹，使用默认直线路径
                ref_path = np.array([[0, i] for i in range(10)])
                
            if not has_valid_trajectory:  # 如果没有有效轨迹，跳过该样本处理
                trajectories[b] = torch.zeros((num_points, 2), device=device)
                continue
                
            # 初始状态设置
            x = lateral_shift[b].item()  # 初始横向位置(偏移)
            y = 0.0                      # 初始纵向位置
            
            # ===== 从参考轨迹估计初始速度 (不考虑速度方向) =====
            theta = 0.0                  # 初始航向角(弧度)
            if ref_path.shape[0] > 1:
                # 计算第一段轨迹的距离
                first_segment_dist = np.sqrt(np.sum(np.square(ref_path[1] - ref_path[0])))
                # 轨迹点的时间间隔为0.5s
                time_interval = 0.5
                estimated_speed = first_segment_dist / time_interval
                # 速度范围约束
                v = max(3.0, min(8.0, estimated_speed))
            else:
                v = 5.0  # 默认速度

            # ===== 从参考轨迹估计初始速度 (考虑速度方向)=====
            # # 计算初始航向角
            # if ref_path.shape[0] > 1:
            #     # 使用参考轨迹的前两个点计算初始航向
            #     initial_dx = ref_path[1, 0] - ref_path[0, 0]
            #     initial_dy = ref_path[1, 1] - ref_path[0, 1]
                
            #     # 在Pure Pursuit中，通常定义向前为0度，向右为90度
            #     # 航向角
            #     theta = math.atan2(initial_dy, initial_dx)
                
            #     # 计算速度大小
            #     first_segment_dist = np.sqrt(initial_dx**2 + initial_dy**2)
            #     time_interval = 0.5
            #     estimated_speed = first_segment_dist / time_interval
            #     v = max(3.0, min(8.0, estimated_speed))
            # else:
            #     theta = 0.0  # 默认航向
            #     v = 5.0      # 默认速度
                
            prev_steering = 0.0         # 用于转向平滑
            
            # 初始横向误差(用于自适应前视)
            initial_lateral_error = abs(x)
            lateral_error_threshold = 0.05  # 5cm阈值，达到后认为已回到原始轨迹
            convergence_factor = 0.0     # 收敛因子初始化
            distance_traveled = 0.0      # 已行驶距离
            steering = 0.0               # 初始化转向角
            
            # ===== 根据偏移量大小调整恢复策略参数 =====
            # 大偏移需要更平缓的恢复曲线
            if initial_lateral_error > 1.0:  # 大于1米的偏移
                # 增大衰减常数，使轨迹更平滑
                adjusted_decay_constant = decay_constant * 1.5
                # 减小初始速度以增加控制稳定性
                v = max(3.0, v * 0.8)
                # 增加转向平滑因子以避免急转弯
                adjusted_steering_smoothing = min(0.95, steering_smoothing * 1.1)
            else:
                adjusted_decay_constant = decay_constant
                adjusted_steering_smoothing = steering_smoothing
            
            # 恢复轨迹生成的主循环
            for i in range(num_points):
                # 记录当前位置
                trajectories[b, i, 0] = x
                trajectories[b, i, 1] = y
                
                # 计算到参考路径的当前最近点和横向误差
                distances = np.sqrt(np.sum(np.square(ref_path - np.array([x, y])), axis=1))
                closest_idx = np.argmin(distances)
                
                # 计算到最近点的横向误差
                path_x, path_y = ref_path[closest_idx]
                displacement_x = path_x - x
                displacement_y = path_y - y
                # 将位移向量投影到车辆横向轴以获得横向误差
                lateral_error = displacement_x * math.cos(theta) - displacement_y * math.sin(theta)
                
                # 随着前进距离增加，增强向原始轨迹的吸引力
                distance_traveled += v * dt
                convergence_factor = min(1.0, distance_traveled / (5.0 * adjusted_decay_constant))
                
                # 计算参考路径在当前点的曲率
                path_curvature = 0.0
                if closest_idx > 0 and closest_idx < len(ref_path) - 1:
                    prev_pt = ref_path[closest_idx - 1]
                    curr_pt = ref_path[closest_idx]
                    next_pt = ref_path[closest_idx + 1]
                    
                    # 使用三点法估计曲率
                    dx1, dy1 = curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1]
                    dx2, dy2 = next_pt[0] - curr_pt[0], next_pt[1] - curr_pt[1]
                    
                    # 计算方向变化
                    heading1 = math.atan2(dx1, dy1)  # 注意这里xy顺序
                    heading2 = math.atan2(dx2, dy2)
                    delta_heading = (heading2 - heading1 + math.pi) % (2 * math.pi) - math.pi
                    
                    # 近似曲率为方向变化除以距离
                    distance = math.sqrt(dx1**2 + dy1**2) + math.sqrt(dx2**2 + dy2**2)
                    if distance > 0.001:  # 避免除以零
                        path_curvature = abs(delta_heading / distance)
                
                # ===== 自适应前瞻距离计算 =====
                current_lateral_error = abs(lateral_error)
                
                # 1. 基于曲率调整 - 高曲率使用较短前瞻
                curvature_factor = 1.0 / (1.0 + 5.0 * path_curvature)
                
                # 2. 基于横向误差调整 - 接近路径时使用较短前瞻
                error_ratio = min(1.0, current_lateral_error / initial_lateral_error)
                error_factor = 0.5 + 0.5 * error_ratio
                
                # 3. 基于速度调整前瞻距离
                speed_lookahead = v * lookahead_time
                
                # 综合各因素调整前瞻距离
                if lookahead_method == 'adaptive':
                    # 完全自适应模式
                    lookahead_distance = min_lookahead + (max_lookahead - min_lookahead) * (
                        0.5 * curvature_factor + 0.3 * error_factor + 0.2 * min(1.0, speed_lookahead / max_lookahead)
                    )
                elif lookahead_method == 'time':
                    # 基于时间的前瞻，但结合曲率因素
                    base_lookahead = v * lookahead_time
                    lookahead_distance = base_lookahead * (0.7 + 0.3 * curvature_factor)
                else:  # 'dynamic'
                    # 动态前瞻，主要基于速度
                    lookahead_distance = min_lookahead + v * speed_factor * curvature_factor
                
                # 确保前瞻距离在合理范围内
                lookahead_distance = max(min_lookahead, min(lookahead_distance, max_lookahead))
                
                # 当接近原始轨迹时，减小前瞻距离以更精确地跟踪
                if current_lateral_error < 0.5:  # 50cm内开始精确跟踪
                    lookahead_distance = max(min_lookahead, lookahead_distance * (0.5 + 0.5 * current_lateral_error / 0.5))
                
                # ===== 检测是否已成功回到原始轨迹 =====
                if current_lateral_error < lateral_error_threshold:
                    # 计算轨迹的切向朝向
                    if closest_idx < len(ref_path) - 1:
                        path_heading = math.atan2(ref_path[closest_idx+1, 0] - ref_path[closest_idx, 0],
                                                ref_path[closest_idx+1, 1] - ref_path[closest_idx, 1])
                    else:
                        # 如果是最后一个点，使用前一段的方向
                        path_heading = math.atan2(ref_path[closest_idx, 0] - ref_path[closest_idx-1, 0],
                                                ref_path[closest_idx, 1] - ref_path[closest_idx-1, 1])
                    
                    # 检查朝向误差（标准化到±π范围内）
                    heading_error = abs((theta - path_heading + math.pi) % (2 * math.pi) - math.pi)
                    heading_threshold = math.radians(10)  # 10度朝向容差
                    
                    # 检查速度投影，确保速度方向与参考轨迹一致
                    velocity_alignment = v * math.cos(theta - path_heading)
                    velocity_threshold = 2.5  # 最小前向速度要求
                    
                    # 同时满足位置、朝向和速度条件时认为恢复完成
                    if heading_error < heading_threshold and velocity_alignment > velocity_threshold:
                        # 使用参考轨迹完成剩余点
                        current_x, current_y = x, y
                        current_theta = theta
                        current_speed = v
                        current_steering = steering
                        
                        for j in range(i+1, num_points):
                            # 找到参考轨迹上的最近点
                            nearest_distances = np.sqrt(np.sum(np.square(ref_path - np.array([current_x, current_y])), axis=1))
                            nearest_idx = np.argmin(nearest_distances)
                            
                            # 计算前瞻点索引
                            lookahead_idx = nearest_idx
                            accumulated_dist = 0.0
                            target_lookahead = current_speed * 1.0  # 速度相关的前瞻距离
                            
                            while lookahead_idx + 1 < len(ref_path) and accumulated_dist < target_lookahead:
                                next_idx = lookahead_idx + 1
                                seg_dist = np.sqrt(np.sum(np.square(ref_path[next_idx] - ref_path[lookahead_idx])))
                                accumulated_dist += seg_dist
                                lookahead_idx = next_idx
                            
                            # 获取目标点
                            target_x, target_y = ref_path[min(lookahead_idx, len(ref_path)-1)]
                            
                            # 计算局部坐标下的目标位置
                            dx = target_x - current_x
                            dy = target_y - current_y
                            dx_local = dx * math.cos(-current_theta) - dy * math.sin(-current_theta)
                            dy_local = dx * math.sin(-current_theta) + dy * math.cos(-current_theta)
                            
                            # 计算曲率和转向角
                            target_dist = math.sqrt(dx_local**2 + dy_local**2)
                            if target_dist > 0.1:
                                curvature = 2 * dx_local / (target_dist**2)
                                raw_steering = math.atan(wheelbase * curvature)
                                max_steering = math.radians(30)
                                raw_steering = max(min(raw_steering, max_steering), -max_steering)
                                
                                # 平滑转向输入
                                current_steering = current_steering * 0.7 + raw_steering * 0.3
                            
                            # 更新位置（使用bicycle模型）
                            current_x += current_speed * dt * math.sin(current_theta)
                            current_y += current_speed * dt * math.cos(current_theta)
                            current_theta += current_speed * dt * math.tan(current_steering) / wheelbase
                            
                            # 保存轨迹点
                            trajectories[b, j, 0] = current_x
                            trajectories[b, j, 1] = current_y
                        
                        # 跳出主循环
                        break
                
                # ===== 找到适当的前瞻点 =====
                # 从最近点开始沿参考路径寻找前瞻点
                accumulated_distance = 0
                lookahead_idx = closest_idx

                # 沿参考路径寻找满足前瞻距离的点
                while lookahead_idx + 1 < len(ref_path) and accumulated_distance < lookahead_distance:
                    next_idx = lookahead_idx + 1
                    segment_distance = np.sqrt(np.sum(np.square(ref_path[next_idx] - ref_path[lookahead_idx])))
                    accumulated_distance += segment_distance
                    lookahead_idx = next_idx

                # 确保索引不越界
                lookahead_idx = min(lookahead_idx, len(ref_path) - 1)
                
                # ===== 计算目标点并实现指数衰减轨迹 =====
                target_x, target_y = ref_path[lookahead_idx]
                
                # 指数衰减方法 - 创建更平滑的接近
                if adjusted_decay_constant > 0:
                    # 计算期望的横向误差(基于指数衰减)
                    desired_error = initial_lateral_error * math.exp(-distance_traveled / adjusted_decay_constant)
                    # 保留符号(左或右)
                    if x < 0:
                        desired_error = -desired_error
                        
                    # 混合原始目标点和基于衰减的目标点
                    blend_factor = min(1.0, distance_traveled / (adjusted_decay_constant * 2))
                    target_x = target_x * blend_factor + desired_error * (1 - blend_factor)
                
                # ===== 大偏移情况的特殊处理 =====
                # 大偏移时使用更积极的初始校正，然后平滑过渡到轨迹跟踪
                large_shift_threshold = 1.0  # 1米
                is_large_shift = initial_lateral_error > large_shift_threshold
                
                if is_large_shift and distance_traveled < 2.0:
                    # 针对大偏移的特殊处理
                    # 寻找沿参考路径前方2-3米的点
                    target_distance = 3.0
                    target_idx = closest_idx
                    
                    # 沿轨迹寻找更远的目标点
                    accumulated_dist = 0
                    while target_idx + 1 < len(ref_path) and accumulated_dist < target_distance:
                        next_idx = target_idx + 1
                        seg_dist = np.sqrt(np.sum(np.square(ref_path[next_idx] - ref_path[target_idx])))
                        accumulated_dist += seg_dist
                        target_idx = next_idx
                    
                    target_idx = min(target_idx, len(ref_path) - 1)
                    direct_target_x, direct_target_y = ref_path[target_idx]
                    
                    # 混合常规目标点和直接目标点
                    direct_blend = max(0, 1.0 - distance_traveled / 2.0)
                    target_x = target_x * (1 - direct_blend) + direct_target_x * direct_blend
                    target_y = target_y * (1 - direct_blend) + direct_target_y * direct_blend
                
                # ===== 最终的转向计算 =====
                # 转换目标点到车辆局部坐标系
                dx = target_x - x
                dy = target_y - y
                dx_local = dx * math.cos(-theta) - dy * math.sin(-theta)
                dy_local = dx * math.sin(-theta) + dy * math.cos(-theta)
                
                # 计算曲率和转向角(Pure Pursuit算法核心)
                target_distance = math.sqrt(dx_local**2 + dy_local**2)
                
                # 避免除以零
                if target_distance < 1e-6 or abs(dy_local) < 1e-6:
                    curvature = 0.0
                else:
                    # Pure Pursuit公式
                    curvature = 2 * dx_local / (target_distance**2)
                
                # 计算转向角
                raw_steering = math.atan(wheelbase * curvature)
                
                # 应用转向角限制
                max_steering = math.radians(30)  # 最大30度转向
                raw_steering = max(min(raw_steering, max_steering), -max_steering)
                
                # 大偏移情况下增加额外的校正
                if is_large_shift and distance_traveled < 1.0:
                    # 计算方向修正因子，使车辆更快地向轨迹中心转向
                    correction_factor = math.copysign(1.0, -lateral_error) * 0.4
                    correction_angle = math.radians(10) * correction_factor * (1.0 - distance_traveled)
                    raw_steering += correction_angle
                    raw_steering = max(min(raw_steering, max_steering), -max_steering)
                
                # 应用转向平滑以防止抖动
                if i > 0:
                    steering = prev_steering * adjusted_steering_smoothing + raw_steering * (1 - adjusted_steering_smoothing)
                else:
                    steering = raw_steering
                    
                prev_steering = steering
                
                # 预测下一状态并检查是否会向后移动
                pred_theta = theta + v * dt * math.tan(steering) / wheelbase
                pred_vy = v * math.cos(pred_theta)  # 纵向速度分量
                
                # 检查是否会倒车并修正
                if pred_vy < 0:
                    if x > 0:  # 中心线右侧
                        steering = -max_steering  # 向左转
                    else:  # 中心线左侧
                        steering = max_steering   # 向右转
                
                # ===== 自适应速度控制 =====
                # 基于横向误差和转向角调整速度
                # 转向角度越大，速度越低
                steering_factor = 1.0 - 0.5 * abs(steering) / max_steering
                # 离参考路径越远，速度越低
                path_dist_factor = 1.0 - 0.4 * min(1.0, abs(lateral_error) / initial_lateral_error)
                # 计算目标速度
                target_speed = 5.0 * min(steering_factor, path_dist_factor)
                
                # 考虑路径曲率对速度的影响
                curvature_speed_factor = 1.0 / (1.0 + 5.0 * path_curvature)
                target_speed *= curvature_speed_factor
                
                # 平滑速度变化
                v = v * 0.9 + target_speed * 0.1
                v = max(2.5, min(8.0, v))  # 速度保持在合理范围内
                
                # ===== 使用自行车模型更新位置 =====
                x = x + v * dt * math.sin(theta)
                y = y + v * dt * math.cos(theta)
                theta = theta + v * dt * math.tan(steering) / wheelbase
        
        return trajectories

    # def generate_ego_future_from_recovery(self, lateral_shift, ego_fut_trajs=None, base_future_length=6, time_interval=0.5):
    def generate_ego_future_from_recovery(self, lateral_shift, ego_fut_trajs=None, base_future_length=6, time_interval=0.5,
                                lookahead_method='time', min_lookahead=2.0, max_lookahead=5.0, 
                                speed_factor=0.5, lookahead_time=5.0, steering_smoothing=0.9, decay_constant=6.0):
        """基于恢复轨迹生成ego未来轨迹预测
        
        Args:
            lateral_shift (Tensor): 横向偏移量
            ego_fut_trajs (Tensor): 真实的未来轨迹，用作参考路径
            base_future_length (int): 未来轨迹的时间长度
            time_interval (float): 轨迹点的时间间隔
            lookahead_time影响轨迹的形成方式（控制算法参数）
            base_future_length和time_interval决定最终显示的轨迹长度（输出结果参数）
            
        Returns:
            Tensor: ego未来轨迹，形状为 [batch_size, fut_mode, future_length, 2]
        """
        batch_size = lateral_shift.shape[0]
        device = lateral_shift.device
        
        # # 计算恢复轨迹（密集采样，用于后续插值）
        # recovery_trajectories = self.plan_recovery_trajectory(
        #     lateral_shift, 
        #     ego_fut_trajs=ego_fut_trajs,
        #     lookahead_distance=3.0, 
        #     num_points=40,  # 采样更多点用于插值
        #     dt=0.1
        # )

        # Calculate recovery trajectory with more points for better sampling
        recovery_trajectories = self.improved_plan_recovery_trajectory_velocity(
            lateral_shift, 
            ego_fut_trajs=ego_fut_trajs,
            num_points=100,  # More points for better sampling
            dt=0.05,         # Smaller timestep for higher precision
            min_lookahead=min_lookahead,
            max_lookahead=max_lookahead,
            speed_factor=speed_factor,
            lookahead_method=lookahead_method,
            lookahead_time=lookahead_time,
            steering_smoothing=steering_smoothing,
            decay_constant=decay_constant
        )
        
        # 初始化未来轨迹预测
        # [batch_size, fut_mode, future_length, 2]
        future_length = base_future_length
        fut_mode = self.ego_fut_mode  # 使用配置中的未来模态数量
        ego_future = torch.zeros((batch_size, fut_mode, future_length, 2), device=device)
        
        # 为不同模态生成略有变化的轨迹
        for b in range(batch_size):
            # 基础恢复轨迹
            base_trajectory = recovery_trajectories[b]
            
            # 对每个模态生成略有变化的轨迹
            for m in range(fut_mode):
                # 时间间隔转换为点的索引间隔
                step = int(time_interval / 0.05)  # 假设recovery轨迹的dt=0.1
                
                # 对于第一个模态，使用原始恢复轨迹
                if m == 0:
                    variation = 0.0
                # 对于其他模态，添加一些变化
                else:
                    # 根据模态索引生成不同的变化
                    variation = (m - fut_mode // 2) * 0.2  # 在原始轨迹周围生成变化
                
                # 提取对应时间点的轨迹，并添加变化
                for t in range(future_length):
                    idx = min((t+1) * step, base_trajectory.shape[0] - 1)
                    ego_future[b, m, t, 0] = base_trajectory[idx, 0] + variation  # x坐标加上变化
                    ego_future[b, m, t, 1] = base_trajectory[idx, 1]              # y坐标保持不变
                    
                    # Add slight longitudinal variation for different modes too
                    if m != 0:
                        # Speed up or slow down slightly based on mode
                        longitudinal_var = (m - fut_mode // 2) * 0.1 * base_trajectory[idx, 1]
                        ego_future[b, m, t, 1] += longitudinal_var
        
        return ego_future, recovery_trajectories

    def improved_visualize_recovery_trajectory(self, lateral_shift, img_metas, ego_fut_trajs=None, recovery_trajectories=None,ego_futures=None,
                                base_dir='/mnt/kuebiko/users/qdeng/GenAD/recovery_trajectory_vis_1_improved',
                                lookahead_method='time', 
                                min_lookahead=2.0, max_lookahead=6.0, 
                                speed_factor=0.5, lookahead_time=5.0,
                                steering_smoothing=0.9, decay_constant=6.0):
        """Visualize recovery trajectories with enhanced parameters
        
        Args:
            lateral_shift (Tensor): Lateral shift magnitude
            img_metas (list): Scene metadata
            ego_fut_trajs (Tensor): Ground truth future trajectory
            base_dir (str): Base directory for saving visualizations
            lookahead_method (str): Method for calculating lookahead distance
            min_lookahead (float): Minimum lookahead distance
            max_lookahead (float): Maximum lookahead distance
            speed_factor (float): Speed multiplier for dynamic lookahead
            lookahead_time (float): Time in seconds for time-based lookahead
            steering_smoothing (float): Steering command smoothing factor
            decay_constant (float): Controls exponential convergence rate
        """
        batch_size = lateral_shift.shape[0]
        
        # 如果没有提供已计算的轨迹，则重新计算
        if recovery_trajectories is None or ego_futures is None:
            # 生成ego未来轨迹，同时获取恢复轨迹
            ego_futures, recovery_trajectories = self.generate_ego_future_from_recovery(
                lateral_shift,
                ego_fut_trajs=ego_fut_trajs,
                base_future_length=6,
                time_interval=0.5,
                lookahead_method=lookahead_method,
                min_lookahead=min_lookahead,
                max_lookahead=max_lookahead,
                speed_factor=speed_factor,
                lookahead_time=lookahead_time,
                steering_smoothing=steering_smoothing,
                decay_constant=decay_constant
            )
    
        # Get BEV boundaries
        x_min, y_min = self.pc_range[0], self.pc_range[1]
        x_max, y_max = self.pc_range[3], self.pc_range[4]
        
        # Create a parameter string for the filename
        param_str = f"{lookahead_method}_min{min_lookahead}_max{max_lookahead}_sf{speed_factor}_lt{lookahead_time}_ss{steering_smoothing}_dc{decay_constant}"
        
        # Visualize each sample
        for b in range(batch_size):
            # Get scene information
            scene_token = img_metas[b]['scene_token'] if img_metas else 'unknown'
            
            # Create directory structure by scene
            save_dir = os.path.join(base_dir, scene_token)
            os.makedirs(save_dir, exist_ok=True)
            
            # Create plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            
            # Set consistent BEV coordinate range
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            
            # Get current sample's shift
            shift = lateral_shift[b].item()
            
            # Vehicle dimensions
            vehicle_length, vehicle_width = 4.0, 1.8
            half_length, half_width = vehicle_length/2, vehicle_width/2
            
            # Draw original vehicle position (black)
            rect_orig = plt.Rectangle(
                (-half_width, -half_length), 
                vehicle_width, vehicle_length, 
                color='black', alpha=0.8, label='Original Position'
            )
            ax.add_patch(rect_orig)
            ax.arrow(0, 0, 0, half_length, head_width=0.3, head_length=0.5, fc='black', ec='black')
            
            # Draw shifted vehicle position (green)
            rect_shifted = plt.Rectangle(
                (shift-half_width, -half_length), 
                vehicle_width, vehicle_length, 
                color='green', alpha=0.8, label='Shifted Position'
            )
            ax.add_patch(rect_shifted)
            ax.arrow(shift, 0, 0, half_length, head_width=0.3, head_length=0.5, fc='green', ec='green')
            
            # Draw original trajectory if available
            # print("ego_fut_trajs: ", ego_fut_trajs)
            if ego_fut_trajs is not None:
                orig_traj_deltas = ego_fut_trajs[b, 0, :, :].cpu().numpy()
                
                # Convert to absolute positions
                local_abs_traj = np.zeros_like(orig_traj_deltas)
                # local_abs_traj[0] = orig_traj_deltas[0]  # 第一个点可能已经是位移
                # Start at origin
                local_abs_traj[0] = np.zeros(2)  # Assuming the first point should be at (0,0) - the vehicle's center
                
                for i in range(1, len(orig_traj_deltas)):
                    local_abs_traj[i] = local_abs_traj[i-1] + orig_traj_deltas[i]
                
                # Plot original trajectory
                ax.plot(local_abs_traj[:, 0], local_abs_traj[:, 1], 'k-', linewidth=4, 
                        label='Original Trajectory', marker='o', markersize=5, zorder=5)
                
                # Add direction arrows
                for i in range(0, len(local_abs_traj)-1, 2):
                    dx = local_abs_traj[i+1, 0] - local_abs_traj[i, 0]
                    dy = local_abs_traj[i+1, 1] - local_abs_traj[i, 1]
                    if dx**2 + dy**2 > 0.01:
                        ax.arrow(local_abs_traj[i, 0], local_abs_traj[i, 1], dx*0.7, dy*0.7, 
                                head_width=0.3, head_length=0.5, fc='black', ec='black', zorder=6)
            
            # Draw recovery trajectory
            recovery_traj = recovery_trajectories[b].cpu().numpy()
            ax.plot(recovery_traj[:, 0], recovery_traj[:, 1], 'b-', linewidth=3, label='Recovery Trajectory')
            
            # Add direction arrows on recovery trajectory
            for i in range(1, len(recovery_traj), 5):
                if i < len(recovery_traj)-1:
                    dx = recovery_traj[i+1, 0] - recovery_traj[i, 0]
                    dy = recovery_traj[i+1, 1] - recovery_traj[i, 1]
                    if dx**2 + dy**2 > 0.01:
                        ax.arrow(recovery_traj[i, 0], recovery_traj[i, 1], dx*0.7, dy*0.7, 
                                head_width=0.3, head_length=0.5, fc='blue', ec='blue', zorder=4)
            
            # Draw ego future trajectory modes
            ego_future = ego_futures[b].cpu().numpy()
            for m in range(ego_future.shape[0]):
                if m == 0:
                    ax.plot(ego_future[m, :, 0], ego_future[m, :, 1], 'g-', linewidth=3, 
                        marker='o', markersize=5, label=f'Ego Future Mode {m}', zorder=3)
                else:
                    ax.plot(ego_future[m, :, 0], ego_future[m, :, 1], 'g--', linewidth=2, 
                        alpha=0.7, zorder=2)
            
            # Plot styling
            plt.grid(True)
            plt.xlabel('Lateral Position (m)')
            plt.ylabel('Longitudinal Position (m)')
            plt.title(f'Recovery Trajectory from Lateral Shift: {shift:.2f} m\n{lookahead_method.capitalize()} lookahead')
            plt.legend(loc='upper right')
            plt.axis('equal')
            
            # Add reference lines
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            plt.axvline(x=shift, color='gray', linestyle='--', alpha=0.5)
            
            # Add BEV boundary reference
            plt.plot([x_min, x_max, x_max, x_min, x_min], 
                    [y_min, y_min, y_max, y_max, y_min], 
                    'k:', alpha=0.3)
            
            # Label forward direction
            plt.annotate('Forward Direction', xy=(0, y_max*0.9), xytext=(0, y_max*0.95), 
                        arrowprops=dict(arrowstyle='->'), ha='center')
            
            # Save the visualization
            lidar_file = img_metas[b]['pts_filename'] if img_metas else 'unknown'
            timestamp = re.search(r'__(\d+)\.pcd\.bin$', lidar_file).group(1) if lidar_file != 'unknown' else datetime.now().strftime('%Y%m%d_%H%M%S')
            sample_idx = img_metas[b]['sample_idx'] if img_metas else 'unknown'
            
            save_path = os.path.join(save_dir, f'sample_{sample_idx}_{timestamp}_shift_{shift:.2f}_{param_str}.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()
            
            # print(f"Saved visualization to {save_path}")

    def save_recovery_trajectories(self, lateral_shift, img_metas, ego_fut_trajs=None, recovery_trajectories=None, ego_futures=None, base_future_length=6, time_interval=0.5,
                            lookahead_method='time', min_lookahead=2.0, max_lookahead=5.0, speed_factor=0.5, lookahead_time=5.0, 
                            steering_smoothing=0.9, decay_constant=6.0, base_dir='/mnt/kuebiko/users/qdeng/GenAD/recovery_trajectory_data'):
        """保存恢复轨迹数据，按场景组织
        """
        batch_size = lateral_shift.shape[0]
        
        # 如果没有提供已计算的轨迹，则重新计算
        if recovery_trajectories is None or ego_futures is None:
            # 生成ego未来轨迹，同时获取恢复轨迹
            ego_futures, recovery_trajectories = self.generate_ego_future_from_recovery(
                lateral_shift,
                ego_fut_trajs=ego_fut_trajs,
                base_future_length=6,
                time_interval=0.5,
                lookahead_method=lookahead_method,
                min_lookahead=min_lookahead,
                max_lookahead=max_lookahead,
                speed_factor=speed_factor,
                lookahead_time=lookahead_time,
                steering_smoothing=steering_smoothing,
                decay_constant=decay_constant
            )
        
        # 保存每个样本的轨迹数据
        for b in range(batch_size):
            # 获取场景信息
            scene_token = img_metas[b]['scene_token'] if img_metas else 'unknown'
            
            # 创建按场景组织的保存目录
            save_dir = os.path.join(base_dir, scene_token)
            os.makedirs(save_dir, exist_ok=True)
            
            lidar_file = img_metas[b]['pts_filename'] if img_metas else 'unknown'
            timestamp = re.search(r'__(\d+)\.pcd\.bin$', lidar_file).group(1) if lidar_file != 'unknown' else datetime.now().strftime('%Y%m%d_%H%M%S')
            sample_idx = img_metas[b]['sample_idx'] if img_metas else 'unknown'
            
            # 创建保存路径
            save_path = os.path.join(save_dir, f'sample_{sample_idx}_{timestamp}_shift_{lateral_shift[b].item():.2f}')
            
            # 保存数据
            data_dict = {
                'lateral_shift': lateral_shift[b].item(),
                'recovery_trajectory': recovery_trajectories[b].cpu().numpy().tolist(),
                'ego_future': ego_futures[b].cpu().numpy().tolist(),
                'metadata': {
                    'scene_token': scene_token,
                    'timestamp': timestamp,
                    'sample_idx': img_metas[b]['sample_idx'] if img_metas else None,
                }
            }
            
            # 保存为JSON文件
            with open(f'{save_path}.json', 'w') as f:
                json.dump(data_dict, f, indent=4)
            
            # 同时保存为PyTorch张量
            torch.save({
                'lateral_shift': lateral_shift[b],
                'recovery_trajectory': recovery_trajectories[b],
                'ego_future': ego_futures[b],
                'metadata': {
                    'scene_token': scene_token,
                    'timestamp': timestamp,
                    'sample_idx': img_metas[b]['sample_idx'] if img_metas else None,
                }
            }, f'{save_path}.pth')
            
            # print(f"Saved trajectory data to {save_path}")

    def augment_batch_with_lateral_shift(self, bev_embed, img_metas, ego_fut_trajs, lateral_shift):
        """
        将原始BEV特征和偏移BEV特征合并为一个更大的批次
        
        Args:
            bev_embed: 原始BEV特征 [H*W, B, C]
            img_metas: 场景元数据
            ego_fut_trajs: 原始ego轨迹
            lateral_shift: 横向偏移量
            
        Returns:
            augmented_bev_embed: 增强后的BEV特征 [H*W, 2*B, C]
            augmented_img_metas: 增强后的元数据
            augmented_ego_fut_trajs: 增强后的ego轨迹
            augmented_lateral_shift: 增强后的偏移量
            augmented_mask: 区分原始/偏移数据的掩码 [2*B]
        """
        device = bev_embed.device
        bs = len(img_metas)
        
        # 1. 创建偏移BEV特征
        T = torch.zeros((bs, 3), device=device)
        T[:, 0] = lateral_shift
        padding_mode = 'reflection'
        bev_embed_shifted = self.improved_se2_transform(bev_embed, T, padding_mode=padding_mode)
        
        # 2. 生成恢复轨迹
        recovery_ego_trajs, recovery_trajectories = self.generate_ego_future_from_recovery(
            lateral_shift,
            ego_fut_trajs=ego_fut_trajs,
            base_future_length=6,
            time_interval=0.5,
            lookahead_method='time',
            min_lookahead=2.0,
            max_lookahead=5.0,
            speed_factor=0.5,
            lookahead_time=5.0,
            steering_smoothing=0.9,
            decay_constant=6.0
        )
        
        # 3. 合并数据
        # BEV特征合并 [H*W, 2*B, C]
        augmented_bev_embed = torch.cat([bev_embed, bev_embed_shifted], dim=1)
        
        # 元数据合并
        augmented_img_metas = img_metas + img_metas
        
        # 轨迹合并 - 原始批次使用原始轨迹，偏移批次使用恢复轨迹
        if ego_fut_trajs is not None:
            # 创建增强后的轨迹张量
            shape = ego_fut_trajs.shape
            augmented_ego_fut_trajs = torch.cat([ego_fut_trajs, recovery_ego_trajs], dim=0)
        else:
            augmented_ego_fut_trajs = None
        
        # 偏移量
        augmented_lateral_shift = torch.cat([torch.zeros_like(lateral_shift), lateral_shift], dim=0)
        
        # 创建掩码区分原始/偏移数据: 0表示原始数据，1表示偏移数据
        augmented_mask = torch.cat([torch.zeros(bs, device=device), torch.ones(bs, device=device)], dim=0)
        
        return augmented_bev_embed, augmented_img_metas, augmented_ego_fut_trajs, augmented_lateral_shift, augmented_mask

    # 関川さんのoption4的实现
    def create_lateral_shift_mask(self, T):
        """创建横向偏移掩码，标记无效区域"""
        device = T.device
        bs = len(T)
        H, W = self.bev_h, self.bev_w
        
        # 创建掩码 [bs, H*W]，初始均为有效(False)
        # 注意：在transformer中，padding_mask中的True表示需要被mask的位置
        query_key_padding_mask = torch.zeros((bs, H*W), device=device, dtype=torch.bool)
        
        # 标记每个样本的无效区域
        for b in range(bs):
            shift_pixels = int(T[b, 0] / (self.real_w / W))
            if shift_pixels > 0:  # 右移，左侧无效
                for h in range(H):
                    mask_range = slice(h*W, h*W + shift_pixels)
                    query_key_padding_mask[b, mask_range] = True
            elif shift_pixels < 0:  # 左移，右侧无效
                for h in range(H):
                    mask_range = slice(h*W + W + shift_pixels, (h+1)*W)
                    query_key_padding_mask[b, mask_range] = True
        
        return query_key_padding_mask

    # @auto_fp16(apply_to=('mlvl_feats'))
    @force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self,
                mlvl_feats,
                img_metas,
                prev_bev=None,
                only_bev=False,
                ego_his_trajs=None,
                ego_lcf_feat=None,
                gt_labels_3d=None,
                gt_attr_labels=None,
                ego_fut_trajs=None,
                ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        # query_embedding是在_init_layers中定义的

        if self.map_query_embed_type == 'all_pts':
            map_query_embeds = self.map_query_embedding.weight.to(dtype)
        elif self.map_query_embed_type == 'instance_pts':
            map_pts_embeds = self.map_pts_embedding.weight.unsqueeze(0)
            map_instance_embeds = self.map_instance_embedding.weight.unsqueeze(1)
            map_query_embeds = (map_pts_embeds + map_instance_embeds).flatten(0, 1).to(dtype)

        # 初始化BEV查询向量
        # bev_queries是BEV特征的查询向量(query embedding)，用于transformer中的注意力机制
        # 可学习的嵌入权重矩阵
        bev_queries = self.bev_embedding.weight.to(dtype)
        # bev_queries.shape -- [10000,256]

        # ============ 生成自车位置偏移的BEV特征 ============ #
        # 定义T
        T = torch.zeros((bs, 3), device=mlvl_feats[0].device)

        # ============ Gasussian noise  ============ #
        mean = 0.0  # 均值，表示在自车位置附近生成偏移量
        std = 1.0   # 标准差，约为车体宽度的1/2
        device = mlvl_feats[0].device
        # ------- 同一场景不同偏移 ------- #
        # # （同一批次内所有样本相同偏移，不同批次不同偏移）
        # # 生成单一偏移值并复制到整个批次
        # single_shift = torch.normal(mean=mean, std=std, size=(1,), device=mlvl_feats[0].device)
        # lateral_shift = single_shift.repeat(bs)
        # # torch.manual_seed(42)  # 确保每次生成相同的随机数
        # # # （同一批次内所有样本不同偏移，不同批次不同偏移）
        # # lateral_shift = torch.normal(mean=mean, std=std, size=(bs,), device=mlvl_feats[0].device)
        # # ------- 同一场景不同偏移 ------- #

        # -------同一场景相同偏移------- #
        # 1. 尝试从文件加载场景偏移映射
        if os.path.exists(self.shift_file) and not self.scene_shifts:
            try:
                with open(self.shift_file, 'r') as f:
                    self.scene_shifts = json.load(f)
            except:
                self.scene_shifts = {}
        
        # 2. 收集需要新偏移值的场景
        new_scenes = []
        for meta in img_metas:
            scene_token = meta['scene_token']
            if scene_token not in self.scene_shifts:
                new_scenes.append(scene_token)
                
        # 3. 为新场景生成偏移值
        if new_scenes:
            # 获取所有现有偏移值
            existing_shifts = list(map(float, self.scene_shifts.values())) if self.scene_shifts else []
            
            # 为新场景生成不重复的偏移值
            for scene in new_scenes:
                while True:
                    # 生成符合高斯分布的候选偏移值
                    shift = float(self.shift_rng.normal(mean, std))
                    # 确保与现有值有足够差异
                    if all(abs(shift - ex_shift) > 0.01 for ex_shift in existing_shifts):
                        self.scene_shifts[scene] = shift
                        existing_shifts.append(shift)
                        break
            # 4. 保存更新后的映射
            try:
                os.makedirs(os.path.dirname(self.shift_file), exist_ok=True)
                with open(self.shift_file, 'w') as f:
                    json.dump(self.scene_shifts, f)
            except:
                print("Warning: Failed to save scene shifts file")

        # 5. 应用偏移值到当前批次
        lateral_shift = torch.zeros((bs,), device=device)
        # -------同一场景相同偏移------- #

        # 记录偏移量和相关信息
        lidar_file = img_metas[0]['pts_filename']
        timestamp = re.search(r'__(\d+)\.pcd\.bin$', lidar_file).group(1)

        for b in range(bs):
            scene_token = img_metas[b]['scene_token']
            lateral_shift[b] = torch.tensor(float(self.scene_shifts[scene_token]), device=device)
            
            # 记录偏移信息
            shift_info = {
                'lateral_shift': np.array([float(self.scene_shifts[scene_token])]),
                'timestamp': re.search(r'__(\d+)\.pcd\.bin$', img_metas[b]['pts_filename']).group(1),
                'scene_token': scene_token,
                'sample_idx': img_metas[b]['sample_idx']
            }
            self.shift_log.append(shift_info)
        self.save_shift_log()

         # ----- 设置固定的偏移量（用于验证bev_embed的具体值）----- #
        # 计算恰好移动10个网格（对应sample_cols中的间距）的物理距离
        # grid_shift = 10  # 移动的网格数量，与sample_cols间隔一致
        # shift_distance = grid_shift * (self.real_w / self.bev_w)
        # T[:, 0] = -shift_distance
        # lateral_shift = torch.ones((bs,), device=mlvl_feats[0].device) * shift_distance
        # ----- 设置固定的偏移量（用于验证bev_embed的具体值）----- #
        
        T[:, 0] = lateral_shift # x方向按高斯分布生成偏移量
        # T[:, 0] = 2.0  # x方向平移2米
        # T[:, 1] = 0.0  # y方向不平移
        # T[:, 2] = 0.0  # 旋转角度保持为零（默认就是零）

        # bev_mask用于标记哪些区域需要计算位置编码
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        # positional_encoding指向LearnedPositionalEncoding类，定义在/mnt/kuebiko/users/qdeng/anaconda3/envs/genad/lib/python3.8/site-packages/mmdet/models/utils/positional_encoding.py路径下
        bev_pos = self.positional_encoding(bev_mask)
        padding_mode = 'reflection'  # 'zeros', 'replicate', 'reflection', 'edge_mask'

        # *** MTG_20250328(option1): 相对自车位置进行shift，不更改bev_embed的值，仅修改bev_pos的值
        # *** MTG_20250328(option3): 相对自车位置进行shift，不更改bev_pos的值，仅修改bev_embed的值
        # 无padding
        bev_pos_t =self.se2_transform(bev_pos, T)
        # 有padding
        # bev_pos_t_p = self.improved_se2_transform(bev_pos, T, padding_mode=padding_mode)
        # ============ 生成自车位置偏移的BEV特征 ============ #

        # =========== 原代码 =========== #
        # # 初始化BEV mask
        # # bev_mask表示BEV视图中的有效区域掩码,在本代码中是全零矩阵，也不会更新        
        # bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
        #                        device=bev_queries.device).to(dtype)
        # # bev_mask.shape -- [1,100,100]

        # # PE
        # # 位置编码作用于key（mlvl_feats）和query(bev_queries)
        # # positional_encoding指向LearnedPositionalEncoding类，定义在/mnt/kuebiko/users/qdeng/anaconda3/envs/genad/lib/python3.8/site-packages/mmdet/models/utils/positional_encoding.py路径下
        # bev_pos = self.positional_encoding(bev_mask).to(dtype)
        # # bev_pos.shape -- [1,256,100,100]
        # =========== 原代码 =========== #

        # 只使用encoder提取BEV特征
        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                # bev_pos=bev_pos_t,
                # bev_pos=bev_pos_t_p,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        # 调用GenAD_transformer的forward函数
        # Call the forward function of GenAD_transformer
        else:
            outputs = self.transformer(
                mlvl_feats,             # 多视角图像特征，作为 Key 和 Value
                bev_queries,            # BEV查询向量，作为 Query。学习"每个位置应该表示什么内容"
                object_query_embeds,
                map_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,        # 用于位置编码。学习"这是空间中的哪个位置"
                # bev_pos=bev_pos_t,
                # bev_pos=bev_pos_t_p,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                map_reg_branches=self.map_reg_branches if self.with_box_refine else None,  # noqa:E501
                map_cls_branches=self.map_cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev
            )

        # bev_embed: bev features
        # hs: agent_query
        # init_reference: reference points init
        # inter_references: reference points processing
        # map_hs: map_query
        # map_init_reference: reference points init
        # map_inter_references: reference points processing

        bev_embed, hs, init_reference, inter_references, \
        map_hs, map_init_reference, map_inter_references = outputs
        # （和uniad的uniad_track.py中的forward_track_train函数进行对比）bev_embed是单帧的结果，不会被更新或修改
        
        # 无padding
        # bev_embed_t = self.se2_transform(bev_embed, T)  # T是自车位置偏移的变换矩阵
        # 有padding
        # bev_embed_t_p = self.improved_se2_transform(bev_embed, T, padding_mode=padding_mode)

        # bev_embed.shape --[10000, 1, 256]，10000代表100*100个grid，1代表1个batch，256代表特征维度
        # config文件中设置了bev_h_ = 100，bev_w_ = 100

        # ========= 特征值检查 ========= #
        # # 将[H*W, B, C]形状重塑为[H, W, B, C]以便于按grid位置比较
        # bev_h = self.bev_h  # 100
        # bev_w = self.bev_w  # 100
        # bev_embed_reshaped = bev_embed.reshape(bev_h, bev_w, -1, bev_embed.shape[-1])
        # # bev_embed_t_reshaped = bev_embed_t.reshape(bev_h, bev_w, -1, bev_embed_t.shape[-1])
        # # bev_embed_t_p_reshaped = bev_embed_t_p.reshape(bev_h, bev_w, -1, bev_embed_t_p.shape[-1])

        # # 计算横向偏移量（以网格数为单位）
        # shift_in_grids = int(lateral_shift[0].item() / (self.real_w / self.bev_w))
        # # print(f"Lateral shift: {lateral_shift[0].item():.3f}m, approximately {shift_in_grids} grids")

        # # 选择关键位置进行比较
        # positions = {
        #     'center': (bev_h // 2, bev_w // 2),                # 中心位置
        #     'left_edge': (bev_h // 2, 0),                      # 左边缘
        #     'right_edge': (bev_h // 2, bev_w - 1),             # 右边缘
        #     'shifted_ref': (bev_h // 2, bev_w // 2 - shift_in_grids)  # 预期偏移后对应的位置
        # }

        # # 创建log目录
        # log_dir = '/mnt/kuebiko/users/qdeng/GenAD/bev_feature_compare_logs'
        # os.makedirs(log_dir, exist_ok=True)

        # # 获取当前时间戳和场景信息
        # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # scene_token = img_metas[0]['scene_token']
        # sample_idx = img_metas[0]['sample_idx']
        # log_file = os.path.join(log_dir, f'bev_compare_{scene_token}_{sample_idx}_{timestamp}.txt')

        # with open(log_file, 'w') as f:
        #     f.write(f"BEV Embedding Comparison\n")
        #     f.write(f"=======================\n")
        #     f.write(f"Lateral shift: {lateral_shift[0].item():.3f}m, approximately {shift_in_grids} grids\n\n")
            
        #     # 比较特定位置的特征值
        #     for pos_name, (h, w) in positions.items():
        #         if 0 <= w < bev_w:  # 确保位置在有效范围内
        #             f.write(f"Position: {pos_name} ({h}, {w})\n")
        #             f.write(f"Original:          {bev_embed_reshaped[h, w, 0, :5].cpu().detach().numpy()}\n")
        #             f.write(f"Shifted (no pad):  {bev_embed_t_reshaped[h, w, 0, :5].cpu().detach().numpy()}\n")
        #             f.write(f"Shifted (with pad):{bev_embed_t_p_reshaped[h, w, 0, :5].cpu().detach().numpy()}\n\n")
            
        #     # 检查横向连续网格的值变化
        #     row_idx = bev_h // 2  # 中心行
        #     f.write(f"Values along middle row (row {row_idx}), showing 1st feature dimension:\n")
            
        #     # 选择11个等间距的列位置（包括中心和边缘）
        #     sample_cols = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
            
        #     f.write("Column index:       " + " ".join([f"{c:6d}" for c in sample_cols]) + "\n")
            
        #     # 显示第一个特征维度的值
        #     feature_idx = 0
            
        #     # 原始BEV特征
        #     values_orig = [f"{bev_embed_reshaped[row_idx, c, 0, feature_idx].item():.1f}" for c in sample_cols]
        #     f.write("Original:           " + " ".join([f"{v:6s}" for v in values_orig]) + "\n")
            
        #     # 无padding的偏移BEV特征
        #     values_shift = [f"{bev_embed_t_reshaped[row_idx, c, 0, feature_idx].item():.1f}" for c in sample_cols]
        #     f.write("Shifted (no pad):   " + " ".join([f"{v:6s}" for v in values_shift]) + "\n")
            
        #     # 有padding的偏移BEV特征
        #     values_shift_pad = [f"{bev_embed_t_p_reshaped[row_idx, c, 0, feature_idx].item():.1f}" for c in sample_cols]
        #     f.write("Shifted (with pad): " + " ".join([f"{v:6s}" for v in values_shift_pad]) + "\n\n")
            
        #     # 检查预期的偏移效果
        #     if 0 <= shift_in_grids < bev_w:
        #         f.write("Checking expected shift pattern:\n")
        #         for col in range(30, 70, 5):  # 只检查中间部分的几个列
        #             if 0 <= col < bev_w and 0 <= col + shift_in_grids < bev_w:
        #                 orig_val = bev_embed_reshaped[row_idx, col, 0, :3].cpu().detach().numpy()
        #                 shifted_val = bev_embed_t_reshaped[row_idx, col + shift_in_grids, 0, :3].cpu().detach().numpy()
        #                 f.write(f"Original at col {col}: {orig_val}\n")
        #                 f.write(f"Shifted at col {col + shift_in_grids}: {shifted_val}\n")
        #                 if np.allclose(orig_val, shifted_val, rtol=1e-1, atol=1e-1):
        #                     f.write("MATCH ✓\n\n")
        #                 else:
        #                     f.write("DIFFERENT ✗\n\n")

        # # print(f"BEV feature comparison saved to {log_file}")
        # ========= 特征值检查 ========= #

        # save BEV features
        # 为了保证可视化代码的一致性，不更改key的名称和变量维度等信息
        def save_bev_features(bev_features, img_metas, bev_h, bev_w, base_path):
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
        
        save_bev_features(bev_embed, img_metas, self.bev_h, self.bev_w, base_path='/mnt/kuebiko/users/qdeng/GenAD/bev_features_pretrained')

        # 保存路径命名规则--bev_features_[进行的操作（shift/rotation/padding..）_[模型名称]_[epoch数]
        # save_bev_features(bev_embed, img_metas, self.bev_h, self.bev_w, base_path='/mnt/kuebiko/users/qdeng/GenAD/bev_features_pretrained')
        # save_bev_features(bev_embed_t, img_metas, self.bev_h, self.bev_w, base_path='/mnt/kuebiko/users/qdeng/GenAD/bev_features_lateral_shift_pretrained')
        # save_bev_features(bev_embed_t_p, img_metas, self.bev_h, self.bev_w, base_path='/mnt/kuebiko/users/qdeng/GenAD/bev_features_lateral_shift_padding_pretrained')

        # hs.shape --[3,300,1,256] = [num_decoder_layers, num_query, batch_size, embed_dims]
        # 维度调整
        hs = hs.permute(0, 2, 1, 3)                 # agent_query特征
        # hs 是来自 transformer decoder 的层级化输出特征
        # hs的原始维度: [num_layers, num_query, batch_size, embed_dims]
        # hs permute后的维度: [num_layers, batch_size, num_query, embed_dims]
        outputs_classes = []                        # 检测分类
        outputs_coords = []                         # 检测框坐标，实际物理坐标
        outputs_coords_bev = []                     # 检测框BEV坐标，在BEV特征图上的归一化坐标，值域在[0,1]之间
        outputs_trajs = []                          # 轨迹预测
        outputs_trajs_classes = []                  # 轨迹分类

        map_hs = map_hs.permute(0, 2, 1, 3)        # map_query特征
        map_outputs_classes = []
        map_outputs_coords = []
        map_outputs_pts_coords = []
        map_outputs_coords_bev = []

        # 目标检测解码
        for lvl in range(hs.shape[0]):
            # 获取参考点
            if lvl == 0:
                reference = init_reference                  # 第一层的参考点, 表示初始预测的目标位置，shape: [batch_size, num_queries, 3]
            else:
                reference = inter_references[lvl - 1]       # 中间层的参考点, ，表示经过一层refinement后更新的目标位置，shape: [batch_size, num_queries, 3]
            reference = inverse_sigmoid(reference)          # 将sigmoid值域[0,1]转回原始预测值域
            # 分类和坐标回归
            outputs_class = self.cls_branches[lvl](hs[lvl]) # 分类预测
            tmp = self.reg_branches[lvl](hs[lvl])           # 回归预测 （box_dim维度的预测值）

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            # xy平面坐标解码
            tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2] # 加上参考点的xy偏移量
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()             # 归一化到[0,1]之间
            outputs_coords_bev.append(tmp[..., 0:2].clone().detach())
            # z轴坐标解码
            tmp[..., 4:5] = tmp[..., 4:5] + reference[..., 2:3] # 加上参考点的高度（z）偏移量
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()             # 归一化到[0,1]之间
            # 坐标转换到实际尺度
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                                              self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                                              self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                                              self.pc_range[2]) + self.pc_range[2])

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)        # outputs_coords--[num_decoder_layers, batch_size, num_queries, box_dim]
            # outputs_coords的最后一维box_dim应该包含以下信息：[x, y, w, l, z, h, rot, vx, vy]

        # 地图元素解码
        for lvl in range(map_hs.shape[0]):
            if lvl == 0:
                reference = map_init_reference
            else:
                reference = map_inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            # 地图元素分类
            map_outputs_class = self.map_cls_branches[lvl](
                map_hs[lvl].view(bs, self.map_num_vec, self.map_num_pts_per_vec, -1).mean(2)
            )
            # 地图元素坐标回归
            tmp = self.map_reg_branches[lvl](map_hs[lvl])
            # TODO: check the shape of reference
            assert reference.shape[-1] == 2
            tmp[..., 0:2] += reference[..., 0:2]
            tmp = tmp.sigmoid()  # cx,cy,w,h
            # 坐标变换
            map_outputs_coord, map_outputs_pts_coord = self.map_transform_box(tmp)
            # 保存结果
            map_outputs_coords_bev.append(map_outputs_pts_coord.clone().detach())
            map_outputs_classes.append(map_outputs_class)
            map_outputs_coords.append(map_outputs_coord)
            map_outputs_pts_coords.append(map_outputs_pts_coord)

        # motion prediction

        # motion query
        # 运动预测
        if self.motion_decoder is not None:
            batch_size, num_agent = outputs_coords_bev[-1].shape[:2]
            # 提取motion_query
            motion_query = hs[-1].permute(1, 0, 2)  # [A, B, D]
            # 多运动模态预测
            mode_query = self.motion_mode_query.weight  # [fut_mode, D]
            # [M, B, D], M=A*fut_mode
            motion_query = (motion_query[:, None, :, :] + mode_query[None, :, None, :]).flatten(0, 1)
            # 运动位置编码
            if self.use_pe:
                motion_coords = outputs_coords_bev[-1]  # [B, A, 2]
                motion_pos = self.pos_mlp_sa(motion_coords)  # [B, A, D]
                motion_pos = motion_pos.unsqueeze(2).repeat(1, 1, self.fut_mode, 1).flatten(1, 2)
                motion_pos = motion_pos.permute(1, 0, 2)  # [M, B, D]
            else:
                motion_pos = None

            if self.motion_det_score is not None:
                motion_score = outputs_classes[-1]
                max_motion_score = motion_score.max(dim=-1)[0]
                invalid_motion_idx = max_motion_score < self.motion_det_score  # [B, A]
                invalid_motion_idx = invalid_motion_idx.unsqueeze(2).repeat(1, 1, self.fut_mode).flatten(1, 2)
            else:
                invalid_motion_idx = None

            # ego query
            # batch = batch_size
            if self.ego_his_encoder is not None:
                ego_his_feats = self.ego_his_encoder(ego_his_trajs)  # [B, 1, dim]
            else:
                ego_his_feats = self.ego_query.weight.unsqueeze(0).repeat(batch_size, 1, 1)
                # ego <-> agent Interaction
            ego_query = ego_his_feats.permute(1, 0, 2)
            ego_pos = torch.zeros((batch_size, 1, 2), device=ego_query.device).permute(1, 0, 2)
            # ego_pos.shape -- [1,1,2]=[1, B, 2]
            # 需要根据bev_pos_t的变化量相应地变换ego_pos
            # ego_pos = self.se2_transform(ego_pos, T)        # T是自车位置偏移的变换矩阵
            
            ego_pos_emb = self.ego_agent_pos_mlp(ego_pos)

            motion_query = torch.cat([motion_query, ego_query], dim=0)
            motion_pos = torch.cat([motion_pos, ego_pos_emb], dim=0)

            # Instance Encoder (论文中的Self-attention + Cross-attention)
            motion_hs = self.motion_decoder(
                query=motion_query,
                key=motion_query,
                value=motion_query,
                query_pos=motion_pos,
                key_pos=motion_pos,
                key_padding_mask=invalid_motion_idx)

            if self.motion_map_decoder is not None:
                # map preprocess
                motion_coords = outputs_coords_bev[-1]  # [B, A, 2]
                motion_coords = motion_coords.unsqueeze(2).repeat(1, 1, self.fut_mode, 1).flatten(1, 2)

                # ego_coords = torch.Tensor(1, 1, 2).cuda(1)
                ego_coords = torch.zeros([batch_size, 1, 2], device=motion_hs.device)
                ego_coords_embd = self.ego_coord_mlp(ego_coords)
                # ego_coords_embd = torch.zeros([batch_size, 1, 2], device=motion_hs.device)
                motion_coords = torch.cat([motion_coords, ego_coords_embd], dim=1)

                map_query = map_hs[-1].view(batch_size, self.map_num_vec, self.map_num_pts_per_vec, -1)
                map_query = self.lane_encoder(map_query)  # [B, P, pts, D] -> [B, P, D]
                map_score = map_outputs_classes[-1]
                map_pos = map_outputs_coords_bev[-1]
                map_query, map_pos, key_padding_mask = self.select_and_pad_pred_map(
                    motion_coords, map_query, map_score, map_pos,
                    map_thresh=self.map_thresh, dis_thresh=self.dis_thresh,
                    pe_normalization=self.pe_normalization, use_fix_pad=True)
                map_query = map_query.permute(1, 0, 2)  # [P, B*M, D]
                ca_motion_query = motion_hs.permute(1, 0, 2).flatten(0, 1).unsqueeze(0)

                # position encoding
                if self.use_pe:
                    (num_query, batch) = ca_motion_query.shape[:2]
                    motion_pos = torch.zeros((num_query, batch, 2), device=motion_hs.device)
                    motion_pos = self.pos_mlp(motion_pos)
                    map_pos = map_pos.permute(1, 0, 2)
                    map_pos = self.pos_mlp(map_pos)
                else:
                    motion_pos, map_pos = None, None

                # Agent-Map Cross-attention
                ca_motion_query = self.motion_map_decoder(
                    query=ca_motion_query,
                    key=map_query,
                    value=map_query,
                    query_pos=motion_pos,
                    key_pos=map_pos,
                    key_padding_mask=key_padding_mask)
            else:
                ca_motion_query = motion_hs.permute(1, 0, 2).flatten(0, 1).unsqueeze(0)

            ########################################
            # generator for planning & motion
            current_states = torch.cat((motion_hs.permute(1, 0, 2), ca_motion_query.reshape(batch_size, -1, self.embed_dims)), dim=2)
            distribution_comp = {}
            # states = torch.randn((2, 1, 64, 200, 200), device=motion_hs.device)
            # future_distribution_inputs = torch.randn((2, 5, 6, 200, 200), device=motion_hs.device)
            noise = None
            # 为VAE（变分自编码器）提供未来真实状态作为条件，用于生成未来轨迹的条件概率分布
            if self.training:       # 在训练阶段，有真实的未来轨迹可以用来指导模型学习
                # future_distribution_inputs即论文中的ground-truth trajectories
                # 处理未来轨迹的ground-truth
                future_distribution_inputs = self.get_future_labels(gt_labels_3d, gt_attr_labels,
                                                                    ego_fut_trajs, motion_hs.device) 
                # print("future_distribution_inputs shape: ", future_distribution_inputs.shape)
            else:                   # 在推理阶段，没有未来轨迹，所以设为None
                future_distribution_inputs = None

            # 1. model CVA distribution for state
            if self.fut_ts > 0:
                # present_state = states[:, :1].contiguous()
                if self.probabilistic:
                    # Do probabilistic computation
                    # VAE encoder，对应论文中的future trajectory encoder
                    sample, output_distribution = self.distribution_forward(
                        current_states, future_distribution_inputs, noise
                    )
                    distribution_comp = {**distribution_comp, **output_distribution}

            # 2. predict future state from distribution
            hidden_states = current_states
            # 基于采样生成（预测）未来状态，对应论文中的future trajectory generator
            states_hs, future_states_hs = self.future_states_predict(
                batch_size=batch_size,
                sample=sample,
                hidden_states=hidden_states,
                current_states=current_states
            )

            # 提取ego相关的查询特征
            ego_query_hs = states_hs[:, :, self.agent_dim * self.fut_mode, :].unsqueeze(1).permute(0, 2, 1, 3)
            motion_query_hs = states_hs[:, :, 0:self.agent_dim * self.fut_mode, :]
            motion_query_hs = motion_query_hs.reshape(self.fut_ts, batch_size, -1, self.fut_ts, motion_query_hs.shape[-1])
            
            # ego-vehicle和agents的轨迹预测
            ego_fut_trajs_list = []         # 构建轨迹预测列表
            motion_fut_trajs_list = []

            # ==== 原有的轨迹预测代码 ==== #
            # for i in range(self.fut_ts):
            #     # 对每个未来时间步解码ego轨迹
            #     outputs_ego_trajs = self.ego_fut_decoder(ego_query_hs[i]).reshape(batch_size, self.ego_fut_mode, 2)
            #     ego_fut_trajs_list.append(outputs_ego_trajs)
            #     # 对每个未来时间步解码agent轨迹
            #     outputs_agent_trajs = self.traj_branches[0](motion_query_hs[i])
            #     motion_fut_trajs_list.append(outputs_agent_trajs)
            # # 将轨迹列表堆叠为统一张量
            # ego_trajs = torch.stack(ego_fut_trajs_list, dim=2)
            # ==== 原有的轨迹预测代码 ==== #

            # ==== joy修改轨迹预测代码 ==== #
            for i in range(self.fut_ts):
                # 生成agent轨迹 (这部分在所有条件下都要执行)
                outputs_agent_trajs = self.traj_branches[0](motion_query_hs[i])
                motion_fut_trajs_list.append(outputs_agent_trajs)

            # 根据lateral_shift生成恢复轨迹作为ego未来轨迹
            # print("ego_fut_trajs: ", ego_fut_trajs)

            # recovery_ego_trajs, recovery_trajectories = self.generate_ego_future_from_recovery(lateral_shift,ego_fut_trajs=ego_fut_trajs)
            recovery_ego_trajs, recovery_trajectories = self.generate_ego_future_from_recovery(
                lateral_shift, 
                ego_fut_trajs=ego_fut_trajs, 
                base_future_length=6, 
                time_interval=0.5,
                lookahead_method='time', 
                min_lookahead=2.0, 
                max_lookahead=5.0, 
                speed_factor=0.5, 
                lookahead_time=5.0, 
                steering_smoothing=0.9, 
                decay_constant=6.0)

            if self.training:  # 训练阶段：随机选择是否使用恢复轨迹
                # 设置偏移阈值，超过此值必须使用恢复轨迹
                recovery_threshold = 0.2  # 20cm
                
                if torch.abs(lateral_shift).max() > recovery_threshold: # 明显偏移时，始终使用恢复轨迹
                    ego_trajs = recovery_ego_trajs
                    # 可以添加小扰动提高泛化性
                    noise_scale = 0.05  # 5cm随机扰动
                    noise = torch.randn_like(ego_trajs) * noise_scale
                    ego_trajs = ego_trajs + noise
                else:   # 微小偏移，80%概率使用恢复轨迹
                    if random.random() < 0.8:
                        ego_trajs = recovery_ego_trajs
                    else:
                        # 原有的ego轨迹生成代码
                        ego_fut_trajs_list = []
                        for i in range(self.fut_ts):
                            outputs_ego_trajs = self.ego_fut_decoder(ego_query_hs[i]).reshape(batch_size, self.ego_fut_mode, 2)
                            ego_fut_trajs_list.append(outputs_ego_trajs)
                        ego_trajs = torch.stack(ego_fut_trajs_list, dim=2)
            else:  # 评估阶段：当有明显偏移时使用恢复轨迹，否则使用原轨迹
                lateral_shift_abs = torch.abs(lateral_shift)
                use_recovery = lateral_shift_abs > 0.3  # 偏移超过0.3米时使用恢复轨迹
                
                if use_recovery.any():
                    # 对每个样本分别处理
                    ego_fut_trajs_list = []
                    for i in range(self.fut_ts):
                        outputs_ego_trajs = self.ego_fut_decoder(ego_query_hs[i]).reshape(batch_size, self.ego_fut_mode, 2)
                        ego_fut_trajs_list.append(outputs_ego_trajs)
                    original_ego_trajs = torch.stack(ego_fut_trajs_list, dim=2)
                    
                    # 创建结果张量
                    ego_trajs = original_ego_trajs.clone()
                    
                    # 对需要恢复的样本使用恢复轨迹
                    for b in range(batch_size):
                        if use_recovery[b]:
                            ego_trajs[b] = recovery_ego_trajs[b]
                else:
                    # 所有样本都使用原始轨迹
                    ego_fut_trajs_list = []
                    for i in range(self.fut_ts):
                        outputs_ego_trajs = self.ego_fut_decoder(ego_query_hs[i]).reshape(batch_size, self.ego_fut_mode, 2)
                        ego_fut_trajs_list.append(outputs_ego_trajs)
                    ego_trajs = torch.stack(ego_fut_trajs_list, dim=2)
            # ==== joy修改轨迹预测代码 ==== #

            agent_trajs = torch.stack(motion_fut_trajs_list, dim=3)
            agent_trajs = agent_trajs.reshape(batch_size, 1, self.agent_dim, self.fut_mode, -1)

        # future_hs = future_states_hs[:, :, 0:self.agent_dim * self.fut_mode, :].reshape(
        #     batch_size, self.agent_dim, self.fut_mode, -1)
        # current_hs = current_states[:, 0:self.agent_dim * self.fut_mode, :].reshape(
        #     batch_size, self.agent_dim, self.fut_mode, -1)
        #
        # motion_cls_hs = torch.cat((future_hs, current_hs), dim=-1)
        motion_cls_hs = torch.cat((future_states_hs[:, :, 0:self.agent_dim * self.fut_mode, :].
                                   reshape(batch_size, self.agent_dim, self.fut_mode, -1),
                                   current_states[:, 0:self.agent_dim * self.fut_mode, :].
                                   reshape(batch_size, self.agent_dim, self.fut_mode, -1)), dim=-1)

        outputs_traj_class = self.traj_cls_branches[0](motion_cls_hs)
        outputs_trajs_classes.append(outputs_traj_class.squeeze(-1))

        map_outputs_classes = torch.stack(map_outputs_classes)
        map_outputs_coords = torch.stack(map_outputs_coords)
        map_outputs_pts_coords = torch.stack(map_outputs_pts_coords)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_trajs = agent_trajs.permute(1, 0, 2, 3, 4)
        outputs_trajs_classes = torch.stack(outputs_trajs_classes)
        # outputs_trajs = outputs_trajs.repeat(outputs_coords.shape[0], 1, 1, 1, 1)
        # outputs_trajs_classes = outputs_trajs_classes.repeat(outputs_coords.shape[0], 1, 1, 1)

        outs = {
            # 'bev_embed': bev_embed,                                                                 # [10000, 1, 256]
            # 'bev_embed': bev_embed_t,                                                             # [10000, 1, 256]
            'bev_embed': bev_embed_t_p,                                                           # [10000, 1, 256]
            'all_cls_scores': outputs_classes,                                                      # [3, 1, 300, 10]
            'all_bbox_preds': outputs_coords,                                                       # [3, 1, 300, 10], agents当前位置等检测结果
            'all_traj_preds': outputs_trajs.repeat(outputs_coords.shape[0], 1, 1, 1, 1),            # [3, 1, 300, 6, 12],agents预测轨迹
            'all_traj_cls_scores': outputs_trajs_classes.repeat(outputs_coords.shape[0], 1, 1, 1),  # [3, 1, 300, 6]
            'map_all_cls_scores': map_outputs_classes,                                              # [3, 1, 100, 3]
            'map_all_bbox_preds': map_outputs_coords,                                               # [3, 1, 100, 4]
            'map_all_pts_preds': map_outputs_pts_coords,                                            # [3, 1, 100, 20, 2],地图元素
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'map_enc_cls_scores': None,
            'map_enc_bbox_preds': None,
            'map_enc_pts_preds': None,
            'ego_fut_preds': ego_trajs,                                                             # ego的预测轨迹值，shape-[1, 3, 6, 2]
            'loss_vae_gen': distribution_comp,                                                      # present_mu[1,1,32]/present_log_sigma[1,1,32]/future_mu[1,1,32]/future_log_sigma[1,1,32]
            'lateral_shift': lateral_shift,                                                         # 保存偏移量
            'recovery_ego_trajs': recovery_ego_trajs,       # 添加恢复轨迹
            'padding_mode': padding_mode,  # 记录使用的填充模式
        }

        # 可视化和保存恢复轨迹
        self.visualization_counter += 1
        
        # 定期可视化轨迹
        if self.visualization_counter % self.visualize_freq == 0:
            # self.visualize_recovery_trajectory(lateral_shift, img_metas, ego_fut_trajs=ego_fut_trajs, recovery_trajectories=recovery_trajectories, ego_futures=recovery_ego_trajs)

            self.improved_visualize_recovery_trajectory(
                lateral_shift, 
                img_metas, 
                ego_fut_trajs=ego_fut_trajs,
                recovery_trajectories=recovery_trajectories,  # 使用已计算的恢复轨迹
                ego_futures=recovery_ego_trajs,              # 使用已计算的ego未来轨迹
                lookahead_method='time',
                min_lookahead=2.0, 
                max_lookahead=6.0,
                speed_factor=0.5,
                lookahead_time=5.0,
                steering_smoothing=0.9, 
                decay_constant=6.0
            )
        
        # 定期保存轨迹数据
        # if self.visualization_counter % self.save_data_freq == 0:
        #     self.save_recovery_trajectories(lateral_shift, img_metas, ego_fut_trajs=ego_fut_trajs)

        # print("outs",outs)

        return outs

    def map_transform_box(self, pts, y_first=False):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """
        pts_reshape = pts.view(pts.shape[0], self.map_num_vec,
                               self.map_num_pts_per_vec, 2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.map_transform_method == 'minmax':
            # import pdb;pdb.set_trace()

            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_attr_labels,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 10].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 9) in [x,y,z,w,l,h,yaw,vx,vy] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_fut_trajs = gt_attr_labels[:, :self.fut_ts * 2]
        gt_fut_masks = gt_attr_labels[:, self.fut_ts * 2:self.fut_ts * 3]
        gt_bbox_c = gt_bboxes.shape[-1]
        num_gt_bbox, gt_traj_c = gt_fut_trajs.shape

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_bbox_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # trajs targets
        traj_targets = torch.zeros((num_bboxes, gt_traj_c), dtype=torch.float32, device=bbox_pred.device)
        traj_weights = torch.zeros_like(traj_targets)
        traj_targets[pos_inds] = gt_fut_trajs[sampling_result.pos_assigned_gt_inds]
        traj_weights[pos_inds] = 1.0

        # Filter out invalid fut trajs
        traj_masks = torch.zeros_like(traj_targets)  # [num_bboxes, fut_ts*2]
        gt_fut_masks = gt_fut_masks.unsqueeze(-1).repeat(1, 1, 2).view(num_gt_bbox, -1)  # [num_gt_bbox, fut_ts*2]
        traj_masks[pos_inds] = gt_fut_masks[sampling_result.pos_assigned_gt_inds]
        traj_weights = traj_weights * traj_masks

        # Extra future timestamp mask for controlling pred horizon
        fut_ts_mask = torch.zeros((num_bboxes, self.fut_ts, 2),
                                  dtype=torch.float32, device=bbox_pred.device)
        fut_ts_mask[:, :self.valid_fut_ts, :] = 1.0
        fut_ts_mask = fut_ts_mask.view(num_bboxes, -1)
        traj_weights = traj_weights * fut_ts_mask

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return (
            labels, label_weights, bbox_targets, bbox_weights, traj_targets,
            traj_weights, traj_masks.view(-1, self.fut_ts, 2)[..., 0],
            pos_inds, neg_inds
        )

    def _map_get_target_single(self,
                               cls_score,
                               bbox_pred,
                               pts_pred,
                               gt_labels,
                               gt_bboxes,
                               gt_shifts_pts,
                               gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        assign_result, order_index = self.map_assigner.assign(bbox_pred, cls_score, pts_pred,
                                                              gt_bboxes, gt_labels, gt_shifts_pts,
                                                              gt_bboxes_ignore)

        sampling_result = self.map_sampler.sample(assign_result, bbox_pred,
                                                  gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.map_num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)
        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        # pts targets
        if order_index is None:
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros((pts_pred.size(0),
                                          pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0
        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds, assigned_shift, :, :]
        return (labels, label_weights, bbox_targets, bbox_weights,
                pts_targets, pts_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_attr_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, traj_targets_list, traj_weights_list,
         gt_fut_masks_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_attr_labels_list, gt_bboxes_ignore_list
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                traj_targets_list, traj_weights_list, gt_fut_masks_list, num_total_pos, num_total_neg)

    def map_get_targets(self,
                        cls_scores_list,
                        bbox_preds_list,
                        pts_preds_list,
                        gt_bboxes_list,
                        gt_labels_list,
                        gt_shifts_pts_list,
                        gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pts_targets_list, pts_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._map_get_target_single, cls_scores_list, bbox_preds_list, pts_preds_list,
            gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list,
                num_total_pos, num_total_neg)

    def loss_planning(self,
                      ego_fut_preds,        # 预测的ego未来轨迹
                      ego_fut_gt,           # 真实的ego未来轨迹
                      ego_fut_masks,
                      ego_fut_cmd,
                      lane_preds,
                      lane_score_preds,
                      agent_preds,
                      agent_fut_preds,
                      agent_score_preds,
                      agent_fut_cls_preds):
        """"Loss function for ego vehicle planning.
        Args:
            ego_fut_preds (Tensor): [B, ego_fut_mode, fut_ts, 2]
            ego_fut_gt (Tensor): [B, fut_ts, 2]
            ego_fut_masks (Tensor): [B, fut_ts]
            ego_fut_cmd (Tensor): [B, ego_fut_mode]
            lane_preds (Tensor): [B, num_vec, num_pts, 2]
            lane_score_preds (Tensor): [B, num_vec, 3]
            agent_preds (Tensor): [B, num_agent, 2]
            agent_fut_preds (Tensor): [B, num_agent, fut_mode, fut_ts, 2]
            agent_score_preds (Tensor): [B, num_agent, 10]
            agent_fut_cls_scores (Tensor): [B, num_agent, fut_mode]
        Returns:
            loss_plan_reg (Tensor): planning reg loss.
            loss_plan_bound (Tensor): planning map boundary constraint loss.
            loss_plan_col (Tensor): planning col constraint loss.
            loss_plan_dir (Tensor): planning directional constraint loss.
        """

        ego_fut_gt = ego_fut_gt.unsqueeze(1).repeat(1, self.ego_fut_mode, 1, 1)
        loss_plan_l1_weight = ego_fut_cmd[..., None, None] * ego_fut_masks[:, None, :, None]
        loss_plan_l1_weight = loss_plan_l1_weight.repeat(1, 1, 1, 2)

        loss_plan_l1 = self.loss_plan_reg(
            ego_fut_preds,
            ego_fut_gt,
            loss_plan_l1_weight
        )

        loss_plan_bound = self.loss_plan_bound(
            ego_fut_preds[ego_fut_cmd == 1],
            lane_preds,
            lane_score_preds,
            weight=ego_fut_masks
        )

        loss_plan_col = self.loss_plan_col(
            ego_fut_preds[ego_fut_cmd == 1],
            agent_preds,
            agent_fut_preds,
            agent_score_preds,
            agent_fut_cls_preds,
            weight=ego_fut_masks[:, :, None].repeat(1, 1, 2)
        )

        loss_plan_dir = self.loss_plan_dir(
            ego_fut_preds[ego_fut_cmd == 1],
            lane_preds,
            lane_score_preds,
            weight=ego_fut_masks
        )

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_plan_l1 = torch.nan_to_num(loss_plan_l1)
            loss_plan_bound = torch.nan_to_num(loss_plan_bound)
            loss_plan_col = torch.nan_to_num(loss_plan_col)
            loss_plan_dir = torch.nan_to_num(loss_plan_dir)

        loss_plan_dict = dict()
        loss_plan_dict['loss_plan_reg'] = loss_plan_l1
        loss_plan_dict['loss_plan_bound'] = loss_plan_bound
        loss_plan_dict['loss_plan_col'] = loss_plan_col
        loss_plan_dict['loss_plan_dir'] = loss_plan_dir

        return loss_plan_dict

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    traj_preds,
                    traj_cls_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_attr_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_attr_labels_list, gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         traj_targets_list, traj_weights_list, gt_fut_masks_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        traj_targets = torch.cat(traj_targets_list, 0)
        traj_weights = torch.cat(traj_weights_list, 0)
        gt_fut_masks = torch.cat(gt_fut_masks_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)

        # traj regression loss
        best_traj_preds = self.get_best_fut_preds(
            traj_preds.reshape(-1, self.fut_mode, self.fut_ts, 2),
            traj_targets.reshape(-1, self.fut_ts, 2), gt_fut_masks)

        neg_inds = (bbox_weights[:, 0] == 0)
        traj_labels = self.get_traj_cls_target(
            traj_preds.reshape(-1, self.fut_mode, self.fut_ts, 2),
            traj_targets.reshape(-1, self.fut_ts, 2),
            gt_fut_masks, neg_inds)

        loss_traj = self.loss_traj(
            best_traj_preds[isnotnan],
            traj_targets[isnotnan],
            traj_weights[isnotnan],
            avg_factor=num_total_pos)

        if self.use_traj_lr_warmup:
            loss_scale_factor = get_traj_warmup_loss_weight(self.epoch, self.tot_epoch)
            loss_traj = loss_scale_factor * loss_traj

        # traj classification loss
        traj_cls_scores = traj_cls_preds.reshape(-1, self.fut_mode)
        # construct weighted avg_factor to match with the official DETR repo
        traj_cls_avg_factor = num_total_pos * 1.0 + \
                              num_total_neg * self.traj_bg_cls_weight
        if self.sync_cls_avg_factor:
            traj_cls_avg_factor = reduce_mean(
                traj_cls_scores.new_tensor([traj_cls_avg_factor]))

        traj_cls_avg_factor = max(traj_cls_avg_factor, 1)
        loss_traj_cls = self.loss_traj_cls(
            traj_cls_scores, traj_labels, label_weights, avg_factor=traj_cls_avg_factor
        )

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_traj = torch.nan_to_num(loss_traj)
            loss_traj_cls = torch.nan_to_num(loss_traj_cls)

        return loss_cls, loss_bbox, loss_traj, loss_traj_cls

    def get_best_fut_preds(self,
                           traj_preds,
                           traj_targets,
                           gt_fut_masks):
        """"Choose best preds among all modes.
        Args:
            traj_preds (Tensor): MultiModal traj preds with shape (num_box_preds, fut_mode, fut_ts, 2).
            traj_targets (Tensor): Ground truth traj for each pred box with shape (num_box_preds, fut_ts, 2).
            gt_fut_masks (Tensor): Ground truth traj mask with shape (num_box_preds, fut_ts).
            pred_box_centers (Tensor): Pred box centers with shape (num_box_preds, 2).
            gt_box_centers (Tensor): Ground truth box centers with shape (num_box_preds, 2).

        Returns:
            best_traj_preds (Tensor): best traj preds (min displacement error with gt)
                with shape (num_box_preds, fut_ts*2).
        """

        cum_traj_preds = traj_preds.cumsum(dim=-2)
        cum_traj_targets = traj_targets.cumsum(dim=-2)

        # Get min pred mode indices.
        # (num_box_preds, fut_mode, fut_ts)
        dist = torch.linalg.norm(cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)
        dist = dist * gt_fut_masks[:, None, :]
        dist = dist[..., -1]
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        min_mode_idxs = torch.argmin(dist, dim=-1).tolist()
        box_idxs = torch.arange(traj_preds.shape[0]).tolist()
        best_traj_preds = traj_preds[box_idxs, min_mode_idxs, :, :].reshape(-1, self.fut_ts * 2)

        return best_traj_preds

    def get_traj_cls_target(self,
                            traj_preds,
                            traj_targets,
                            gt_fut_masks,
                            neg_inds):
        """"Get Trajectory mode classification target.
        Args:
            traj_preds (Tensor): MultiModal traj preds with shape (num_box_preds, fut_mode, fut_ts, 2).
            traj_targets (Tensor): Ground truth traj for each pred box with shape (num_box_preds, fut_ts, 2).
            gt_fut_masks (Tensor): Ground truth traj mask with shape (num_box_preds, fut_ts).
            neg_inds (Tensor): Negtive indices with shape (num_box_preds,)

        Returns:
            traj_labels (Tensor): traj cls labels (num_box_preds,).
        """

        cum_traj_preds = traj_preds.cumsum(dim=-2)
        cum_traj_targets = traj_targets.cumsum(dim=-2)

        # Get min pred mode indices.
        # (num_box_preds, fut_mode, fut_ts)
        dist = torch.linalg.norm(cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)
        dist = dist * gt_fut_masks[:, None, :]
        dist = dist[..., -1]
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        traj_labels = torch.argmin(dist, dim=-1)
        traj_labels[neg_inds] = self.fut_mode

        return traj_labels

    def map_loss_single(self,
                        cls_scores,
                        bbox_preds,
                        pts_preds,
                        gt_bboxes_list,
                        gt_labels_list,
                        gt_shifts_pts_list,
                        gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_pts_list (list[Tensor]): Ground truth pts for each image
                with shape (num_gts, fixed_num, 2) in [x,y] format.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.map_get_targets(cls_scores_list, bbox_preds_list, pts_preds_list,
                                               gt_bboxes_list, gt_labels_list, gt_shifts_pts_list,
                                               gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.map_cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.map_bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_map_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        # normalized_bbox_targets = bbox_targets
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.map_code_weights

        loss_bbox = self.loss_map_bbox(
            bbox_preds[isnotnan, :4],
            normalized_bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos)

        # regression pts CD loss
        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        if self.map_num_pts_per_vec != self.map_num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0, 2, 1)
            pts_preds = F.interpolate(pts_preds, size=(self.map_num_pts_per_gt_vec), mode='linear',
                                      align_corners=True)
            pts_preds = pts_preds.permute(0, 2, 1).contiguous()

        loss_pts = self.loss_map_pts(
            pts_preds[isnotnan, :, :],
            normalized_pts_targets[isnotnan, :, :],
            pts_weights[isnotnan, :, :],
            avg_factor=num_total_pos)

        dir_weights = pts_weights[:, :-self.map_dir_interval, 0]
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = denormed_pts_preds[:, self.map_dir_interval:, :] - \
                                 denormed_pts_preds[:, :-self.map_dir_interval, :]
        pts_targets_dir = pts_targets[:, self.map_dir_interval:, :] - pts_targets[:, :-self.map_dir_interval, :]

        loss_dir = self.loss_map_dir(
            denormed_pts_preds_dir[isnotnan, :, :],
            pts_targets_dir[isnotnan, :, :],
            dir_weights[isnotnan, :],
            avg_factor=num_total_pos)

        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_map_iou(
            bboxes[isnotnan, :4],
            bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)

        return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir

    def distribution_loss(self, output):
        kl_loss = self.loss_vae_gen(output)
        return kl_loss

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             map_gt_bboxes_list,
             map_gt_labels_list,
             preds_dicts,       # 这里接收的是GenAD_head.py中forward函数返回的outs字典
             ego_fut_gt,        # 这里接收的是GenAD.py中forward_pts_train函数传入的ego_fut_trajs
             ego_fut_masks,
             ego_fut_cmd,
             gt_attr_labels,
             gt_bboxes_ignore=None,
             map_gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        map_gt_vecs_list = copy.deepcopy(map_gt_bboxes_list)

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        all_traj_preds = preds_dicts['all_traj_preds']
        all_traj_cls_scores = preds_dicts['all_traj_cls_scores']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        map_all_cls_scores = preds_dicts['map_all_cls_scores']
        map_all_bbox_preds = preds_dicts['map_all_bbox_preds']
        map_all_pts_preds = preds_dicts['map_all_pts_preds']
        map_enc_cls_scores = preds_dicts['map_enc_cls_scores']
        map_enc_bbox_preds = preds_dicts['map_enc_bbox_preds']
        map_enc_pts_preds = preds_dicts['map_enc_pts_preds']
        ego_fut_preds = preds_dicts['ego_fut_preds']
        distribution_pred = preds_dicts['loss_vae_gen']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_attr_labels_list = [gt_attr_labels for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox, loss_traj, loss_traj_cls = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_traj_preds,
            all_traj_cls_scores, all_gt_bboxes_list, all_gt_labels_list,
            all_gt_attr_labels_list, all_gt_bboxes_ignore_list)

        num_dec_layers = len(map_all_cls_scores)
        device = map_gt_labels_list[0].device

        map_gt_bboxes_list = [
            map_gt_bboxes.bbox.to(device) for map_gt_bboxes in map_gt_vecs_list]
        map_gt_pts_list = [
            map_gt_bboxes.fixed_num_sampled_points.to(device) for map_gt_bboxes in map_gt_vecs_list]
        if self.map_gt_shift_pts_pattern == 'v0':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v1':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v1.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v2':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v2.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v3':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v3.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v4':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v4.to(device) for gt_bboxes in map_gt_vecs_list]
        else:
            raise NotImplementedError
        map_all_gt_bboxes_list = [map_gt_bboxes_list for _ in range(num_dec_layers)]
        map_all_gt_labels_list = [map_gt_labels_list for _ in range(num_dec_layers)]
        map_all_gt_pts_list = [map_gt_pts_list for _ in range(num_dec_layers)]
        map_all_gt_shifts_pts_list = [map_gt_shifts_pts_list for _ in range(num_dec_layers)]
        map_all_gt_bboxes_ignore_list = [
            map_gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        map_losses_cls, map_losses_bbox, map_losses_iou, \
        map_losses_pts, map_losses_dir = multi_apply(
            self.map_loss_single, map_all_cls_scores, map_all_bbox_preds,
            map_all_pts_preds, map_all_gt_bboxes_list, map_all_gt_labels_list,
            map_all_gt_shifts_pts_list, map_all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_traj'] = loss_traj[-1]
        loss_dict['loss_traj_cls'] = loss_traj_cls[-1]
        # loss from the last decoder layer
        loss_dict['loss_map_cls'] = map_losses_cls[-1]
        loss_dict['loss_map_bbox'] = map_losses_bbox[-1]
        loss_dict['loss_map_iou'] = map_losses_iou[-1]
        loss_dict['loss_map_pts'] = map_losses_pts[-1]
        loss_dict['loss_map_dir'] = map_losses_dir[-1]

        # Planning Loss
        ego_fut_gt = ego_fut_gt.squeeze(1)
        ego_fut_masks = ego_fut_masks.squeeze(1).squeeze(1)
        ego_fut_cmd = ego_fut_cmd.squeeze(1).squeeze(1)

        batch, num_agent = all_traj_preds[-1].shape[:2]
        agent_fut_preds = all_traj_preds[-1].view(batch, num_agent, self.fut_mode, self.fut_ts, 2)
        agent_fut_cls_preds = all_traj_cls_scores[-1].view(batch, num_agent, self.fut_mode)
        loss_plan_input = [ego_fut_preds, ego_fut_gt, ego_fut_masks, ego_fut_cmd,
                           map_all_pts_preds[-1], map_all_cls_scores[-1].sigmoid(),
                           all_bbox_preds[-1][..., 0:2], agent_fut_preds,
                           all_cls_scores[-1].sigmoid(), agent_fut_cls_preds.sigmoid()]

        loss_planning_dict = self.loss_planning(*loss_plan_input)
        loss_dict['loss_plan_reg'] = loss_planning_dict['loss_plan_reg']
        loss_dict['loss_plan_bound'] = loss_planning_dict['loss_plan_bound']
        loss_dict['loss_plan_col'] = loss_planning_dict['loss_plan_col']
        loss_dict['loss_plan_dir'] = loss_planning_dict['loss_plan_dir']

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        # loss from other decoder layers
        num_dec_layer = 0
        for map_loss_cls_i, map_loss_bbox_i, map_loss_iou_i, map_loss_pts_i, map_loss_dir_i in zip(
                map_losses_cls[:-1],
                map_losses_bbox[:-1],
                map_losses_iou[:-1],
                map_losses_pts[:-1],
                map_losses_dir[:-1]
        ):
            loss_dict[f'd{num_dec_layer}.loss_map_cls'] = map_loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_map_bbox'] = map_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_map_iou'] = map_loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_map_pts'] = map_loss_pts_i
            loss_dict[f'd{num_dec_layer}.loss_map_dir'] = map_loss_dir_i
            num_dec_layer += 1

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list,
                                 gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        if map_enc_cls_scores is not None:
            map_binary_labels_list = [
                torch.zeros_like(map_gt_labels_list[i])
                for i in range(len(map_all_gt_labels_list))
            ]
            # TODO bug here, but we dont care enc_loss now
            map_enc_loss_cls, map_enc_loss_bbox, map_enc_loss_iou, \
            map_enc_loss_pts, map_enc_loss_dir = \
                self.map_loss_single(
                    map_enc_cls_scores, map_enc_bbox_preds,
                    map_enc_pts_preds, map_gt_bboxes_list,
                    map_binary_labels_list, map_gt_pts_list,
                    map_gt_bboxes_ignore
                )
            loss_dict['enc_loss_map_cls'] = map_enc_loss_cls
            loss_dict['enc_loss_map_bbox'] = map_enc_loss_bbox
            loss_dict['enc_loss_map_iou'] = map_enc_loss_iou
            loss_dict['enc_loss_map_pts'] = map_enc_loss_pts
            loss_dict['enc_loss_map_dir'] = map_enc_loss_dir

        loss_dict['loss_vae_gen'] = self.loss_vae_gen(distribution_pred)

        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        det_preds_dicts = self.bbox_coder.decode(preds_dicts)
        # map_bboxes: xmin, ymin, xmax, ymax
        map_preds_dicts = self.map_bbox_coder.decode(preds_dicts)

        num_samples = len(det_preds_dicts)
        assert len(det_preds_dicts) == len(map_preds_dicts), \
            'len(preds_dict) should be equal to len(map_preds_dicts)'
        ret_list = []
        for i in range(num_samples):
            preds = det_preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']
            trajs = preds['trajs']

            map_preds = map_preds_dicts[i]
            map_bboxes = map_preds['map_bboxes']
            map_scores = map_preds['map_scores']
            map_labels = map_preds['map_labels']
            map_pts = map_preds['map_pts']

            ret_list.append([bboxes, scores, labels, trajs, map_bboxes,
                             map_scores, map_labels, map_pts])

        return ret_list

    def select_and_pad_pred_map(
            self,
            motion_pos,
            map_query,
            map_score,
            map_pos,
            map_thresh=0.5,
            dis_thresh=None,
            pe_normalization=True,
            use_fix_pad=False
    ):
        """select_and_pad_pred_map.
        Args:
            motion_pos: [B, A, 2]
            map_query: [B, P, D].
            map_score: [B, P, 3].
            map_pos: [B, P, pts, 2].
            map_thresh: map confidence threshold for filtering low-confidence preds
            dis_thresh: distance threshold for masking far maps for each agent in cross-attn
            use_fix_pad: always pad one lane instance for each batch
        Returns:
            selected_map_query: [B*A, P1(+1), D], P1 is the max inst num after filter and pad.
            selected_map_pos: [B*A, P1(+1), 2]
            selected_padding_mask: [B*A, P1(+1)]
        """

        if dis_thresh is None:
            raise NotImplementedError('Not implement yet')

        # use the most close pts pos in each map inst as the inst's pos
        batch, num_map = map_pos.shape[:2]
        map_dis = torch.sqrt(map_pos[..., 0] ** 2 + map_pos[..., 1] ** 2)
        min_map_pos_idx = map_dis.argmin(dim=-1).flatten()  # [B*P]
        min_map_pos = map_pos.flatten(0, 1)  # [B*P, pts, 2]
        min_map_pos = min_map_pos[range(min_map_pos.shape[0]), min_map_pos_idx]  # [B*P, 2]
        min_map_pos = min_map_pos.view(batch, num_map, 2)  # [B, P, 2]

        # select & pad map vectors for different batch using map_thresh
        map_score = map_score.sigmoid()
        map_max_score = map_score.max(dim=-1)[0]
        map_idx = map_max_score > map_thresh
        batch_max_pnum = 0
        for i in range(map_score.shape[0]):
            pnum = map_idx[i].sum()
            if pnum > batch_max_pnum:
                batch_max_pnum = pnum

        selected_map_query, selected_map_pos, selected_padding_mask = [], [], []
        for i in range(map_score.shape[0]):
            dim = map_query.shape[-1]
            valid_pnum = map_idx[i].sum()
            valid_map_query = map_query[i, map_idx[i]]
            valid_map_pos = min_map_pos[i, map_idx[i]]
            pad_pnum = batch_max_pnum - valid_pnum
            padding_mask = torch.tensor([False], device=map_score.device).repeat(batch_max_pnum)
            if pad_pnum != 0:
                valid_map_query = torch.cat([valid_map_query, torch.zeros((pad_pnum, dim), device=map_score.device)],
                                            dim=0)
                valid_map_pos = torch.cat([valid_map_pos, torch.zeros((pad_pnum, 2), device=map_score.device)], dim=0)
                padding_mask[valid_pnum:] = True
            selected_map_query.append(valid_map_query)
            selected_map_pos.append(valid_map_pos)
            selected_padding_mask.append(padding_mask)

        selected_map_query = torch.stack(selected_map_query, dim=0)
        selected_map_pos = torch.stack(selected_map_pos, dim=0)
        selected_padding_mask = torch.stack(selected_padding_mask, dim=0)

        # generate different pe for map vectors for each agent
        num_agent = motion_pos.shape[1]
        selected_map_query = selected_map_query.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, D]
        selected_map_pos = selected_map_pos.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, 2]
        selected_padding_mask = selected_padding_mask.unsqueeze(1).repeat(1, num_agent, 1)  # [B, A, max_P]
        # move lane to per-car coords system
        selected_map_dist = selected_map_pos - motion_pos[:, :, None, :]  # [B, A, max_P, 2]
        if pe_normalization:
            selected_map_pos = selected_map_pos - motion_pos[:, :, None, :]  # [B, A, max_P, 2]

        # filter far map inst for each agent
        map_dis = torch.sqrt(selected_map_dist[..., 0] ** 2 + selected_map_dist[..., 1] ** 2)
        valid_map_inst = (map_dis <= dis_thresh)  # [B, A, max_P]
        invalid_map_inst = (valid_map_inst == False)
        selected_padding_mask = selected_padding_mask + invalid_map_inst

        selected_map_query = selected_map_query.flatten(0, 1)
        selected_map_pos = selected_map_pos.flatten(0, 1)
        selected_padding_mask = selected_padding_mask.flatten(0, 1)

        num_batch = selected_padding_mask.shape[0]
        feat_dim = selected_map_query.shape[-1]
        if use_fix_pad:
            pad_map_query = torch.zeros((num_batch, 1, feat_dim), device=selected_map_query.device)
            pad_map_pos = torch.ones((num_batch, 1, 2), device=selected_map_pos.device)
            pad_lane_mask = torch.tensor([False], device=selected_padding_mask.device).unsqueeze(0).repeat(num_batch, 1)
            selected_map_query = torch.cat([selected_map_query, pad_map_query], dim=1)
            selected_map_pos = torch.cat([selected_map_pos, pad_map_pos], dim=1)
            selected_padding_mask = torch.cat([selected_padding_mask, pad_lane_mask], dim=1)

        return selected_map_query, selected_map_pos, selected_padding_mask

    def select_and_pad_query(
            self,
            query,
            query_pos,
            query_score,
            score_thresh=0.5,
            use_fix_pad=True
    ):
        """select_and_pad_query.
        Args:
            query: [B, Q, D].
            query_pos: [B, Q, 2]
            query_score: [B, Q, C].
            score_thresh: confidence threshold for filtering low-confidence query
            use_fix_pad: always pad one query instance for each batch
        Returns:
            selected_query: [B, Q', D]
            selected_query_pos: [B, Q', 2]
            selected_padding_mask: [B, Q']
        """

        # select & pad query for different batch using score_thresh
        query_score = query_score.sigmoid()
        query_score = query_score.max(dim=-1)[0]
        query_idx = query_score > score_thresh
        batch_max_qnum = 0
        for i in range(query_score.shape[0]):
            qnum = query_idx[i].sum()
            if qnum > batch_max_qnum:
                batch_max_qnum = qnum

        selected_query, selected_query_pos, selected_padding_mask = [], [], []
        for i in range(query_score.shape[0]):
            dim = query.shape[-1]
            valid_qnum = query_idx[i].sum()
            valid_query = query[i, query_idx[i]]
            valid_query_pos = query_pos[i, query_idx[i]]
            pad_qnum = batch_max_qnum - valid_qnum
            padding_mask = torch.tensor([False], device=query_score.device).repeat(batch_max_qnum)
            if pad_qnum != 0:
                valid_query = torch.cat([valid_query, torch.zeros((pad_qnum, dim), device=query_score.device)], dim=0)
                valid_query_pos = torch.cat([valid_query_pos, torch.zeros((pad_qnum, 2), device=query_score.device)],
                                            dim=0)
                padding_mask[valid_qnum:] = True
            selected_query.append(valid_query)
            selected_query_pos.append(valid_query_pos)
            selected_padding_mask.append(padding_mask)

        selected_query = torch.stack(selected_query, dim=0)
        selected_query_pos = torch.stack(selected_query_pos, dim=0)
        selected_padding_mask = torch.stack(selected_padding_mask, dim=0)

        num_batch = selected_padding_mask.shape[0]
        feat_dim = selected_query.shape[-1]
        if use_fix_pad:
            pad_query = torch.zeros((num_batch, 1, feat_dim), device=selected_query.device)
            pad_query_pos = torch.ones((num_batch, 1, 2), device=selected_query_pos.device)
            pad_mask = torch.tensor([False], device=selected_padding_mask.device).unsqueeze(0).repeat(num_batch, 1)
            selected_query = torch.cat([selected_query, pad_query], dim=1)
            selected_query_pos = torch.cat([selected_query_pos, pad_query_pos], dim=1)
            selected_padding_mask = torch.cat([selected_padding_mask, pad_mask], dim=1)

        return selected_query, selected_query_pos, selected_padding_mask

    def distribution_forward(self, present_features, future_distribution_inputs=None, noise=None):
        """distribution_forward.
        Args:
            present_features:: output features of transformer model.
            future_distribution_inputs: the agent and ego gt trajectory in the future.
            noise: gaussian noise.
        Returns:
            sample: sample taken from present/future distribution
            present_distribution_mu: mean value of present gaussian distribution with shape (B, S, D)
            present_distribution_log_sigma: variance of present gaussian distribution with shape (B, S, D)
            future_distribution_mu: mean value of future gaussian distribution with shape (B, S, D)
            future_distribution_log_sigma: variance of future gaussian distribution with shape (B, S, D)
        """

        b = present_features.shape[0]
        c = present_features.shape[1]
        # 根据经过一系列计算得到的特征，计算当前分布
        present_mu, present_log_sigma = self.present_distribution(present_features)

        # 计算未来分布
        future_mu, future_log_sigma = None, None
        if future_distribution_inputs is not None:
            # Concatenate future labels to z_t
            # future_features = future_distribution_inputs[:, 1:].contiguous().view(b, 1, -1, h, w)
            # future_features = torch.cat([present_features, future_distribution_inputs], dim=2)
            future_features = torch.cat([  # [1, 1801, 524]
                present_features,  # [1, 1801, 512]
                future_distribution_inputs  # [1, 1801, 12]
            ], dim=2)
            future_mu, future_log_sigma = self.future_distribution(future_features)

        if noise is None:
            if self.training:
                noise = torch.randn_like(present_mu)
            else:
                noise = torch.randn_like(present_mu)
        # print('################################')
        # print('noise: ', noise)
        # print('################################')
        if self.training:       # 训练阶段
            mu = future_mu
            sigma = torch.exp(future_log_sigma)
        else:
            mu = present_mu     # 测试阶段
            sigma = torch.exp(present_log_sigma)
        sample = mu + sigma * noise

        # Spatially broadcast sample to the dimensions of present_features
        sample = sample.permute(0, 2, 1).expand(b, self.latent_dim, c)

        output_distribution = {
            'present_mu': present_mu,
            'present_log_sigma': present_log_sigma,
            'future_mu': future_mu,
            'future_log_sigma': future_log_sigma,
        }

        return sample, output_distribution

    def get_future_labels(self, gt_labels_3d, gt_attr_labels, ego_fut_trajs, device):

        """get_future_label.
        Args:
            gt_labels_3d: agent future 3d labels
            gt_attr_labels: agent future 3d labels
            ego_fut_trajs: ego future trajectory.
            device: gpu device id
        Returns:
            gt_trajs: [B, A, T, 2]
        """

        agent_dim = 300
        veh_list = [0, 1, 3, 4]
        mapped_class_names = [
            'car', 'truck', 'construction_vehicle', 'bus',
            'trailer', 'barrier', 'motorcycle', 'bicycle',
            'pedestrian', 'traffic_cone'
        ]
        ignore_list = ['construction_vehicle', 'barrier',
                       'traffic_cone', 'motorcycle', 'bicycle']

        batch_size = len(gt_labels_3d)

        # gt_label = gt_labels_3d[0]
        # gt_attr_label = gt_attr_labels[0]

        gt_fut_trajs_bz_list = []

        for bz in range(batch_size):                # 对每个批次的数据进行处理
            gt_fut_trajs_list = []
            gt_label = gt_labels_3d[bz]
            gt_attr_label = gt_attr_labels[bz]
            for i in range(gt_label.shape[0]):      # 对每个批次中的每个agent进行处理
                # 将车辆类型（0,1,3,4）统一映射为car(0)
                gt_label[i] = 0 if gt_label[i] in veh_list else gt_label[i]
                box_name = mapped_class_names[gt_label[i]]
                # 忽略掉一些车辆类型
                if box_name in ignore_list:
                    continue
                # 提取masks信息来确定哪些时间步有效
                gt_fut_masks = gt_attr_label[i][self.fut_ts * 2:self.fut_ts * 3]
                num_valid_ts = sum(gt_fut_masks == 1)
                # 提取轨迹数据并重塑为(T,2)格式
                gt_fut_traj = gt_attr_label[i][:self.fut_ts * 2].reshape(-1, 2)
                # 只保留有效的时间步
                gt_fut_traj = gt_fut_traj[:num_valid_ts]
                # 处理不完整的轨迹（确保所有轨迹都有相同的长度（self.fut_ts），通过零填充短轨迹）
                if gt_fut_traj.shape[0] == 0:
                    gt_fut_traj = torch.zeros([self.fut_ts - gt_fut_traj.shape[0], 2], device=device)
                if gt_fut_traj.shape[0] < self.fut_ts:
                    gt_fut_traj = torch.cat(
                        (gt_fut_traj, torch.zeros([self.fut_ts - gt_fut_traj.shape[0], 2], device=device)), 0)
                gt_fut_trajs_list.append(gt_fut_traj)

            # 处理完一个批次中的所有agent后，确保数据形状统一
            if len(gt_fut_trajs_list) != 0 & len(gt_fut_trajs_list) < agent_dim:
                # 如果有agent但少于agent_dim，填充到agent_dim
                gt_fut_trajs = torch.cat(
                    (torch.stack(gt_fut_trajs_list),
                     torch.zeros([agent_dim - len(gt_fut_trajs_list), self.fut_ts, 2], device=device)), 0)
            else:
                # 如果没有agent或者超过agent_dim（不太可能），创建全零tensor
                gt_fut_trajs = torch.zeros([agent_dim, self.fut_ts, 2], device=device)

            gt_fut_trajs_bz_list.append(gt_fut_trajs)

        # 将agent和ego的轨迹数据合并
        if len(gt_fut_trajs_bz_list) != 0:
            # 将agent轨迹和ego轨迹连接起来，agent轨迹重复6次（对应6种模式）
            gt_trajs = torch.cat((torch.stack(gt_fut_trajs_bz_list).repeat(1, 6, 1, 1), ego_fut_trajs), dim=1)
        else:
            gt_trajs = ego_fut_trajs
        # future_states =  gt_trajs.reshape(batch_size, gt_trajs.shape[1], -1)

        # [bz, a, t, 2]
        return gt_trajs.reshape(batch_size, gt_trajs.shape[1], -1)

    def future_states_predict(self, batch_size, sample, hidden_states, current_states):
        """get_future_label.
              Args:
                  batch_size: batch size
                  sample: sample taken from present/future distribution
                  hidden_states: hidden states input of autoregressive model.
                  current_states: current states input of autoregressive model.
              Returns:
                  states_hs: the final features combined with the generative features and current features
                  future_states_hs: the generative features predicted by generate model(VAE)
              """

        future_prediction_input = sample.unsqueeze(0).expand(self.fut_ts, -1, -1, -1)
        future_prediction_input = future_prediction_input.reshape(self.fut_ts, -1, self.latent_dim)

        hidden_state = hidden_states.reshape(self.layer_dim, -1, int(self.embed_dims / 2))
        future_states = self.predict_model(future_prediction_input, hidden_state)

        current_states_hs = current_states.unsqueeze(0).repeat(6, 1, 1, 1)
        future_states_hs = future_states.reshape(self.fut_ts, batch_size, -1, future_states.shape[2])

        if self.with_cur:
            states_hs = torch.cat((current_states_hs, future_states_hs), dim=-1)
        else:
            states_hs = future_states_hs

        return states_hs, future_states_hs






