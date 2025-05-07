import torch
import numpy as np
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.utils import ext_loader
from torch.nn.init import normal_
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER
from torchvision.transforms.functional import rotate
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

from projects.mmdet3d_plugin.GenAD.modules.decoder import CustomMSDeformableAttention
from projects.mmdet3d_plugin.GenAD.modules.temporal_self_attention import TemporalSelfAttention
from projects.mmdet3d_plugin.GenAD.modules.spatial_cross_attention import MSDeformableAttention3D


import os
import re
import torch

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])

def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MapDetectionTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default:
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(MapDetectionTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        # 该模块目的：通过多个Transformer层，模型可以逐步改进对地图元素位置的预测。
        # 每一层都接收前一层的预测结果，然后通过应用回归分支产生的偏移量来细化这些预测。

        # query：输入的查询嵌入，形状为(num_query, bs, embed_dims)，代表初始的地图元素特征
        # reference_points：初始参考点坐标，表示2D空间中的位置预测
        # reg_branches：回归分支网络，用于细化位置预测
        # key_padding_mask：在注意力机制中用于忽略某些关键元素的掩码
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):

            reference_points_input = reference_points[..., :2].unsqueeze(
                2)  # BS NUM_QUERY NUM_LEVEL 2
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                # 应用回归分支预测偏移量
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 2

                new_reference_points = torch.zeros_like(reference_points)
                # 关键步骤：将预测的偏移量添加到逆sigmoid变换后的参考点上
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                # new_reference_points[..., 2:3] = tmp[
                #     ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
                
                # 应用sigmoid确保坐标在[0,1]范围内
                new_reference_points = new_reference_points.sigmoid()
                # 更新参考点（detach停止梯度流）
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@TRANSFORMER.register_module()
# 类比UniAD中的PerceptionTransformer（定义在/UniAD/projects/mmdet3d_plugin/uniad/modules/transformer.py中）
class GenADPerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 map_decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 map_num_vec=50,
                 map_num_pts_per_vec=10,
                 **kwargs):
        super(GenADPerceptionTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        if decoder is not None:
            self.decoder = build_transformer_layer_sequence(decoder)
        else:
            self.decoder = None
        if map_decoder is not None:
            self.map_decoder = build_transformer_layer_sequence(map_decoder)
        else:
            self.map_decoder = None

        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False
        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.two_stage_num_proposals = two_stage_num_proposals
        self.rotate_center = rotate_center
        self.map_num_vec = map_num_vec
        self.map_num_pts_per_vec = map_num_pts_per_vec
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        # nn.Parameter 是一个特殊的张量（tensor）对象，用于指示这个张量是可训练参数，需要在训练中进行反向传播和梯度更新
        # 当把一个 nn.Parameter 赋值给某个 nn.Module（比如网络模型）内部的属性时，这个张量就会被自动识别为模型的参数，出现在 model.parameters() 列表中
        
        # 用于增强图像特征的level_embeds
        # 用于区分不同特征层/level（如 FPN 的 P2、P3、P4、P5 等不同分辨率输出，或者多尺度特征图）
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        # 用于增强图像特征的cams_embeds
        # 用于区分不同相机视角的特征
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.map_reference_points = nn.Linear(self.embed_dims, 2)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.map_reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_bev_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain bev features.
        """

        # mlvl_feats[0]的shape是[B, N, C, H, W]
        # B -- batch size, N -- num_cams, C -- 特征通道数, H -- 特征图高度, W -- 特征图宽度
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # bev_queries.shape--[10000,1,256]

        # 1. 处理ego运动补偿
        # 空间位移补偿，目的是让当前帧的BEV特征能够与上一帧在空间位置上对齐
        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each['can_bus'][0]                  # x轴方向的位移，单位：m
                           for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1]                  # y轴方向的位移，单位：m
                           for each in kwargs['img_metas']])
        ego_angle = np.array(                                   # ego车辆的角度，单位：°
            [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        # 获取BEV视图中每个网格的实际物理尺寸
        grid_length_y = grid_length[0]                                  # 单位：m/格
        grid_length_x = grid_length[1]                                  # 单位：m/格
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)       # 计算位移距离，单位：m
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180  # 计算位移方向角度，单位：°
        # 计算bev视图中的位移
        bev_angle = ego_angle - translation_angle
        # 计算网格偏移量
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        # 创建shift张量
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        #其实用不到shift，因为prev_bev存在时只进行了rotation操作，并未平移到bev
        #因此后续prev_bev的ref_2d也不需要shift到当前时刻

        # 2. 处理时序对齐，如果存在上一帧BEV特征,进行旋转对齐
        # 视角变化补偿，目的是补偿由于自车转向导致的视角变化（关注rotation）
        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    # 获取旋转角度
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    # 重塑特征图
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    # 旋转BEV特征
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                        center=self.rotate_center)
                    # 恢复原始维度顺序
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # 3. 融合can bus信息
        # add can bus signals
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        # 以广播的方式在bev_queries中融入了can_bus信息
        bev_queries = bev_queries + can_bus * self.use_can_bus

        # 图像特征处理
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            # cams_embeds.shape -- torch.Size([6,256]),添加cam编码
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            # level_embeds.shape -- torch.Size([4,256])，添加level编码，num_level=1
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
        # 将不同level上的在C通道上进行cat操作
        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        # 将不同（H, W）的平面flatten为（H*W）的向量后cat在一起
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        # 4. 通过encoder更新BEV特征
        # 进到/mnt/kuebiko/users/qdeng/GenAD/projects/mmdet3d_plugin/GenAD/modules/encoder.py的forward函数
        # ---encoder功能---
        # BEV encoder的作用：PE / MultiHeadAttention(通过 self-attention 机制处理 bev_queries 和 feat_flatten 之间的关系) / 特征融合(SCA + TSA) / FFN
        # 生成 ref_3d 和 ref_2d，供后续 TSA、CSA 模块中的 deformable attentiuon 采样使用；
        # 利用6个相机的内参、外参，将真实尺度下的 ref_3d 从自车坐标系（Lidar坐标系）投影到6个相机的像素坐标系内，判断哪些点会出现在哪些相机内，该信息体现在 bev_mask 内；
        # 循环进入3个相同的 BEVFormerLayer 模块，一个 BEVFormerLayer 包含 (‘self_attn’, ‘norm’, ‘cross_attn’, ‘norm’, ‘ffn’, ‘norm’)。
        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,       # key
            feat_flatten,       # value
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs
        )
        # 为了保证可视化代码的一致性，不更改key的名称和变量维度等信息
        def save_bev_features(bev_features, img_metas, bev_h, bev_w, base_path='/mnt/kuebiko/users/qdeng/GenAD/bev_features_0212'):
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
        
        # save_bev_features(bev_embed, kwargs['img_metas'], 100, 100)

        return bev_embed

    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                bev_queries,
                object_query_embed,
                map_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                map_reg_branches=None,
                map_cls_branches=None,                
                prev_bev=None,            
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        bev_embed = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        # bev_embed.shape - [1,10000,256]
        bs = mlvl_feats[0].size(0)
        # 3D detection查询处理
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)     # 3D空间
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        # map semantic segmentation查询处理
        map_query_pos, map_query = torch.split(
            map_query_embed, self.embed_dims, dim=1)
        map_query_pos = map_query_pos.unsqueeze(0).expand(bs, -1, -1)
        map_query = map_query.unsqueeze(0).expand(bs, -1, -1)
        map_reference_points = self.map_reference_points(map_query_pos)
        map_reference_points = map_reference_points.sigmoid()   # 2D地图空间
        map_init_reference_out = map_reference_points        

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        map_query = map_query.permute(1, 0, 2)
        map_query_pos = map_query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        # 3D detection decoder
        if self.decoder is not None:
            # [L, Q, B, D], [L, B, Q, D]
            # 调用DetectionTransformerDecoder的forward函数
            # decoder指向DetectionTransformerDecoder这个类，定义在projects/mmdet3d_plugin/GenAD/modules/decoder.py文件中
            inter_states, inter_references = self.decoder(
                query=query,
                key=None,
                value=bev_embed,            # bev_embed影响解码器输出
                query_pos=query_pos,
                reference_points=reference_points,
                reg_branches=reg_branches,
                cls_branches=cls_branches,
                spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                level_start_index=torch.tensor([0], device=query.device),
                **kwargs)
            inter_references_out = inter_references
        else:
            inter_states = query.unsqueeze(0)
            inter_references_out = reference_points.unsqueeze(0)

        # map decoder
        if self.map_decoder is not None:
            # [L, Q, B, D], [L, B, Q, D]
            # 调用MapDetectionTransformerDecoder
            map_inter_states, map_inter_references = self.map_decoder(
                query=map_query,
                key=None,
                value=bev_embed,            # bev_embed影响解码器输出
                query_pos=map_query_pos,
                reference_points=map_reference_points,
                reg_branches=map_reg_branches,
                cls_branches=map_cls_branches,
                spatial_shapes=torch.tensor([[bev_h, bev_w]], device=map_query.device),
                level_start_index=torch.tensor([0], device=map_query.device),
                **kwargs)
            map_inter_references_out = map_inter_references
        else:
            map_inter_states = map_query.unsqueeze(0)
            map_inter_references_out = map_reference_points.unsqueeze(0)

        return (
            bev_embed, inter_states, init_reference_out, inter_references_out,
            map_inter_states, map_init_reference_out, map_inter_references_out)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class CustomTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default: `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(CustomTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                key_padding_mask=None,
                *args,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        intermediate = []
        for lid, layer in enumerate(self.layers):
            query = layer(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                key_padding_mask=key_padding_mask,
                *args,
                **kwargs)

            if self.return_intermediate:
                intermediate.append(query)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return query