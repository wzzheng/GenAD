import copy
import numpy as np
import os
from os import path as osp
import torch
import random
import json, pickle
import tempfile
import cv2
from pyquaternion import Quaternion
from mmcv.datasets import DATASETS
from mmcv.utils import save_tensor
from mmcv.parallel import DataContainer as DC
from mmcv.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from mmcv.fileio.io import load, dump
from mmcv.utils import track_iter_progress, mkdir_or_exist
from mmcv.datasets.pipelines import to_tensor
from .custom_3d import Custom3DDataset
from .pipelines import Compose
from .nuscenes_styled_eval_utils import DetectionMetrics, EvalBoxes, DetectionBox,center_distance,accumulate,DetectionMetricDataList,calc_ap, calc_tp, quaternion_yaw
from prettytable import PrettyTable



@DATASETS.register_module()
class B2D_E2E_Dataset(Custom3DDataset):
    def __init__(self, queue_length=4, bev_size=(200, 200),overlap_test=False,with_velocity=True,sample_interval=5,name_mapping= None,eval_cfg = None, map_root =None,map_file=None,past_frames=4, future_frames=4,predict_frames=12,planning_frames=6,patch_size = [102.4, 102.4],point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0] ,occ_receptive_field=3,occ_n_future=6,occ_filter_invalid_sample=False,occ_filter_by_valid_flag=False,eval_mod=None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.bev_size = (200, 200)
        self.overlap_test = overlap_test
        self.with_velocity = with_velocity
        self.NameMapping  = name_mapping
        self.eval_cfg  = eval_cfg
        self.sample_interval = sample_interval
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.predict_frames = predict_frames
        self.planning_frames = planning_frames
        self.map_root = map_root
        self.map_file = map_file
        self.point_cloud_range = np.array(point_cloud_range)
        self.patch_size = patch_size
        self.occ_receptive_field = occ_receptive_field  # past + current
        self.occ_n_future = occ_n_future  # future only
        self.occ_filter_invalid_sample = occ_filter_invalid_sample
        self.occ_filter_by_valid_flag = occ_filter_by_valid_flag
        self.occ_only_total_frames = 7  # NOTE: hardcode, not influenced by planning   
        self.eval_mod = eval_mod     
        self.map_element_class = {'Broken':0, 'Solid':1, 'SolidSolid':2,'Center':3,'TrafficLight':4,'StopSign':5}
        with open(self.map_file,'rb') as f: 
            self.map_infos = pickle.load(f)

    def invert_pose(self, pose):
        inv_pose = np.eye(4)
        inv_pose[:3, :3] = np.transpose(pose[:3, :3])
        inv_pose[:3, -1] = - inv_pose[:3, :3] @ pose[:3, -1]
        return inv_pose

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index-self.queue_length*self.sample_interval, index,self.sample_interval))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            if self.filter_empty_gt and \
                    (example is None or ~(example['gt_labels_3d']._data != -1).any()):
                return None
            queue.append(example)
        return self.union2one(queue)
    
    def union2one(self, queue):
        imgs_list = [each['img'].data for each in queue]
        gt_labels_3d_list = [each['gt_labels_3d'].data for each in queue]
        gt_sdc_label_list = [each['gt_sdc_label'].data for each in queue]
        gt_inds_list = [to_tensor(each['gt_inds']) for each in queue]
        gt_bboxes_3d_list = [each['gt_bboxes_3d'].data for each in queue]
        gt_past_traj_list = [to_tensor(each['gt_past_traj']) for each in queue]
        gt_past_traj_mask_list = [ to_tensor(each['gt_past_traj_mask']) for each in queue]
        gt_sdc_bbox_list = [each['gt_sdc_bbox'].data for each in queue]
        l2g_r_mat_list = [to_tensor(each['l2g_r_mat']) for each in queue]
        l2g_t_list = [to_tensor(each['l2g_t']) for each in queue]
        timestamp_list = [to_tensor(each['timestamp']) for each in queue]
        gt_fut_traj = to_tensor(queue[-1]['gt_fut_traj'])
        gt_fut_traj_mask = to_tensor(queue[-1]['gt_fut_traj_mask'])
        if 'gt_future_boxes' in queue[-1]:
            gt_future_boxes_list = queue[-1]['gt_future_boxes']
        else:
            gt_future_boxes_list = None
        if 'gt_future_labels' in queue[-1]:    
            gt_future_labels_list = [to_tensor(each) for each in queue[-1]['gt_future_labels']]
        else:
            gt_future_labels_list = None

        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if metas_map[i]['folder'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['folder']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
        queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        queue['gt_labels_3d'] = DC(gt_labels_3d_list)
        queue['gt_sdc_label'] = DC(gt_sdc_label_list)
        queue['gt_inds'] = DC(gt_inds_list)
        queue['gt_bboxes_3d'] = DC(gt_bboxes_3d_list, cpu_only=True)
        queue['gt_sdc_bbox'] = DC(gt_sdc_bbox_list, cpu_only=True)
        queue['l2g_r_mat'] = DC(l2g_r_mat_list)
        queue['l2g_t'] = DC(l2g_t_list)
        queue['timestamp'] = DC(timestamp_list)
        queue['gt_fut_traj'] = DC(gt_fut_traj)
        queue['gt_fut_traj_mask'] = DC(gt_fut_traj_mask)
        queue['gt_past_traj'] = DC(gt_past_traj_list)
        queue['gt_past_traj_mask'] = DC(gt_past_traj_mask_list)
        if gt_future_boxes_list is not None:
            queue['gt_future_boxes'] = DC(gt_future_boxes_list, cpu_only=True)
        if gt_future_labels_list is not None:
            queue['gt_future_labels'] = DC(gt_future_labels_list)

        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]

        for i in range(len(info['gt_names'])):
            if info['gt_names'][i] in self.NameMapping.keys():
                info['gt_names'][i] = self.NameMapping[info['gt_names'][i]]


        gt_masks,gt_labels,gt_bboxes = self.get_map_info(index)


        input_dict = dict(
            folder=info['folder'],
            scene_token=info['folder'],
            frame_idx=info['frame_idx'],
            ego_yaw=np.nan_to_num(info['ego_yaw'],nan=np.pi/2),
            ego_translation=info['ego_translation'],
            sensors=info['sensors'],
            world2lidar=info['sensors']['LIDAR_TOP']['world2lidar'],
            gt_ids=info['gt_ids'],
            gt_boxes=info['gt_boxes'],
            gt_names=info['gt_names'],
            ego_vel = info['ego_vel'],
            ego_accel = info['ego_accel'],
            ego_rotation_rate = info['ego_rotation_rate'],
            npc2world = info['npc2world'],
            gt_lane_labels=gt_labels,
            gt_lane_bboxes=gt_bboxes,
            gt_lane_masks=gt_masks,
            timestamp=info['frame_idx']/10

        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            lidar2ego = info['sensors']['LIDAR_TOP']['lidar2ego']
            for sensor_type, cam_info in info['sensors'].items():
                if not 'CAM' in sensor_type:
                    continue
                image_paths.append(osp.join(self.data_root,cam_info['data_path']))
                # obtain lidar to image transformation matrix
                cam2ego = cam_info['cam2ego']
                intrinsic = cam_info['intrinsic']
                intrinsic_pad = np.eye(4)
                intrinsic_pad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2cam = self.invert_pose(cam2ego) @ lidar2ego
                lidar2img = intrinsic_pad @ lidar2cam
                lidar2img_rts.append(lidar2img)
                cam_intrinsics.append(intrinsic_pad)
                lidar2cam_rts.append(lidar2cam)
            ego2world = np.eye(4)
            ego2world[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=input_dict['ego_yaw']).rotation_matrix
            ego2world[0:3,3] = input_dict['ego_translation']
            lidar2global = ego2world @ lidar2ego
            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                    l2g_r_mat=lidar2global[0:3,0:3],
                    l2g_t=lidar2global[0:3,3]

                ))

        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos
        yaw = input_dict['ego_yaw']
        rotation = list(Quaternion(axis=[0, 0, 1], radians=yaw))
        if yaw < 0:
            yaw += 2*np.pi
        yaw_in_degree = yaw / np.pi * 180 
        
        can_bus = np.zeros(18)
        can_bus[:3] = input_dict['ego_translation']
        can_bus[3:7] = rotation
        can_bus[7:10] = input_dict['ego_vel']
        can_bus[10:13] = input_dict['ego_accel']
        can_bus[13:16] = input_dict['ego_rotation_rate']
        can_bus[16] = yaw
        can_bus[17] = yaw_in_degree
        input_dict['can_bus'] = can_bus
        all_frames = []
        for adj_idx in range(index-self.occ_receptive_field+1,index+self.occ_n_future+1):
            if adj_idx<0 or adj_idx>=len(self.data_infos):
                all_frames.append(-1)
            elif self.data_infos[adj_idx]['folder'] != self.data_infos[index]['folder']:
                all_frames.append(-1)
            else: 
                all_frames.append(adj_idx)
            
        future_frames = all_frames[self.occ_receptive_field-1:]
        input_dict['occ_has_invalid_frame'] = (-1 in all_frames[:self.occ_only_total_frames])
        input_dict['occ_img_is_valid'] = np.array(all_frames) >= 0
        occ_future_ann_infos = []
        for future_frame in future_frames:
            if future_frame >= 0:
                occ_future_ann_infos.append(
                    self.get_ann_boxes_only(future_frame),
                )
            else:
                occ_future_ann_infos.append(None)
        input_dict['occ_future_ann_infos'] = occ_future_ann_infos

        input_dict.update(self.occ_get_transforms(future_frames))
        sdc_planning, sdc_planning_mask = self.get_ego_future_xy(index,self.sample_interval,self.planning_frames)
        input_dict['sdc_planning'] = sdc_planning
        input_dict['sdc_planning_mask'] = sdc_planning_mask
        command = info['command_near']
        if command < 0:
            command = 4
        command -= 1
        input_dict['command'] = command

        return input_dict


    def get_map_info(self, index):

        gt_masks = []
        gt_labels = []
        gt_bboxes = []

        ann_info = self.data_infos[index]
        town_name = ann_info['town_name']
        map_info = self.map_infos[town_name]
        lane_points = map_info['lane_points']
        lane_sample_points = map_info['lane_sample_points']
        lane_types = map_info['lane_types']
        trigger_volumes_points = map_info['trigger_volumes_points']
        trigger_volumes_sample_points = map_info['trigger_volumes_sample_points']
        trigger_volumes_types = map_info['trigger_volumes_types']
        world2lidar = np.array(ann_info['sensors']['LIDAR_TOP']['world2lidar'])
        ego_xy = np.linalg.inv(world2lidar)[0:2,3]

        #1st search
        max_distance = 100
        chosed_idx = []
        for idx in range(len(lane_sample_points)):
            single_sample_points = lane_sample_points[idx]
            distance = np.linalg.norm((single_sample_points[:,0:2]-ego_xy),axis=-1)
            if np.min(distance) < max_distance:
                chosed_idx.append(idx)

        for idx in chosed_idx:
            if not lane_types[idx] in self.map_element_class.keys():
                continue
            points = lane_points[idx]
            points = np.concatenate([points,np.ones((points.shape[0],1))],axis=-1)
            points_in_ego = (world2lidar @ points.T).T
            #print(points_in_ego)
            mask = (points_in_ego[:,0]>self.point_cloud_range[0]) & (points_in_ego[:,0]<self.point_cloud_range[3]) & (points_in_ego[:,1]>self.point_cloud_range[1]) & (points_in_ego[:,1]<self.point_cloud_range[4])
            points_in_ego_range = points_in_ego[mask,0:2]
            if len(points_in_ego_range) > 1:
                gt_mask = np.zeros(self.bev_size,dtype=np.uint8)
                normalized_points = np.zeros_like(points_in_ego_range)
                normalized_points[:,0] = (points_in_ego_range[:,0] + self.patch_size[0]/2)*(self.bev_size[0]/self.patch_size[0])
                normalized_points[:,1] = (points_in_ego_range[:,1] + self.patch_size[1]/2)*(self.bev_size[1]/self.patch_size[1])
                cv2.polylines(gt_mask, [normalized_points.astype(np.int32)], False, color=1, thickness=2)
                gt_label =  self.map_element_class[lane_types[idx]]
                gt_masks.append(gt_mask)
                gt_labels.append(gt_label)
                ys, xs = np.where(gt_mask==1)
                gt_bboxes.append([min(xs), min(ys), max(xs), max(ys)]) 

        for idx in range(len(trigger_volumes_points)):
            if not trigger_volumes_types[idx] in self.map_element_class.keys():
                continue
            points = trigger_volumes_points[idx]
            points = np.concatenate([points,np.ones((points.shape[0],1))],axis=-1)
            points_in_ego = (world2lidar @ points.T).T
            mask = (points_in_ego[:,0]>self.point_cloud_range[0]) & (points_in_ego[:,0]<self.point_cloud_range[3]) & (points_in_ego[:,1]>self.point_cloud_range[1]) & (points_in_ego[:,1]<self.point_cloud_range[4])
            points_in_ego_range = points_in_ego[mask,0:2]
            if mask.all():
                gt_mask = np.zeros(self.bev_size,dtype=np.uint8)
                normalized_points = np.zeros_like(points_in_ego_range)
                normalized_points[:,0] = (points_in_ego_range[:,0] + self.patch_size[0]/2)*(self.bev_size[0]/self.patch_size[0])
                normalized_points[:,1] = (points_in_ego_range[:,1] + self.patch_size[1]/2)*(self.bev_size[1]/self.patch_size[1])
                cv2.fillConvexPoly(gt_mask, normalized_points.astype(np.int32), color=1)
                gt_label = self.map_element_class[trigger_volumes_types[idx]]
                gt_masks.append(gt_mask)
                gt_labels.append(gt_label)
                ys, xs = np.where(gt_mask==1)
                gt_bboxes.append([min(xs), min(ys), max(xs), max(ys)]) 

        if len(gt_masks) == 0:
            gt_masks.append(np.zeros(self.bev_size,dtype=np.uint8))
            gt_labels.append(-1)
            gt_bboxes.append([0,0,0,0])

        gt_masks = np.stack(gt_masks)
        gt_labels = np.array(gt_labels)
        gt_bboxes = np.array(gt_bboxes)

        return gt_masks,gt_labels,gt_bboxes


    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points

        for i in range(len(info['gt_names'])):
            if info['gt_names'][i] in self.NameMapping.keys():
                info['gt_names'][i] = self.NameMapping[info['gt_names'][i]]
        mask = (info['num_points'] >= -1)
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_inds = info['gt_ids']
        gt_labels_3d = []

        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)
        if not self.with_velocity:
            gt_bboxes_3d = gt_bboxes_3d[:,0:7]
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        
        ego_future_track, ego_future_mask = self.get_ego_future_xy(index,self.sample_interval,self.predict_frames)
        past_track, past_mask = self.get_past_or_future_xy(index,self.sample_interval,self.past_frames,past_or_future='past',local_xy=True)
        predict_track, predict_mask = self.get_past_or_future_xy(index,self.sample_interval,self.predict_frames,past_or_future='future',local_xy=False)
        mask = (past_mask.sum((1,2))>0).astype(np.int)
        future_track = predict_track[:,0:self.future_frames,:]*mask[:,None,None]
        future_mask = predict_mask[:,0:self.future_frames,:]*mask[:,None,None]
        full_past_track = np.concatenate([past_track,future_track],axis=1)
        full_past_mask = np.concatenate([past_mask,future_mask],axis=1)
        gt_sdc_bbox, gt_sdc_label =self.generate_sdc_info(index)
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            gt_inds=gt_inds,
            gt_fut_traj=predict_track,
            gt_fut_traj_mask=predict_mask,
            gt_past_traj=full_past_track,
            gt_past_traj_mask=full_past_mask,
            gt_sdc_bbox=gt_sdc_bbox,
            gt_sdc_label=gt_sdc_label,
            gt_sdc_fut_traj=ego_future_track[:,:,0:2],
            gt_sdc_fut_traj_mask=ego_future_mask,
            )
        return anns_results

    def get_ann_boxes_only(self, index):

        info = self.data_infos[index]
        for i in range(len(info['gt_names'])):
            if info['gt_names'][i] in self.NameMapping.keys():
                info['gt_names'][i] = self.NameMapping[info['gt_names'][i]]
        gt_bboxes_3d = info['gt_boxes']
        gt_names_3d = info['gt_names']
        gt_inds = info['gt_ids']
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)
        if not self.with_velocity:
            gt_bboxes_3d = gt_bboxes_3d[:,0:7]
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        boxes_annos = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_inds=gt_inds,
            )
        return boxes_annos

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
        
    def generate_sdc_info(self,idx):

        info = self.data_infos[idx]
        ego_size = info['ego_size']
        ego_vel = info['ego_vel']
        psudo_sdc_bbox = np.array([0.0, 0.0, 0.0, ego_size[0], ego_size[1], ego_size[2], -np.pi, ego_vel[1], ego_vel[0] ])
        if not self.with_velocity:
            psudo_sdc_bbox = psudo_sdc_bbox[0:7]
        gt_bboxes_3d = np.array([psudo_sdc_bbox]).astype(np.float32)
        gt_names_3d = ['car']
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
  
        gt_labels_3d = DC(to_tensor(gt_labels_3d))
        gt_bboxes_3d = DC(gt_bboxes_3d, cpu_only=True)

        return gt_bboxes_3d, gt_labels_3d

    def get_past_or_future_xy(self,idx,sample_rate,frames,past_or_future,local_xy=False):

        assert past_or_future in ['past','future']
        if past_or_future == 'past':
            adj_idx_list = range(idx-sample_rate,idx-(frames+1)*sample_rate,-sample_rate)
        else:
            adj_idx_list = range(idx+sample_rate,idx+(frames+1)*sample_rate,sample_rate)

        cur_frame = self.data_infos[idx]
        box_ids = cur_frame['gt_ids']
        adj_track = np.zeros((len(box_ids),frames,2))
        adj_mask = np.zeros((len(box_ids),frames,2))
        world2lidar_ego_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
        for i in range(len(box_ids)):
            box_id = box_ids[i]
            cur_box2lidar = world2lidar_ego_cur @ cur_frame['npc2world'][i]
            cur_xy = cur_box2lidar[0:2,3]      
            for j in range(len(adj_idx_list)):
                adj_idx = adj_idx_list[j]
                if adj_idx <0 or adj_idx>=len(self.data_infos):
                    break
                adj_frame = self.data_infos[adj_idx]
                if adj_frame['folder'] != cur_frame ['folder']:
                    break
                if len(np.where(adj_frame['gt_ids']==box_id)[0])==0:
                    continue
                assert len(np.where(adj_frame['gt_ids']==box_id)[0]) == 1 , np.where(adj_frame['gt_ids']==box_id)[0]
                adj_idx = np.where(adj_frame['gt_ids']==box_id)[0][0]
                adj_box2lidar = world2lidar_ego_cur @ adj_frame['npc2world'][adj_idx]
                adj_xy = adj_box2lidar[0:2,3]    
                if local_xy:
                    adj_xy -= cur_xy
                adj_track[i,j,:] = adj_xy
                adj_mask[i,j,:] = 1
        return adj_track, adj_mask

    def get_ego_future_xy(self,idx,sample_rate,frames):

        adj_idx_list = range(idx+sample_rate,idx+(frames+1)*sample_rate,sample_rate)
        cur_frame = self.data_infos[idx]
        adj_track = np.zeros((1,frames,3))
        adj_mask = np.zeros((1,frames,2))
        world2lidar_ego_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
        for j in range(len(adj_idx_list)):
            adj_idx = adj_idx_list[j]
            if adj_idx <0 or adj_idx>=len(self.data_infos):
                break
            adj_frame = self.data_infos[adj_idx]
            if adj_frame['folder'] != cur_frame ['folder']:
                break
            world2lidar_ego_adj = adj_frame['sensors']['LIDAR_TOP']['world2lidar']
            adj2cur_lidar = world2lidar_ego_cur @ np.linalg.inv(world2lidar_ego_adj)
            xy = adj2cur_lidar[0:2,3]
            yaw = np.arctan2(adj2cur_lidar[1,0],adj2cur_lidar[0,0])
            yaw = -yaw -np.pi
            while yaw > np.pi:
                yaw -= np.pi*2
            while yaw < -np.pi:
                yaw += np.pi*2
            adj_track[0,j,0:2] = xy
            adj_track[0,j,2] = yaw
            adj_mask[0,j,:] = 1

        return adj_track, adj_mask

    def occ_get_transforms(self, indices, data_type=torch.float32):

        l2e_r_mats = []
        l2e_t_vecs = []
        e2g_r_mats = []
        e2g_t_vecs = []

        for index in indices:
            if index == -1:
                l2e_r_mats.append(None)
                l2e_t_vecs.append(None)
                e2g_r_mats.append(None)
                e2g_t_vecs.append(None)
            else:
                info = self.data_infos[index]
                lidar2ego = info['sensors']['LIDAR_TOP']['lidar2ego']
                l2e_r = lidar2ego[0:3,0:3]
                l2e_t = lidar2ego[0:3,3]
                ego2global = np.linalg.inv(info['world2ego'])
                e2g_r = ego2global[0:3,0:3]
                e2g_t = ego2global[0:3,3]
                l2e_r_mats.append(torch.tensor(l2e_r).to(data_type))
                l2e_t_vecs.append(torch.tensor(l2e_t).to(data_type))
                e2g_r_mats.append(torch.tensor(e2g_r).to(data_type))
                e2g_t_vecs.append(torch.tensor(e2g_t).to(data_type))
        res = {
            'occ_l2e_r_mats': l2e_r_mats,
            'occ_l2e_t_vecs': l2e_t_vecs,
            'occ_e2g_r_mats': e2g_r_mats,
            'occ_e2g_t_vecs': e2g_t_vecs,
        }

        return res

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        # NOTE:Curremtly we only support evaluation on detection and planning 

        result_files, tmp_dir = self.format_results(results['bbox_results'], jsonfile_prefix)    
        result_path = result_files
        with open(result_path) as f:
            result_data = json.load(f)
        pred_boxes = EvalBoxes.deserialize(result_data['results'], DetectionBox)
        meta = result_data['meta']

        gt_boxes = self.load_gt()

        metric_data_list = DetectionMetricDataList()
        for class_name in self.eval_cfg['class_names']:
            for dist_th in self.eval_cfg['dist_ths']:
                md = accumulate(gt_boxes, pred_boxes, class_name, center_distance, dist_th)
                metric_data_list.set(class_name, dist_th, md)
                metrics = DetectionMetrics(self.eval_cfg)

        for class_name in self.eval_cfg['class_names']:
            # Compute APs.
            for dist_th in self.eval_cfg['dist_ths']:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.eval_cfg['min_recall'], self.eval_cfg['min_precision'])
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in self.eval_cfg['tp_metrics']:
                metric_data = metric_data_list[(class_name, self.eval_cfg['dist_th_tp'])]
                tp = calc_tp(metric_data, self.eval_cfg['min_recall'], metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = meta.copy()
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        #print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print()
        print('Per-class results:')
        print('Object Class\tAP\tATE\tASE\tAOE\tAVE')
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                  % (class_name, class_aps[class_name],
                     class_tps[class_name]['trans_err'],
                     class_tps[class_name]['scale_err'],
                     class_tps[class_name]['orient_err'],
                     class_tps[class_name]['vel_err']))        

        detail = dict()
        metric_prefix = 'bbox_NuScenes'
        for name in self.eval_cfg['class_names']:
            for k, v in metrics_summary['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics_summary['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics_summary['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,self.eval_cfg['err_name_maping'][k])] = val
        detail['{}/NDS'.format(metric_prefix)] = metrics_summary['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics_summary['mean_ap']

        if 'planning_results_computed' in results.keys():
                planning_results_computed = results['planning_results_computed']
                planning_tab = PrettyTable()
                planning_tab.field_names = [
                    "metrics", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s"]
                for key in planning_results_computed.keys():
                    value = planning_results_computed[key]
                    row_value = []
                    row_value.append(key)
                    for i in range(len(value)):
                        row_value.append('%.4f' % float(value[i]))
                    planning_tab.add_row(row_value)
                print(planning_tab)


        return detail

    def load_gt(self):
        all_annotations = EvalBoxes()
        for i in range(len(self.data_infos)):
            sample_boxes = []
            sample_data = self.data_infos[i]

            gt_boxes = sample_data['gt_boxes']
            
            for j in range(gt_boxes.shape[0]):
                class_name = self.NameMapping[sample_data['gt_names'][j]]
                if not class_name in self.eval_cfg['class_range'].keys():
                    continue
                range_x, range_y = self.eval_cfg['class_range'][class_name]
                if abs(gt_boxes[j,0]) > range_x or abs(gt_boxes[j,1]) > range_y:
                    continue
                sample_boxes.append(DetectionBox(
                                                sample_token=sample_data['folder']+'_'+str(sample_data['frame_idx']),
                                                translation=gt_boxes[j,0:3],
                                                size=gt_boxes[j,3:6],
                                                rotation=list(Quaternion(axis=[0, 0, 1], radians=-gt_boxes[j,6]-np.pi/2)),
                                                velocity=gt_boxes[j,7:9],
                                                num_pts=int(sample_data['num_points'][j]),
                                                detection_name=self.NameMapping[sample_data['gt_names'][j]],
                                                detection_score=-1.0,  
                                                attribute_name=self.NameMapping[sample_data['gt_names'][j]]
                                                ))
            all_annotations.add_boxes(sample_data['folder']+'_'+str(sample_data['frame_idx']), sample_boxes)
        return all_annotations
    
    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """


        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(track_iter_progress(results)):
            #pdb.set_trace()
            annos = []
            box3d = det['boxes_3d']
            scores = det['scores_3d']
            labels = det['labels_3d']
            box_gravity_center = box3d.gravity_center
            box_dims = box3d.dims
            box_yaw = box3d.yaw.numpy()
            box_yaw = -box_yaw - np.pi / 2
            sample_token = self.data_infos[sample_id]['folder'] + '_' + str(self.data_infos[sample_id]['frame_idx'])



            for i in range(len(box3d)):
                #import pdb;pdb.set_trace()
                quat = list(Quaternion(axis=[0, 0, 1], radians=box_yaw[i]))
                velocity = [box3d.tensor[i, 7].item(),box3d.tensor[i, 8].item()]
                name = mapped_class_names[labels[i]]
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box_gravity_center[i].tolist(),
                    size=box_dims[i].tolist(),
                    rotation=quat,
                    velocity=velocity,
                    detection_name=name,
                    detection_score=scores[i].item(),
                    attribute_name=name)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        dump(nusc_submissions, res_path)
        return res_path  

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        # assert len(results) == len(self), (
        #     'The length of results is not equal to the dataset len: {} != {}'.
        #     format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

