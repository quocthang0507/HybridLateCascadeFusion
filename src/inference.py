from pathlib import Path
from typing_extensions import Tuple, List, Dict, Union, Sequence
from copy import deepcopy
from PIL import Image
import torch
from torch import Tensor
import json
import numpy as np
import warnings
from bbox_utils import *
from visualization_utils import *
from visualization_3d import show_bboxes_3d
from utils import compute_fundamental_matrix

from mmdet.models.task_modules import BboxOverlaps2D
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
from mmcv.transforms import Compose as ComposeMMCV

from mmdet3d.structures import get_box_type
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
from mmdet3d.apis.inference import init_model, inference_detector
from mmengine.dataset import Compose as ComposeMMEngine, pseudo_collate
from mmdet3d.structures import xywhr2xyxyr
from mmdet3d.models.layers import box3d_multiclass_nms, nms_bev

class LateFusionStereoViewInferencer:
    """
    Late Fusion inference class for single view setup.

    Args:
        late_fusion_cfg (Union[Dict, str, Path]): Configuration dict for late fusion.
            If a file path is passed, it will load the configuration from it
        lidar_to_cam (Union[np.ndarray, Tensor, Sequence[Sequence[float]]], optional): 
            Transformation matrix from LiDAR to camera coordinates.
        cam_to_img_left (Union[np.ndarray, Tensor, Sequence[Sequence[float]]], optional): 
            Transformation matrix from camera to image coordinates.
        cam_to_img_right (Union[np.ndarray, Tensor, Sequence[Sequence[float]]], optional): 
            Transformation matrix from camera to image coordinates.
        device (Union[str, torch.device], optional): Device to run the models on. Defaults to 'cuda:0'.
    """
    
    def __init__(self,
                 late_fusion_cfg: Union[Dict, str, Path],
                 lidar_to_cam: Union[np.ndarray, Tensor, Sequence[Sequence[float]]] = None,
                 cam_to_img_left: Union[np.ndarray, Tensor, Sequence[Sequence[float]]] = None,
                 cam_to_img_right: Union[np.ndarray, Tensor, Sequence[Sequence[float]]] = None,
                 device: Union[str, torch.device] = 'cuda:0'):
        
        self.device = device
        
        if isinstance(late_fusion_cfg, dict):
            self.late_fusion_cfg = late_fusion_cfg
        else:
            with open(late_fusion_cfg, 'r') as cfg_file:
                self.late_fusion_cfg = json.load(cfg_file)

        detector2d = self.late_fusion_cfg.get('detector2d', None)
        detector3d = self.late_fusion_cfg.get('detector3d', None)
        frustum_detector = self.late_fusion_cfg.get('frustum_detector', None)

        assert detector2d is not None, 'detector2d is not defined in the late fusion config'
        assert detector3d is not None, 'detector3d is not defined in the late fusion config'

        if frustum_detector is None:
            warnings.warn('frustum_detector is not defined in the late fusion config, detection recovery will not be performed')

        self._init_detector2d(detector2d['cfg_path'], detector2d['checkpoint_path'])
        self._init_detector3d(detector3d['cfg_path'], detector3d['checkpoint_path'])
        if frustum_detector is not None:
            self._init_frustum_detector(frustum_detector['cfg_path'], frustum_detector['checkpoint_path'])
        else:
            self.frustum_detector = None
                
        self.lidar_to_cam = lidar_to_cam
        if lidar_to_cam is None:
            warnings.warn('lidar_to_cam is None, if not passed sample by sample the results will be not consistent')
            self.lidar_to_cam = torch.eye(4, dtype=torch.float32)    
        elif not isinstance(lidar_to_cam, Tensor):
            self.lidar_to_cam = torch.tensor(self.lidar_to_cam, dtype=torch.float32)
        self.lidar_to_cam = self.lidar_to_cam.to(self.device)
        
        self.cam_to_img_left = cam_to_img_left
        if cam_to_img_left is None:
            warnings.warn('cam_to_img_left is None, if not passed sample by sample the results will be not consistent')
            self.cam_to_img_left = torch.eye(4, dtype=torch.float32)[:3, :]  
        elif not isinstance(cam_to_img_left, Tensor):
            self.cam_to_img_left = torch.tensor(self.cam_to_img_left, dtype=torch.float32)[:3, :]
        self.cam_to_img_left = self.cam_to_img_left.to(self.device)
        
        self.cam_to_img_right = cam_to_img_right
        if cam_to_img_right is None:
            warnings.warn('cam_to_img_right is None, if not passed sample by sample the results will be not consistent')
            self.cam_to_img_right = torch.eye(4, dtype=torch.float32)[:3, :]  
        elif not isinstance(cam_to_img_right, Tensor):
            self.cam_to_img_right = torch.tensor(self.cam_to_img_right, dtype=torch.float32)[:3, :]
        self.cam_to_img_right = self.cam_to_img_right.to(self.device)
                
        self.num_classes = self.late_fusion_cfg.get('num_classes', 3)
        self.classes = self.late_fusion_cfg.get('classes', [''] * self.num_classes)
        self.class_dict = {i: self.classes[i] for i in range(self.num_classes)}
        self.valid_2d_classes = self.late_fusion_cfg.get('valid_2d_classes', list(range(self.num_classes)))
        self.valid_2d_classes = torch.tensor(self.valid_2d_classes, dtype=torch.int32, device=self.device)
        self.img_label_mapping = self.late_fusion_cfg.get('img_label_mapping', list(range(self.num_classes)))
        self.img_label_mapping = torch.tensor(self.img_label_mapping, dtype=torch.int32, device=self.device)
        
        self.score_thr_2d = self.late_fusion_cfg.get('score_thr_2d', [0.5] * self.num_classes)
        self.score_thr_2d = torch.tensor(self.score_thr_2d, dtype=torch.float32, device=self.device)
        self.score_thr_3d = self.late_fusion_cfg.get('score_thr_3d', [0.3] * self.num_classes)
        self.score_thr_3d = torch.tensor(self.score_thr_3d, dtype=torch.float32, device=self.device)
        self.bb_match_iou_thr = self.late_fusion_cfg.get('bbox_matching_iou_thr', 0.5)
        self.recovery_iou_thr = self.late_fusion_cfg.get('detection_recovery_iou_thr', 0.4)
        self.bb_match_mode = self.late_fusion_cfg.get('bbox_matching_mode', 'iou')
        self.min_pts_frustum = self.late_fusion_cfg.get('min_pts_frustum', 10)
        self.iou_calculator = BboxOverlaps2D()
        
        self.use_detection_recovery = self.late_fusion_cfg.get('use_detection_recovery', True)
        self.use_label_fusion = self.late_fusion_cfg.get('use_label_fusion', True)
        self.use_score_fusion = self.late_fusion_cfg.get('use_score_fusion', True)
        self.class_priors = self.late_fusion_cfg.get('class_prior', [1 / self.num_classes for _ in range(self.num_classes)])
        self.class_priors = torch.tensor(self.class_priors, dtype=torch.float32, device=self.device)
        self.use_final_nms = self.late_fusion_cfg.get('use_final_nms', True)
        self.keep_oov_bboxes = self.late_fusion_cfg.get('keep_oov_bboxes', True)
        
        self.visualization_cfg = self.late_fusion_cfg.get('visualization_cfg', {})
        self.color_dict = self.visualization_cfg.get('colors', None)
        if self.color_dict:
            color_dict = {}
            for i in range(self.num_classes):
                color_dict[i] = tuple(self.color_dict[i])
            self.color_dict = color_dict
        self.final_nms_cfg = self.late_fusion_cfg.get('final_nms_cfg', {})
        
        self.align_frustum = self.late_fusion_cfg.get('align_frustum', False)
        self.use_gaussian_likelihoods = self.late_fusion_cfg.get('use_gaussian_likelihoods', True)
        self.enlarge_factor = self.late_fusion_cfg.get('enlarge_factor', 0.05)
        
    def _init_detector3d(self, cfg_file, checkpoint):
        self.detector3d = init_model(cfg_file, checkpoint, device=self.device)

        cfg = self.detector3d.cfg

        test_pipeline_3d = deepcopy(cfg.test_dataloader.dataset.pipeline)
        self.test_pipeline_3d = ComposeMMEngine(test_pipeline_3d)
        self.box_type_3d, self.box_mode_3d = get_box_type(cfg.test_dataloader.dataset.box_type_3d)
        
    def _init_detector2d(self, cfg_file, checkpoint):
        self.detector2d = init_detector(cfg_file, checkpoint, device=self.device)

        cfg = self.detector2d.cfg.copy()
        test_pipeline_2d = get_test_pipeline_cfg(cfg)
        self.test_pipeline_2d = ComposeMMCV(test_pipeline_2d)
        
    def _init_frustum_detector(self, cfg_file, checkpoint):
        self.frustum_detector = init_model(cfg_file, checkpoint, device=self.device)

        cfg = self.frustum_detector.cfg.copy()
        test_pipeline_3d_frustum = get_test_pipeline_cfg(cfg)
        self.test_pipeline_3d_frustum = ComposeMMCV(test_pipeline_3d_frustum)
        
    def predict(self, img_file_left, img_file_right, lidar_file, 
                lidar_to_cam: Tensor = None, cam_to_img_left: Tensor = None, cam_to_img_right: Tensor = None):
        """
        Runs inference on the provided image and LiDAR files and returns the final detections.

        Args:
            img_file (str): Path to the image file.
            lidar_file (str): Path to the LiDAR file.
            lidar_to_cam (Tensor, optional): Transformation matrix from LiDAR to camera coordinates.
            cam_to_img_left (Tensor, optional): Transformation matrix from camera to left image coordinates.
            cam_to_img_right (Tensor, optional): Transformation matrix from camera to right image coordinates.

        Returns:
            InstanceData: The final detections including 3d bounding boxes, scores, and labels.
        """
        if lidar_to_cam is None:
            lidar_to_cam = self.lidar_to_cam
        if cam_to_img_left is None:
            cam_to_img_left = self.cam_to_img_left
        if cam_to_img_right is None:
            cam_to_img_right = self.cam_to_img_right
            
        bboxes_2d_left, labels_2d_left, scores_2d_left, data_samples_2d_left = self.rgb_branch_inference(img_file_left)
        bboxes_2d_right, labels_2d_right, scores_2d_right, data_samples_2d_right = self.rgb_branch_inference(img_file_right)
        bboxes_3d, corners_3d, labels_3d, scores_3d, point_cloud = self.lidar_branch_inference(lidar_file)
        
        bbox_matching_dict = self.bbox_matching(bboxes_3d, scores_3d, labels_3d, corners_3d,
                                                bboxes_2d_left, scores_2d_left, labels_2d_left, 
                                                bboxes_2d_right, scores_2d_right, labels_2d_right,
                                                data_samples_2d_left[0].img_shape, data_samples_2d_right[0].img_shape,
                                                lidar_to_cam, cam_to_img_left, cam_to_img_right)
        matching = bbox_matching_dict['matching']
        oov_detections = bbox_matching_dict['oov_bboxes']
        
        if self.use_label_fusion:
            new_labels_3d, new_scores_3d = self.semantic_fusion(**matching)
            matching['labels_3d'] = new_labels_3d
            matching['scores_3d'] = new_scores_3d
            
        if self.use_detection_recovery and self.frustum_detector is not None:
            recovery_output = self.detection_recovery(
                point_cloud, bbox_matching_dict['unmatched_rgb_left'], bbox_matching_dict['unmatched_rgb_right'], 
                lidar_to_cam=lidar_to_cam, cam_to_img_left=cam_to_img_left, cam_to_img_right=cam_to_img_right,
                img_shape_left=data_samples_2d_left[0].img_shape, img_shape_right=data_samples_2d_right[0].img_shape,
            )
            matching = self.merge_matchings(matching, recovery_output)

        bboxes_3d = matching['bboxes_3d']
        scores_3d = matching['scores_3d']
        labels_3d = matching['labels_3d']
            
        if self.keep_oov_bboxes:
            bboxes_3d = torch.cat([bboxes_3d, oov_detections['bboxes_3d']], dim=0)
            scores_3d = torch.cat([scores_3d, oov_detections['scores_3d']], dim=0)
            labels_3d = torch.cat([labels_3d, oov_detections['labels_3d']], dim=0)
        
        # return back to mmdet3d bounding boxes format
        bboxes_3d[:, 2] -= bboxes_3d[:, 5] / 2
            
        if self.use_final_nms:
            bboxes_for_nms = xywhr2xyxyr(self.box_type_3d(bboxes_3d, box_dim=7).bev)
            selected_indices = nms_bev(bboxes_for_nms, scores_3d, thresh=self.final_nms_cfg.get('thresh', 0.01),
                                       pre_max_size=self.final_nms_cfg.get('pre_max_size', None), 
                                       post_max_size=self.final_nms_cfg.get('post_max_size', None))
            bboxes_3d = bboxes_3d[selected_indices]
            scores_3d = scores_3d[selected_indices]
            labels_3d = labels_3d[selected_indices]
            
        final_detections = InstanceData()
        final_detections.bboxes_3d = self.box_type_3d(bboxes_3d)
        final_detections.scores_3d = scores_3d
        final_detections.labels_3d = labels_3d
        return final_detections
    
    def visualize_predict(self, img_file_left, img_file_right, lidar_file, save_path,
                          lidar_to_cam: Tensor = None, cam_to_img_left: Tensor = None, cam_to_img_right: Tensor = None):
        """
        Runs inference and saves visualizations of detections to the specified path.

        Args:
            img_file (str): Path to the image file.
            lidar_file (str): Path to the LiDAR file.
            save_path (Union[str, Path]): Path to save the following visualization images
                - rgb_detections.png: image containing the 2d detections from the rgb branch
                - lidar_detections.png: image containing the 3d detections from the lidar branch
                - late_fusion_detections.png: image containing the final 3d detections
                The visualizations do not contain the bounding boxes that are out of view in the image plane.
            lidar_to_cam (Tensor, optional): Transformation matrix from LiDAR to camera coordinates.
            cam_to_img (Tensor, optional): Transformation matrix from camera to image coordinates.

        Returns:
            InstanceData: The final detections including bounding boxes, scores, and labels.
        """
        if lidar_to_cam is None:
            lidar_to_cam = self.lidar_to_cam
        if cam_to_img_left is None:
            cam_to_img_left = self.cam_to_img_left
        if cam_to_img_right is None:
            cam_to_img_right = self.cam_to_img_right
        if isinstance(save_path, str):
            save_path = Path(save_path)
            
        bboxes_2d_left, labels_2d_left, scores_2d_left, data_samples_2d_left = self.rgb_branch_inference(img_file_left)
        bboxes_2d_right, labels_2d_right, scores_2d_right, data_samples_2d_right = self.rgb_branch_inference(img_file_right)
        bboxes_3d, corners_3d, labels_3d, scores_3d, point_cloud = self.lidar_branch_inference(lidar_file)
        
        image_left = np.array(Image.open(img_file_left)) 
        save_path_faster_rcnn = save_path / 'rgb_detections_left.png'
        draw_bboxes_2d(image_left, bboxes_2d_left.clone().cpu().numpy(), labels_2d_left.clone().cpu().numpy(), self.class_dict,
                       scores_2d_left.clone().cpu().numpy(), save_path_faster_rcnn, self.color_dict,
                       fill=self.visualization_cfg.get('fill_bboxes_2d', True), alpha=self.visualization_cfg.get('alpha', 50))
        
        image_right = np.array(Image.open(img_file_right)) 
        save_path_faster_rcnn = save_path / 'rgb_detections_right.png'
        draw_bboxes_2d(image_right, bboxes_2d_right.clone().cpu().numpy(), labels_2d_right.clone().cpu().numpy(), self.class_dict,
                       scores_2d_right.clone().cpu().numpy(), save_path_faster_rcnn, self.color_dict,
                       fill=self.visualization_cfg.get('fill_bboxes_2d', True), alpha=self.visualization_cfg.get('alpha', 50))
        
        bboxes_3d_mmdet = bboxes_3d.clone().cpu()
        bboxes_3d_mmdet[:, 2] -= bboxes_3d_mmdet[:, 5] / 2
        save_path_detector3d = save_path / 'lidar_detections_left_proj.png'
        draw_bboxes_3d_image(image_left, bboxes_3d_mmdet, labels_3d.clone().cpu(), cam_to_img_left.clone().cpu(), 
                             lidar_to_cam.clone().cpu(), self.color_dict, True, save_path_detector3d)
        
        bbox_matching_dict = self.bbox_matching(bboxes_3d, scores_3d, labels_3d, corners_3d,
                                                bboxes_2d_left, scores_2d_left, labels_2d_left, 
                                                bboxes_2d_right, scores_2d_right, labels_2d_right,
                                                data_samples_2d_left[0].img_shape, data_samples_2d_right[0].img_shape,
                                                lidar_to_cam, cam_to_img_left, cam_to_img_right)
        matching = bbox_matching_dict['matching']
        oov_detections = bbox_matching_dict['oov_bboxes']
            
        if self.use_label_fusion:
            new_labels_3d, new_scores_3d = self.semantic_fusion(**matching)
            matching['labels_3d'] = new_labels_3d
            matching['scores_3d'] = new_scores_3d
            
        if self.use_detection_recovery and self.frustum_detector is not None:
            recovery_output = self.detection_recovery(
                point_cloud, bbox_matching_dict['unmatched_rgb_left'], bbox_matching_dict['unmatched_rgb_right'], 
                lidar_to_cam=lidar_to_cam, cam_to_img_left=cam_to_img_left, cam_to_img_right=cam_to_img_right,
                img_shape_left=data_samples_2d_left[0].img_shape, img_shape_right=data_samples_2d_right[0].img_shape,
            )
            matching = self.merge_matchings(matching, recovery_output)
            
        # return back to mmdet3d bounding boxes format
        bboxes_3d = torch.cat([matching['bboxes_3d'], oov_detections['bboxes_3d']], dim=0)
        bboxes_3d[:, 2] -= bboxes_3d[:, 5] / 2
        scores_3d = torch.cat([matching['scores_3d'], oov_detections['scores_3d']], dim=0)
        labels_3d = torch.cat([matching['labels_3d'], oov_detections['labels_3d']], dim=0)
            
        bboxes_for_nms = xywhr2xyxyr(self.box_type_3d(bboxes_3d, box_dim=7).bev)
        selected_indices = nms_bev(bboxes_for_nms, scores_3d, thresh=self.final_nms_cfg.get('thresh', 0.01),
                                   pre_max_size=self.final_nms_cfg.get('pre_max_size', None), 
                                   post_max_size=self.final_nms_cfg.get('post_max_size', None))
            
        final_detections = InstanceData()
        final_detections.bboxes_3d = self.box_type_3d(bboxes_3d[selected_indices])
        final_detections.scores_3d = scores_3d[selected_indices]
        final_detections.labels_3d = labels_3d[selected_indices]
        
        save_path_final = save_path / 'fusion_detections_left_proj.png'
        draw_bboxes_3d_image(image_left, final_detections.bboxes_3d.clone().cpu(), final_detections.labels_3d.clone().cpu(), 
                             cam_to_img_left.clone().cpu(), lidar_to_cam.clone().cpu(), self.color_dict, True, save_path_final)
        
        # save_path_3d_final = save_path / 'fusion_detections.png'
        # show_bboxes_3d(lidar_file, lidar_to_cam, final_detections.bboxes_3d.clone().cpu(), final_detections.labels_3d.clone().cpu().numpy(), 
        #                self.color_dict, save_path=save_path_3d_final, save_capture=True, show_window=False, adjust_position=True, fill_points_inside=True)
        
        return final_detections   
        
    def rgb_branch_inference(self, img_file: str) -> Tuple[Tensor, Tensor, Tensor, Tensor, List]: 
        inputs = dict(img_path=img_file, img_id=0)
        inputs = self.test_pipeline_2d(inputs)
        inputs['inputs'] = [inputs['inputs']]
        inputs['data_samples'] = [inputs['data_samples']]
        
        with torch.no_grad():
            results = self.detector2d.test_step(inputs)[0]
        
        labels = results.pred_instances.labels
        scores = results.pred_instances.scores
        valid_boxes = torch.isin(labels, self.valid_2d_classes)
        scores = scores[valid_boxes]
        bboxes = results.pred_instances.bboxes[valid_boxes]
        labels = labels[valid_boxes]
        labels = self.img_label_mapping[labels].long()

        filter_scores = scores >= self.score_thr_2d[labels]
        bboxes = bboxes[filter_scores]
        labels = labels[filter_scores]
        scores = scores[filter_scores]
        
        return bboxes, labels, scores, inputs['data_samples']
    
    def lidar_branch_inference(self, pc_file: str) -> Tuple[Tensor, Tensor, Tensor]:
        inputs = dict(
            lidar_points=dict(lidar_path=pc_file),
            timestamp=1,
            axis_align_matrix=np.eye(4),
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d)

        collate_data = pseudo_collate([self.test_pipeline_3d(inputs)])
        
        with torch.no_grad():
            detection_output = self.detector3d.test_step(collate_data)[0]

        bboxes = detection_output.pred_instances_3d.bboxes_3d.tensor
        corners = detection_output.pred_instances_3d.bboxes_3d.corners
        # move the returned (x,y,z) to the actual box center
        bboxes[:, 2] += bboxes[:, 5] / 2
        labels = detection_output.pred_instances_3d.labels_3d.long()
        scores = detection_output.pred_instances_3d.scores_3d
        
        filter_scores = scores >= self.score_thr_3d[labels]
        return bboxes[filter_scores], corners[filter_scores], labels[filter_scores], scores[filter_scores], collate_data
    
    def bbox_matching(self, bboxes_3d, scores_3d, labels_3d, corners_3d, bboxes_2d_left, 
                      scores_2d_left, labels_2d_left, bboxes_2d_right, 
                      scores_2d_right, labels_2d_right, img_shape_left, 
                      img_shape_right, lidar_to_cam, cam_to_img_left, cam_to_img_right):
        
        # projecting the bounding boxes in the images and keeping the ones that are inside at least one of them
        bboxes_proj_left = corners_to_axis_aligned_bbox(corners_3d, P=cam_to_img_left, lidar=True, T=lidar_to_cam)
        bboxes_proj_left = clamp_boxes(bboxes_proj_left, img_shape_left)
        inside_image_left = (bboxes_proj_left[:, 0] < bboxes_proj_left[:, 2]) & (bboxes_proj_left[:, 1] < bboxes_proj_left[:, 3])
        bboxes_proj_right = corners_to_axis_aligned_bbox(corners_3d, P=cam_to_img_right, lidar=True, T=lidar_to_cam)
        bboxes_proj_right = clamp_boxes(bboxes_proj_right, img_shape_right)
        inside_image_right = (bboxes_proj_right[:, 0] < bboxes_proj_right[:, 2]) & (bboxes_proj_right[:, 1] < bboxes_proj_right[:, 3])
        keep_boxes = inside_image_left | inside_image_right
        
        bboxes_3d_valid = bboxes_3d[keep_boxes]
        scores_3d_valid = scores_3d[keep_boxes]
        labels_3d_valid = labels_3d[keep_boxes]
        bboxes_proj_left_valid = bboxes_proj_left[keep_boxes]
        bboxes_proj_right_valid = bboxes_proj_right[keep_boxes]
        left_assignment = match_bboxes_linear_sum_assign(self.iou_calculator, bboxes_proj_left_valid, bboxes_2d_left, mode='iou', iou_thr=self.bb_match_iou_thr)
        right_assignment = match_bboxes_linear_sum_assign(self.iou_calculator, bboxes_proj_right_valid, bboxes_2d_right, mode='iou', iou_thr=self.bb_match_iou_thr)
        
        matches_mask = torch.where((left_assignment >= 1) | (right_assignment >= 1), True, False)
        both_matches = torch.where((left_assignment >= 1) & (right_assignment >= 1), True, False)[matches_mask]
        right_matches = torch.where(right_assignment >= 1, True, False)[matches_mask]
        left_matches = torch.where(left_assignment >= 1, True, False)[matches_mask]
        left_indices = torch.clamp(left_assignment - 1, min=-1)[matches_mask]
        right_indices = torch.clamp(right_assignment - 1, min=-1)[matches_mask]
        num_matches = matches_mask.sum()
        matching = {
            'bboxes_3d': bboxes_3d_valid[matches_mask], 
            'scores_3d': scores_3d_valid[matches_mask], 
            'labels_3d': labels_3d_valid[matches_mask],
            'bboxes_2d_left': bboxes_2d_left[left_indices] if bboxes_2d_left.shape[0] > 0 else bboxes_2d_left.new_tensor([[-1] * 7] * num_matches),
            'scores_2d_left': scores_2d_left[left_indices] if bboxes_2d_left.shape[0] > 0 else scores_2d_left.new_tensor([0.0] * num_matches),
            'labels_2d_left': labels_2d_left[left_indices] if bboxes_2d_left.shape[0] > 0 else labels_2d_left.new_tensor([-1] * num_matches),
            'bboxes_2d_right': bboxes_2d_right[right_indices] if bboxes_2d_right.shape[0] > 0 else bboxes_2d_right.new_tensor([[-1] * 7] * num_matches),
            'scores_2d_right': scores_2d_right[right_indices] if bboxes_2d_right.shape[0] > 0 else scores_2d_right.new_tensor([0.0] * num_matches),
            'labels_2d_right': labels_2d_right[right_indices] if bboxes_2d_right.shape[0] > 0 else labels_2d_right.new_tensor([-1] * num_matches),
            'both_matches_mask': both_matches,
            'left_matches_mask': left_matches,
            'right_matches_mask': right_matches
        }
        
        unmatched_mask_left = torch.isin(torch.arange(bboxes_2d_left.shape[0]).to(bboxes_proj_left.device), left_assignment - 1, invert=True)
        unmatched_rgb_left = {
            'bboxes_2d': bboxes_2d_left[unmatched_mask_left],
            'scores_2d': scores_2d_left[unmatched_mask_left],
            'labels_2d': labels_2d_left[unmatched_mask_left],
        }
        
        unmatched_mask_right = torch.isin(torch.arange(bboxes_2d_right.shape[0]).to(bboxes_proj_left.device), right_assignment - 1, invert=True)
        unmatched_rgb_right = {
            'bboxes_2d': bboxes_2d_right[unmatched_mask_right],
            'scores_2d': scores_2d_right[unmatched_mask_right],
            'labels_2d': labels_2d_right[unmatched_mask_right],
        }
        
        out_of_view_bboxes = {
            'bboxes_3d': bboxes_3d[~keep_boxes],
            'scores_3d': scores_3d[~keep_boxes],
            'labels_3d': labels_3d[~keep_boxes],
        }
        return dict(matching=matching, unmatched_rgb_left=unmatched_rgb_left, 
                    unmatched_rgb_right=unmatched_rgb_right, oov_bboxes=out_of_view_bboxes)
    
    def detection_recovery(self, collate_data: dict, unmatched_left: dict[str, Tensor], unmatched_right: dict[str, Tensor], 
                           lidar_to_cam: Tensor, cam_to_img_left: Tensor, cam_to_img_right: Tensor, 
                           img_shape_left: Tuple, img_shape_right: Tuple, **kwargs):
        
        F = compute_fundamental_matrix(cam_to_img_left.cpu().numpy(), cam_to_img_right.cpu().numpy())
        F = torch.tensor(F, device=unmatched_left['bboxes_2d'].device, dtype=torch.float32)
        left_ids_2d, right_ids_2d = assign_with_epipolar_lines(unmatched_left['bboxes_2d'], unmatched_right['bboxes_2d'], 
                                                               unmatched_left['labels_2d'], unmatched_right['labels_2d'], F)
        
        bboxes_left_enlarge = enlarge_bboxes_2d(unmatched_left['bboxes_2d'][left_ids_2d].clone(), 0.05, 0.05)
        bboxes_right_enlarge = enlarge_bboxes_2d(unmatched_right['bboxes_2d'][right_ids_2d].clone(), 0.05, 0.05)
        ori_left = unmatched_left['bboxes_2d'][left_ids_2d].cpu().numpy()
        ori_right = unmatched_right['bboxes_2d'][right_ids_2d].cpu().numpy()
        
        scores = torch.cat([unmatched_left['scores_2d'][left_ids_2d].unsqueeze(1), unmatched_right['scores_2d'][right_ids_2d].unsqueeze(1)], dim=1)
        _, indices = torch.max(scores, dim=1)
        scores_2d = torch.min(scores, dim=1)[0]
        lables_2d_cat = torch.cat([unmatched_left['labels_2d'][left_ids_2d].unsqueeze(1), unmatched_right['labels_2d'][right_ids_2d].unsqueeze(1)], dim=1)
        best_labels_2d = torch.gather(lables_2d_cat, 1, indices.unsqueeze(1)).squeeze(1)
        one_hot_vectors = torch.nn.functional.one_hot(best_labels_2d, num_classes=self.valid_2d_classes.shape[0])
        
        scan = collate_data['inputs']['points'][0].to(lidar_to_cam.device)
        scan = scan[scan[:, 0] > 0]
        points = scan[:, 0:3]
        velo = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
        camera_coord = lidar_to_cam.matmul(velo.t())
        
        coord_left = cam_to_img_left.matmul(camera_coord)
        coord_left[:2] /= coord_left[2,:]
        coord_left[2] = 1
        coord_left = coord_left.T
        coord_right = cam_to_img_right.matmul(camera_coord)
        coord_right[:2] /= coord_right[2,:]
        coord_right[2] = 1
        coord_right = coord_right.T
        
        if self.align_frustum:
            K_inv_left = np.linalg.inv(cam_to_img_left.cpu().numpy()[:, :3])
            K_inv_right = np.linalg.inv(cam_to_img_right.cpu().numpy()[:, :3])
            cam_to_lidar = np.linalg.inv(lidar_to_cam.cpu().numpy())
        
        new_bboxes_3d = []
        new_scores_3d = []
        new_labels_3d = []
        indices_2d = []
        rt_matrices = []
        yaw_angles = []
        frustum_proposals = {'inputs': {'points': []}, 'data_samples': []} #[]
        for i, (left_box, right_box) in enumerate(zip(bboxes_left_enlarge.cpu().numpy(), bboxes_right_enlarge.cpu().numpy())):
            # frustum == points that project inside the bounding box
            # points must project inside both bounding boxes
            fov_inds = (coord_left[:, 0] <= left_box[2]) & \
                (coord_left[:, 0] >= left_box[0]) & \
                (coord_left[:, 1] <= left_box[3]) & \
                (coord_left[:, 1] >= left_box[1]) & \
                (coord_right[:, 0] <= right_box[2]) & \
                (coord_right[:, 0] >= right_box[0]) & \
                (coord_right[:, 1] <= right_box[3]) & \
                (coord_right[:, 1] >= right_box[1])
            
            if torch.sum(fov_inds) > self.min_pts_frustum:
                indices_2d.append(i)
                data_sample = collate_data['data_samples'][0]
                metainfo = data_sample.metainfo
                metainfo['one_hot_vector'] = one_hot_vectors[i, :]
                data_sample.set_metainfo(metainfo)
                
                frustum_pc = scan[fov_inds, :].clone()
                if self.use_gaussian_likelihoods:
                    wl, hl = ori_left[i, 2] - ori_left[i, 0], ori_left[i, 3] - ori_left[i, 1]
                    xl, yl = ori_left[i, 0] + wl/2, ori_left[i, 1] + hl/2
                    left_likelihood = torch.exp(
                        -((coord_left[fov_inds, 0] - xl)**2 / (2 * wl**2)) - ((coord_left[fov_inds, 1] - yl)**2 / (2 * hl**2)))
                    
                    wr, hr = ori_right[i, 2] - ori_right[i, 0], ori_right[i, 3] - ori_right[i, 1]
                    xr, yr = ori_right[i, 0] + wr/2, ori_right[i, 1] + hr/2
                    right_likelihood = torch.exp(
                        -((coord_right[fov_inds, 0] - xr)**2 / (2 * wr**2)) - ((coord_right[fov_inds, 1] - yr)**2 / (2 * hr**2)))
                    
                    likelihoods = torch.maximum(left_likelihood, right_likelihood)
                    frustum_pc = torch.cat([frustum_pc, likelihoods.unsqueeze(1)], dim=-1)
                
                if self.align_frustum:
                    left_center = np.concatenate([(left_box[2:4] + left_box[0:2]) / 2, [1]])[:, np.newaxis]
                    right_center = np.concatenate([(right_box[2:4] + right_box[0:2]) / 2, [1]])[:, np.newaxis]
                    
                    # backprojection == point at the infinity -> append 0 as last coordinate
                    left_backproj = np.append(K_inv_left.dot(left_center).flatten(), [0])[np.newaxis, :]
                    right_backproj = np.append(K_inv_right.dot(right_center).flatten(), [0])[np.newaxis, :]
                    
                    backprojections = np.vstack([left_backproj, right_backproj])
                    backprojections = cam_to_lidar.dot(backprojections.T).T
                    yaw_lidar = np.arctan2(backprojections[:, 1], backprojections[:, 0])[0]
                    
                    rotation_matrix = scan.new_tensor([
                        [np.cos(-yaw_lidar), -np.sin(-yaw_lidar), 0],
                        [np.sin(-yaw_lidar), np.cos(-yaw_lidar), 0],
                        [0, 0, 1],
                    ])
                    frustum_pc[:, :3] = rotation_matrix.matmul(frustum_pc[:, :3].t()).t()
                    rt_matrices.append(rotation_matrix)
                    yaw_angles.append(yaw_lidar)
                frustum_proposals['inputs']['points'].append(frustum_pc)
                frustum_proposals['data_samples'].append(data_sample)
        
        if len(indices_2d) > 0:
            bboxes_3d = []
            scores_3d = []
            labels_3d = []
            for i in range(len(indices_2d)):
                proposal = {'data_samples': [frustum_proposals['data_samples'][i]],
                            'inputs': {'points': [frustum_proposals['inputs']['points'][i]]}}
                with torch.no_grad():
                    detection_output = self.frustum_detector.test_step(proposal)
                
                if len(detection_output) == 4:
                    _, new_bboxes_3d, new_scores_3d, new_labels_3d = detection_output
                    scores_3d.append(new_scores_3d)
                    labels_3d.append(new_labels_3d)
                if len(detection_output) == 3:
                    _, new_bboxes_3d, new_scores_3d = detection_output
                    scores_3d.append(new_scores_3d)
                else:
                    _, new_bboxes_3d = detection_output
                if self.align_frustum:
                    new_bboxes_3d[:, :3] = torch.linalg.inv(rt_matrices[i]).matmul(new_bboxes_3d[:, :3].t()).t()
                    new_bboxes_3d[:, -1] += yaw_angles[i]
                bboxes_3d.append(new_bboxes_3d.cpu())
                    
            new_bboxes_3d = torch.cat(bboxes_3d, dim=0)
            new_bboxes_3d[:, 2] += new_bboxes_3d[:, 5] / 2
            if len(scores_3d) > 0:
                new_scores_3d = torch.cat(scores_3d, dim=0).squeeze(1)
            elif self.use_label_fusion:
                new_scores_3d = -new_bboxes_3d.new_ones((new_bboxes_3d.shape[0],), dtype=torch.float32)     
            else:
                new_scores_3d = scores_2d[indices_2d]
            if len(labels_3d) > 0:
                new_labels_3d = torch.cat(labels_3d, dim=0).squeeze(1)
            elif self.use_label_fusion:
                new_labels_3d = -new_bboxes_3d.new_ones((new_bboxes_3d.shape[0],), dtype=torch.long)
            else:
                new_labels_3d = best_labels_2d[indices_2d]
            new_left_boxes = unmatched_left['bboxes_2d'][left_ids_2d][indices_2d]
            new_right_boxes = unmatched_right['bboxes_2d'][right_ids_2d][indices_2d]
            
            # if the 3d bounding boxes are not consistent with the 2d ones, they are filtered
            bboxes_proj_left = project_bboxes(new_bboxes_3d, P=cam_to_img_left.cpu(), lidar=True, T=lidar_to_cam.cpu())
            bboxes_proj_right = project_bboxes(new_bboxes_3d, P=cam_to_img_right.cpu(), lidar=True, T=lidar_to_cam.cpu())
            ious_left = self.iou_calculator(bboxes_proj_left.cpu(), new_left_boxes.cpu(), is_aligned=True)
            ious_right = self.iou_calculator(bboxes_proj_right.cpu(), new_right_boxes.cpu(), is_aligned=True)
            iou_max = torch.maximum(ious_left, ious_right)
            iou_filter = iou_max > self.recovery_iou_thr
            
            new_bboxes_3d = new_bboxes_3d[iou_filter]
            new_scores_3d = new_scores_3d[iou_filter]
            new_labels_3d = new_labels_3d[iou_filter]
            new_left_boxes = new_left_boxes[iou_filter]
            new_right_boxes = new_right_boxes[iou_filter]
            
            num_valid = torch.sum(iou_filter)
            if self.use_label_fusion and num_valid > 0:
                new_labels_3d, new_scores_3d = self.semantic_fusion(
                    new_scores_3d.to(self.device), new_labels_3d.to(self.device), 
                    unmatched_left['scores_2d'][left_ids_2d][indices_2d][iou_filter].to(self.device), 
                    unmatched_left['labels_2d'][left_ids_2d][indices_2d][iou_filter].to(self.device), 
                    unmatched_right['scores_2d'][right_ids_2d][indices_2d][iou_filter].to(self.device), 
                    unmatched_right['labels_2d'][right_ids_2d][indices_2d][iou_filter].to(self.device),
                    torch.tensor([True] * new_bboxes_3d.shape[0], dtype=torch.bool), 
                    torch.tensor([True] * new_bboxes_3d.shape[0], dtype=torch.bool),
                    torch.tensor([True] * new_bboxes_3d.shape[0], dtype=torch.bool),
                ) 
                new_scores_3d = new_scores_3d * ious_left[iou_filter].to(self.device) * ious_right[iou_filter].to(self.device)
            elif num_valid > 0:
                new_scores_3d = new_scores_3d * ious_left[iou_filter].to(self.device) * ious_right[iou_filter].to(self.device)
        else:
            new_bboxes_3d = unmatched_left['bboxes_2d'].new_empty((0, 7), dtype=torch.float32)
            new_scores_3d = unmatched_left['bboxes_2d'].new_empty((0,), dtype=torch.float32)
            new_labels_3d = unmatched_left['bboxes_2d'].new_empty((0,), dtype=torch.long)
        
        return {
            'bboxes_3d': new_bboxes_3d.to(self.device), 
            'scores_3d': new_scores_3d.to(self.device), 
            'labels_3d': new_labels_3d.to(self.device)
        }
        
    def merge_matchings(self, matching1: Dict[str, Tensor], matching2: Dict[str, Tensor], **kwargs):
        common_keys = list(set(matching1.keys()) & set(matching2.keys()))
        return {key: torch.cat([matching1[key], matching2[key]], dim=0) for key in common_keys}
    
    def semantic_fusion(self, scores_3d: Tensor, labels_3d: Tensor, scores_2d_left: Tensor, labels_2d_left: Tensor,
                        scores_2d_right: Tensor, labels_2d_right: Tensor, left_matches_mask: Tensor, 
                        right_matches_mask: Tensor, both_matches_mask: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        same_left = labels_3d == labels_2d_left
        same_right = labels_3d == labels_2d_right
        same_images = labels_2d_left == labels_2d_right
        
        mask = left_matches_mask & ~right_matches_mask
        if torch.sum(mask) > 0:
            fused_scores_left = self.semantic_fusion_single(scores_3d[mask], labels_3d[mask], 
                                                            scores_2d_left[mask], labels_2d_left[mask])
            labels_3d[mask] = fused_scores_left[0]
            scores_3d[mask] = fused_scores_left[1]
        
        mask = right_matches_mask & ~left_matches_mask
        if torch.sum(mask) > 0:
            fused_scores_right = self.semantic_fusion_single(scores_3d[mask], labels_3d[mask],
                                                             scores_2d_right[mask], labels_2d_right[mask])
            labels_3d[mask] = fused_scores_right[0]
            scores_3d[mask] = fused_scores_right[1]
        
        both_matches_mask = both_matches_mask.to(scores_3d.device)
        if self.use_score_fusion:
            # same labels on all views
            mask = both_matches_mask & same_left & same_right
            scores_3d[mask] = (scores_2d_left[mask] * scores_2d_right[mask] * scores_3d[mask]) / self.class_priors[labels_3d[mask]]**2
            
            # same labels only on the left image -> taking only its score
            mask = both_matches_mask & same_left & ~same_right
            scores_3d[mask] = (scores_2d_left[mask] * scores_3d[mask]) / self.class_priors[labels_2d_left[mask]]
                
            # same labels only on the right image -> taking only its score
            mask = both_matches_mask & ~same_left & same_right
            scores_3d[mask] = (scores_2d_right[mask] * scores_3d[mask]) / self.class_priors[labels_2d_right[mask]]
            
        # different labels between 3d and 2d, same between both images
        mask = both_matches_mask & ~same_left & ~same_right & same_images
        labels_3d[mask] = labels_2d_left[mask]
        if self.use_score_fusion:
            scores_3d[mask] = (scores_2d_left[mask] * scores_2d_right[mask]) / self.class_priors[labels_2d_left[mask]]
        
        # different labels in all -> taking the most confident image
        mask = both_matches_mask & ~same_left & ~same_right & ~same_images
        if torch.sum(mask) > 0:
            scores_2d, indices = torch.max(torch.cat([scores_2d_left[mask].unsqueeze(1), 
                                                    scores_2d_right[mask].unsqueeze(1)], dim=1), dim=1)
            lables_2d_cat = torch.cat([labels_2d_left[mask].unsqueeze(1), 
                                    labels_2d_right[mask].unsqueeze(1)], dim=1)
            labels_3d[mask] = torch.gather(lables_2d_cat, 1, indices.unsqueeze(1)).squeeze(1)
            if self.use_score_fusion:
                scores_3d[mask] = scores_2d
        
        return labels_3d, scores_3d
        

    def semantic_fusion_single(self, scores_3d: Tensor, labels_3d: Tensor, scores_2d: Tensor, labels_2d: Tensor):
        different_labels = labels_3d != labels_2d
        
        new_labels_3d = labels_3d.clone()
        new_labels_3d[different_labels] = labels_2d[different_labels]
        
        new_scores_3d = scores_3d.clone()
        if self.use_score_fusion:
            new_scores_3d[different_labels] = scores_2d[different_labels]
            new_scores_3d[~different_labels] = (new_scores_3d[~different_labels] * scores_2d[~different_labels]) / self.class_priors[labels_2d[~different_labels]]
        return new_labels_3d, new_scores_3d