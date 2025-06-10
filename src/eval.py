import sys
sys.path.append("/home/it4i-carlos00/3d_object_detection")

import numpy as np
import torch
from pathlib import Path
from PIL import Image
import os
import argparse
from copy import deepcopy
import time 
import json 
import logging
import pickle
from tqdm import tqdm

from utils import read_kitti_calibration_data, calibration_to_torch
from inference import LateFusionStereoViewInferencer

from mmengine.logging import print_log
from mmengine.dataset import Compose as ComposeMMengine
from mmdet3d.evaluation import KittiMetric

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


if __name__ == '__main__':
    class_dict_pointpillars = {
        0: 'Pedestrian',
        1: 'Cyclist',
        2: 'Car',
    }
    
    parser = argparse.ArgumentParser(
        description="Inference script for expert late fusion")
    parser.add_argument("-output_dir", default="/mnt/it4i-carlos00/experiments/expert_late_fusion_predictions/data", type=str,
                        help="The directory where the models will be placed")
    parser.add_argument("-kitti_root_path", default=None, type=str, required=True,
                        help="Directory where the dataset is placed")
    parser.add_argument("--use_frustum_detector", default=False, action='store_true',
                        help="Whether to use the specialized detector on the frustum crops")
    parser.add_argument("--test", default=False, action='store_true',
                        help="Whether to evaluate or not the method")
    parser.add_argument("-validation_split_path", 
                        default='/home/it4i-carlos00/3d_object_detection/src/expert_late_fusion/val.txt', 
                        type=str, required=False,
                        help="Pointpillars checkpoint path")
    parser.add_argument("-annotation_file_eval", 
                        default='/mnt/proj2/dd-24-8/kitti_mmdet3d/kitti_infos_val.pkl', 
                        type=str, required=False,
                        help="Annotation file to evaluate with mmdet3d KittiMetric")
    parser.add_argument("--save_predictions", default=False, action='store_true',
                        help="Whether to align or not the frustum point cloud")
    parser.add_argument("-late_fusion_config", default=None, type=str, required=True,
                        help="Late Fusion config path")
    parser.add_argument("-device", default='cpu', type=str, required=False,
                        help="Device to be used (e.g. cpu or cuda:0)")

    args = parser.parse_args()
    kitti_root_path = Path(args.kitti_root_path)
    label_path = kitti_root_path / 'label_2'
    left_path = kitti_root_path / 'image_2'
    right_path = kitti_root_path / 'image_3'
    velo_path = kitti_root_path / 'velodyne_reduced'
    calib_path = kitti_root_path / 'calib'
    planes_path = kitti_root_path / 'planes'

    output_dir = Path(args.output_dir)
    predictions_path = output_dir / 'data'
    metrics_path = output_dir / 'metrics.json'
    
    predictions_path.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(filename=str(output_dir / 'log.txt'), filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    inferencer = LateFusionStereoViewInferencer(
        late_fusion_cfg=args.late_fusion_config,
        device=args.device
    )

    with open(args.validation_split_path, 'r') as split_file:
        validation_ids = split_file.readlines()
        
    validation_ids = [val_id.rstrip('\n') for val_id in validation_ids]         
    results_list = []
    initial_time = time.time()

    for i, val_id in enumerate(tqdm(validation_ids)):
        if i % 100 == 0:
            logging.info(f'Progress: {100 * i / len(validation_ids):.2f}%, elapsed {int(time.time() - initial_time)}s')
        left_image = str(left_path / f'{val_id}.png')
        right_image = str(right_path / f'{val_id}.png')
        velo_pc = str(velo_path / f'{val_id}.bin')
        calibration_data = read_kitti_calibration_data(calib_path / f'{val_id}.txt')
        calibration_data = calibration_to_torch(calibration_data)
        
        lidar_to_cam = calibration_data['R0_rect'] @ calibration_data['Tr_velo_to_cam']
        cam_to_img_left = calibration_data['P2']
        cam_to_img_right = calibration_data['P3']
        
        detection_out = inferencer.predict(left_image, right_image, velo_pc,
                                           lidar_to_cam, cam_to_img_left, cam_to_img_right)
        
        result = {
            'pred_instances_3d': {
                'bboxes_3d': detection_out.bboxes_3d.to('cpu'),
                'scores_3d': detection_out.scores_3d.to('cpu'),
                'labels_3d': detection_out.labels_3d.to('cpu'),
            },
            'sample_idx': i
        }
        results_list.append(result)
                   
    logging.info('Starting evaluation')
    
    kitti_metric = KittiMetric(
        ann_file=args.annotation_file_eval,
        metric=inferencer.detector3d.cfg.val_evaluator.get('metric', 'bbox'),
        backend_args=inferencer.detector3d.cfg.val_evaluator.get('backend_args', None),
        submission_prefix=str(predictions_path) if args.save_predictions else None,
        format_only=args.test
    )
    kitti_metric._dataset_meta = inferencer.detector3d.dataset_meta
    metrics_dict = kitti_metric.compute_metrics(results_list)
    
    with open(metrics_path, 'w') as metrics_file:
        json.dump(metrics_dict, metrics_file, indent=4)
        
    logging.info(f'Results: {metrics_dict}')
    
    with open(output_dir / 'predictions.pkl', 'wb') as fp:
        pickle.dump(results_list, fp)
